"""Policy boundary for safe dashboard access to the established replay orchestrator."""

import hmac
import threading
import time
import uuid
from typing import Protocol

import httpx
from sqlalchemy.orm import Session, sessionmaker

from turbine_guard.api.schemas.dashboard import (
    ReplayActionRequest,
    ReplayActionResponse,
    ReplayRunResponse,
    ReplayStatusResponse,
)
from turbine_guard.config.settings import Settings
from turbine_guard.database.enums import ReplayMode, ReplayRunStatus
from turbine_guard.replay.client import ReplayClientConfig, ReplayIngestionClient
from turbine_guard.replay.engine import ReplayEngineConfig, ReplayOrchestrator, ReplayStatusReport
from turbine_guard.replay.errors import ReplayError
from turbine_guard.replay.source import ReplaySource, ReplaySourceConfig
from turbine_guard.replay.state import PostgresReplayStateStore, ReplayRunState
from turbine_guard.services.errors import (
    ReplayControlConflictError,
    ReplayControlDisabledError,
    ReplayControlForbiddenError,
    ReplayControlRateLimitedError,
    RequestParameterError,
)


class ReplayOrchestratorContract(Protocol):
    def start(
        self,
        source_asset_id: int,
        *,
        mode: ReplayMode = ReplayMode.STEP,
        cycle_delay_seconds: float | None = None,
        force_restart: bool = False,
    ) -> ReplayRunState: ...

    def step(self, run_id: uuid.UUID) -> ReplayRunState: ...
    def resume(self, run_id: uuid.UUID, *, max_cycles: int | None = None) -> ReplayRunState: ...
    def drive(self, run_id: uuid.UUID, *, max_cycles: int | None = None) -> ReplayRunState: ...
    def stop(self, run_id: uuid.UUID) -> ReplayRunState: ...
    def status(self, run_id: uuid.UUID) -> ReplayStatusReport: ...
    def list_status(self, *, limit: int = 100) -> list[ReplayStatusReport]: ...
    def replay_asset_ids(self) -> tuple[int, ...]: ...


class ReplayControlService:
    """Enforce demo selection, authorization, limits, cooldowns, and reset confirmation."""

    def __init__(
        self,
        orchestrator: ReplayOrchestratorContract,
        settings: Settings,
        *,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._settings = settings
        self._http_client = http_client
        self._last_action: dict[str, float] = {}
        self._lock = threading.Lock()

    @classmethod
    def create(cls, sessions: sessionmaker[Session], settings: Settings) -> "ReplayControlService":
        http = httpx.Client(
            base_url=settings.replay_api_base_url,
            timeout=settings.replay_http_timeout_seconds,
        )
        orchestrator = ReplayOrchestrator(
            PostgresReplayStateStore(sessions),
            ReplaySource(ReplaySourceConfig(settings.data_dir)),
            ReplayIngestionClient(
                http,
                ReplayClientConfig(
                    max_attempts=settings.replay_max_send_attempts,
                    backoff_seconds=settings.replay_retry_backoff_seconds,
                ),
            ),
            ReplayEngineConfig(
                lease_seconds=float(settings.replay_lease_seconds),
                default_cycle_delay_seconds=settings.replay_cycle_delay_seconds,
                simulated_cycle_duration_seconds=(settings.replay_simulated_cycle_duration_seconds),
            ),
        )
        return cls(orchestrator, settings, http_client=http)

    def close(self) -> None:
        if self._http_client is not None:
            self._http_client.close()

    def status(self, *, limit: int = 20) -> ReplayStatusResponse:
        try:
            source_ids = list(self._orchestrator.replay_asset_ids())
            reports = self._orchestrator.list_status(limit=limit)
        except ReplayError:
            source_ids = []
            reports = []
        allowed = self._allowed_ids(source_ids)
        return ReplayStatusResponse(
            enabled=self._settings.replay_controls_enabled,
            writable=self._settings.replay_controls_enabled,
            public_demo_mode=self._settings.public_demo_mode,
            allowed_source_asset_ids=allowed,
            restrictions=self._restrictions(),
            runs=[_run_response(report.run) for report in reports],
        )

    def perform(self, request: ReplayActionRequest, *, client_id: str) -> ReplayActionResponse:
        self._authorize(request.control_token)
        self._rate_limit(client_id)
        try:
            if request.action == "start":
                source_id = self._required_source(request.source_asset_id)
                self._check_attempt_limit(source_id, force=False)
                state = self._orchestrator.start(source_id, mode=ReplayMode.STEP)
                message = "Replay created. Advance it explicitly; no future cycle was exposed."
            elif request.action == "reset":
                if not request.confirm_reset:
                    raise RequestParameterError("Reset requires explicit confirmation.")
                source_id = self._required_source(request.source_asset_id)
                self._check_attempt_limit(source_id, force=True)
                state = self._orchestrator.start(
                    source_id, mode=ReplayMode.STEP, force_restart=True
                )
                message = "A new append-only replay attempt was created; prior history remains."
            else:
                run_id = request.run_id
                if run_id is None:
                    raise RequestParameterError("This replay action requires a run_id.")
                report = self._orchestrator.status(run_id)
                self._enforce_run_source(report.run)
                if request.action == "step":
                    state = self._orchestrator.step(run_id)
                    message = "One cycle was advanced."
                elif request.action == "pause":
                    state = self._orchestrator.stop(run_id)
                    message = "Replay paused after any in-flight cycle."
                elif request.action == "resume":
                    state = self._orchestrator.resume(run_id, max_cycles=1)
                    message = "Replay resumed for one bounded cycle."
                else:
                    maximum = (
                        request.max_cycles or self._settings.replay_public_max_accelerated_cycles
                    )
                    maximum = min(maximum, self._settings.replay_public_max_accelerated_cycles)
                    state = self._orchestrator.resume(run_id, max_cycles=maximum)
                    message = f"Replay advanced by at most {maximum} cycles."
        except (ReplayControlDisabledError, ReplayControlForbiddenError, RequestParameterError):
            raise
        except ReplayError as exc:
            raise ReplayControlConflictError(str(exc)) from exc
        return ReplayActionResponse(
            action=request.action, run=_run_response(state), message=message
        )

    def _authorize(self, supplied: str | None) -> None:
        if not self._settings.replay_controls_enabled:
            raise ReplayControlDisabledError("Replay controls are read-only in this deployment.")
        if self._settings.public_demo_mode:
            return
        expected = self._settings.replay_admin_token
        if expected is None or supplied is None or not hmac.compare_digest(expected, supplied):
            raise ReplayControlForbiddenError("A valid replay control token is required.")

    def _rate_limit(self, client_id: str) -> None:
        now = time.monotonic()
        with self._lock:
            previous = self._last_action.get(client_id)
            if (
                previous is not None
                and now - previous < self._settings.replay_control_cooldown_seconds
            ):
                raise ReplayControlRateLimitedError(
                    "Replay controls are temporarily rate limited; retry shortly."
                )
            self._last_action[client_id] = now

    def _required_source(self, source_id: int | None) -> int:
        if source_id is None:
            raise RequestParameterError("Starting or resetting requires source_asset_id.")
        try:
            available = self._orchestrator.replay_asset_ids()
        except ReplayError as exc:
            raise ReplayControlConflictError("Verified replay data is unavailable.") from exc
        if source_id not in available:
            raise ReplayControlForbiddenError(
                "The selected source asset is not in the protected replay split."
            )
        if (
            self._settings.public_demo_mode
            and source_id != self._settings.replay_demo_source_asset_id
        ):
            raise ReplayControlForbiddenError(
                "Public demo mode permits only the configured demo asset."
            )
        return source_id

    def _enforce_run_source(self, state: ReplayRunState) -> None:
        if (
            self._settings.public_demo_mode
            and state.source_asset_id != self._settings.replay_demo_source_asset_id
        ):
            raise ReplayControlForbiddenError("This replay run is not public-demo eligible.")

    def _check_attempt_limit(self, source_id: int, *, force: bool) -> None:
        reports = self._orchestrator.list_status(limit=100)
        attempts = [
            report.run.attempt for report in reports if report.run.source_asset_id == source_id
        ]
        if force and attempts and max(attempts) >= self._settings.replay_public_max_attempts:
            raise ReplayControlForbiddenError(
                "This demo asset reached its configured replay-attempt limit."
            )

    def _allowed_ids(self, available: list[int]) -> list[int]:
        if not self._settings.public_demo_mode:
            return available
        demo = self._settings.replay_demo_source_asset_id
        return [demo] if demo in available else []

    def _restrictions(self) -> list[str]:
        values = [
            "Only verified replay-split assets are eligible.",
            "Accelerated requests advance a bounded number of cycles.",
            "Concurrent advancement is rejected by the durable replay lease.",
            "Final-cycle ground truth remains hidden until completion.",
            "Reset creates append-only history and requires explicit confirmation.",
        ]
        if not self._settings.replay_controls_enabled:
            values.insert(0, "This deployment exposes replay status as read-only.")
        elif self._settings.public_demo_mode:
            values.insert(0, "Anonymous writes are limited to one predefined demo asset.")
        return values


def _run_response(state: ReplayRunState) -> ReplayRunResponse:
    completed = state.status is ReplayRunStatus.COMPLETED
    return ReplayRunResponse(
        run_id=state.run_id,
        source_asset_id=state.source_asset_id,
        attempt=state.attempt,
        external_asset_id=state.external_asset_id,
        operational_asset_id=state.asset_id,
        status=state.status.value,
        mode=state.mode.value,
        last_confirmed_cycle=state.last_confirmed_cycle,
        final_cycle=state.final_cycle if completed else None,
        progress_percent=round(100.0 * state.last_confirmed_cycle / state.final_cycle, 1),
        started_at=state.replay_started_at,
        last_advanced_at=state.last_advanced_at,
        completed_at=state.completed_at,
        error_message=state.error_message,
    )

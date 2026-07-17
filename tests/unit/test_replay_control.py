"""Public replay policy tests above the already-tested Loop 8 engine."""

import uuid
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from turbine_guard.api.schemas.dashboard import ReplayActionRequest
from turbine_guard.config.settings import Environment, Settings
from turbine_guard.database.enums import ReplayMode, ReplayRunStatus
from turbine_guard.replay.engine import ReplayStatusReport
from turbine_guard.replay.errors import ReplayConcurrencyError, ReplayStateError
from turbine_guard.replay.state import ReplayRunState
from turbine_guard.services.errors import (
    ReplayControlConflictError,
    ReplayControlDisabledError,
    ReplayControlForbiddenError,
    RequestParameterError,
)
from turbine_guard.services.replay_control import ReplayControlService

NOW = datetime(2026, 7, 13, 12, tzinfo=UTC)


def _state(*, source: int = 9, attempt: int = 1) -> ReplayRunState:
    return ReplayRunState(
        run_id=uuid.uuid4(),
        dataset_name="NASA C-MAPSS",
        dataset_subset="FD001",
        source_asset_id=source,
        attempt=attempt,
        external_asset_id=f"replay-FD001-{source:03d}-r{attempt}",
        asset_id=None,
        final_cycle=20,
        last_confirmed_cycle=0,
        status=ReplayRunStatus.CREATED,
        mode=ReplayMode.STEP,
        cycle_delay_seconds=0,
        simulated_cycle_duration_seconds=1,
        replay_started_at=NOW,
        last_advanced_at=None,
        completed_at=None,
        failure_event_id=None,
        labels_backfilled_at=None,
        evaluation_completed_at=None,
        error_message=None,
        metadata={},
    )


class FakeOrchestrator:
    def __init__(self) -> None:
        self.runs: dict[uuid.UUID, ReplayRunState] = {}
        self.drive_budget: int | None = None
        self.fail_step = False

    def replay_asset_ids(self) -> tuple[int, ...]:
        return (9, 10)

    def start(
        self,
        source_asset_id: int,
        *,
        mode: ReplayMode = ReplayMode.STEP,
        cycle_delay_seconds: float | None = None,
        force_restart: bool = False,
    ) -> ReplayRunState:
        del mode, cycle_delay_seconds
        prior = [state for state in self.runs.values() if state.source_asset_id == source_asset_id]
        if prior and not force_restart:
            return prior[-1]
        state = _state(source=source_asset_id, attempt=len(prior) + 1)
        self.runs[state.run_id] = state
        return state

    def step(self, run_id: uuid.UUID) -> ReplayRunState:
        if self.fail_step:
            raise ReplayConcurrencyError("already advancing")
        state = self.runs[run_id]
        state = replace(
            state,
            status=ReplayRunStatus.RUNNING,
            last_confirmed_cycle=state.last_confirmed_cycle + 1,
        )
        self.runs[run_id] = state
        return state

    def resume(self, run_id: uuid.UUID, *, max_cycles: int | None = None) -> ReplayRunState:
        self.drive_budget = max_cycles
        state = self.runs[run_id]
        state = replace(
            state,
            status=ReplayRunStatus.RUNNING,
            last_confirmed_cycle=min(
                state.final_cycle, state.last_confirmed_cycle + (max_cycles or 1)
            ),
        )
        self.runs[run_id] = state
        return state

    def drive(self, run_id: uuid.UUID, *, max_cycles: int | None = None) -> ReplayRunState:
        return self.resume(run_id, max_cycles=max_cycles)

    def stop(self, run_id: uuid.UUID) -> ReplayRunState:
        state = replace(self.runs[run_id], status=ReplayRunStatus.PAUSED)
        self.runs[run_id] = state
        return state

    def status(self, run_id: uuid.UUID) -> ReplayStatusReport:
        state = self.runs.get(run_id)
        if state is None:
            raise ReplayStateError("missing")
        return ReplayStatusReport(state, ())

    def list_status(self, *, limit: int = 100) -> list[ReplayStatusReport]:
        return [ReplayStatusReport(state, ()) for state in list(self.runs.values())[-limit:]]


def _settings(**updates: object) -> Settings:
    return Settings(
        environment=Environment.TESTING,
        online_inference_enabled=False,
        replay_controls_enabled=True,
        public_demo_mode=True,
        application_secret="unit-test-application-secret",
        replay_control_cooldown_seconds=0,
        **updates,
    )


def test_read_only_mode_blocks_all_mutations() -> None:
    service = ReplayControlService(
        FakeOrchestrator(),
        Settings(environment=Environment.TESTING, online_inference_enabled=False),
    )
    with pytest.raises(ReplayControlDisabledError):
        service.perform(ReplayActionRequest(action="start", source_asset_id=9), client_id="one")


def test_public_demo_rejects_protected_or_non_demo_asset() -> None:
    service = ReplayControlService(FakeOrchestrator(), _settings())
    with pytest.raises(ReplayControlForbiddenError):
        service.perform(ReplayActionRequest(action="start", source_asset_id=10), client_id="one")
    with pytest.raises(ReplayControlForbiddenError):
        service.perform(ReplayActionRequest(action="start", source_asset_id=99), client_id="two")


def test_start_step_pause_resume_and_bounded_acceleration() -> None:
    orchestrator = FakeOrchestrator()
    service = ReplayControlService(orchestrator, _settings())
    started = service.perform(
        ReplayActionRequest(action="start", source_asset_id=9), client_id="one"
    )
    assert started.run.last_confirmed_cycle == 0
    assert started.run.final_cycle is None
    stepped = service.perform(
        ReplayActionRequest(action="step", run_id=started.run.run_id), client_id="one"
    )
    assert stepped.run.last_confirmed_cycle == 1
    paused = service.perform(
        ReplayActionRequest(action="pause", run_id=started.run.run_id), client_id="one"
    )
    assert paused.run.status == "paused"
    resumed = service.perform(
        ReplayActionRequest(action="resume", run_id=started.run.run_id), client_id="one"
    )
    assert resumed.run.last_confirmed_cycle == 2
    service.perform(
        ReplayActionRequest(action="accelerate", run_id=started.run.run_id, max_cycles=100),
        client_id="one",
    )
    assert orchestrator.drive_budget == 10


def test_reset_requires_confirmation_and_attempt_limit() -> None:
    service = ReplayControlService(FakeOrchestrator(), _settings(replay_public_max_attempts=2))
    started = service.perform(
        ReplayActionRequest(action="start", source_asset_id=9), client_id="one"
    )
    with pytest.raises(RequestParameterError):
        service.perform(ReplayActionRequest(action="reset", source_asset_id=9), client_id="one")
    reset = service.perform(
        ReplayActionRequest(action="reset", source_asset_id=9, confirm_reset=True), client_id="one"
    )
    assert reset.run.attempt == 2
    with pytest.raises(ReplayControlForbiddenError):
        service.perform(
            ReplayActionRequest(action="reset", source_asset_id=9, confirm_reset=True),
            client_id="one",
        )
    assert started.run.attempt == 1


def test_concurrent_engine_rejection_becomes_safe_conflict() -> None:
    orchestrator = FakeOrchestrator()
    service = ReplayControlService(orchestrator, _settings())
    run = service.perform(
        ReplayActionRequest(action="start", source_asset_id=9), client_id="one"
    ).run
    orchestrator.fail_step = True
    with pytest.raises(ReplayControlConflictError, match="already advancing"):
        service.perform(ReplayActionRequest(action="step", run_id=run.run_id), client_id="one")


def test_non_demo_controls_require_constant_time_token_match() -> None:
    settings = Settings(
        environment=Environment.TESTING,
        online_inference_enabled=False,
        replay_controls_enabled=True,
        public_demo_mode=False,
        application_secret="secret",
        replay_admin_token="admin-token",
        replay_control_cooldown_seconds=0,
    )
    service = ReplayControlService(FakeOrchestrator(), settings)
    with pytest.raises(ReplayControlForbiddenError):
        service.perform(ReplayActionRequest(action="start", source_asset_id=9), client_id="one")
    response = service.perform(
        ReplayActionRequest(action="start", source_asset_id=9, control_token="admin-token"),
        client_id="one",
    )
    assert response.run.source_asset_id == 9

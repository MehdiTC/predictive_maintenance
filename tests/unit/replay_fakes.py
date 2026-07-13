"""In-memory fakes implementing the replay store and ingestion contracts.

``FakeOnlineSystem`` mimics the Loop 7 ingestion semantics (contiguity,
exact-retry idempotency, model-version-pinned predictions) over shared
in-memory tables, and ``InMemoryReplayStateStore`` implements
``ReplayStateStore`` over the same tables, so orchestrator unit tests can
exercise the full lifecycle deterministically without PostgreSQL or HTTP.
"""

import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Any

from turbine_guard.api.schemas.online import SensorReadingRequest
from turbine_guard.database.commands import (
    NewMaintenanceEvent,
    NewModelEvaluation,
    NewPredictionOutcome,
    NewReplayRun,
)
from turbine_guard.database.enums import ReplayRunStatus
from turbine_guard.replay.client import IngestionResult
from turbine_guard.replay.errors import (
    ReplayConcurrencyError,
    ReplayIngestionError,
    ReplayOutcomeError,
    ReplayStateError,
)
from turbine_guard.replay.state import (
    AdvanceClaim,
    ReplayRunState,
    StoredEvaluation,
    StoredOutcome,
    StoredPrediction,
)

ACTIVE = frozenset({ReplayRunStatus.CREATED, ReplayRunStatus.RUNNING})
TERMINAL = frozenset({ReplayRunStatus.COMPLETED, ReplayRunStatus.CANCELLED})


class MutableClock:
    """Deterministic, test-advanceable UTC clock."""

    def __init__(self, start: datetime | None = None) -> None:
        self.now = start or datetime(2026, 7, 13, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


@dataclass
class _StoredEvent:
    event_id: uuid.UUID
    command: NewMaintenanceEvent


class FakeOnlineSystem:
    """Shared in-memory operational state standing in for API + PostgreSQL."""

    def __init__(self, prediction_fn: Callable[[int], float] | None = None) -> None:
        self.assets: dict[str, uuid.UUID] = {}
        self.readings: dict[tuple[uuid.UUID, int], dict[str, Any]] = {}
        self.predictions: list[StoredPrediction] = []
        self.prediction_assets: dict[uuid.UUID, uuid.UUID] = {}
        self.events: dict[str, _StoredEvent] = {}
        self.outcomes: dict[tuple[uuid.UUID, uuid.UUID], NewPredictionOutcome] = {}
        self.evaluations: list[tuple[uuid.UUID | None, NewModelEvaluation]] = []
        self.model_version = "1"
        self.prediction_fn = prediction_fn or (lambda cycle: max(0.0, 100.0 - float(cycle)))
        self.sent_payloads: list[dict[str, Any]] = []


class FakeIngestionClient:
    """Loop 7 ingestion semantics without HTTP; records every payload sent."""

    def __init__(self, system: FakeOnlineSystem) -> None:
        self.system = system
        self.fail_next: list[Exception] = []

    def send_reading(self, request: SensorReadingRequest) -> IngestionResult:
        if self.fail_next:
            raise self.fail_next.pop(0)
        system = self.system
        payload = request.model_dump(mode="json")
        system.sent_payloads.append(payload)
        asset_id = system.assets.get(request.external_asset_id)
        if asset_id is None:
            if request.cycle != 1:
                raise ReplayIngestionError(
                    "HTTP 409 history_conflict: a new asset must begin at cycle 1."
                )
            asset_id = uuid.uuid4()
            system.assets[request.external_asset_id] = asset_id
        key = (asset_id, request.cycle)
        existing = system.readings.get(key)
        reading_idempotent = False
        if existing is not None:
            if existing != payload:
                raise ReplayIngestionError(
                    "HTTP 409 sensor_reading_conflict: cycle exists with different data."
                )
            reading_idempotent = True
        else:
            latest = max(
                (cycle for (asset, cycle) in system.readings if asset == asset_id),
                default=0,
            )
            if request.cycle != latest + 1:
                raise ReplayIngestionError(
                    f"HTTP 409 history_conflict: expected cycle {latest + 1}."
                )
            system.readings[key] = payload
        prediction = next(
            (
                p
                for p in system.predictions
                if p.cycle == request.cycle
                and p.model_version == system.model_version
                and system.prediction_assets.get(p.prediction_id) == asset_id
            ),
            None,
        )
        prediction_idempotent = prediction is not None
        if prediction is None:
            point = float(self.system.prediction_fn(request.cycle))
            prediction = StoredPrediction(
                prediction_id=uuid.uuid4(),
                cycle=request.cycle,
                predicted_rul=point,
                lower_rul=max(0.0, point - 5.0),
                upper_rul=point + 5.0,
                risk_level=("critical" if point <= 30 else "warning" if point <= 50 else "healthy"),
                model_name="fake-rul",
                model_version=system.model_version,
                model_run_id=f"run-{system.model_version}",
                prediction_timestamp=request.observed_at or datetime.now(UTC),
            )
            system.predictions.append(prediction)
            system.prediction_assets[prediction.prediction_id] = asset_id
        return IngestionResult(
            asset_id=asset_id,
            external_asset_id=request.external_asset_id,
            cycle=request.cycle,
            idempotent=reading_idempotent and prediction_idempotent,
            predicted_rul=prediction.predicted_rul,
            risk_level=prediction.risk_level,
            model_version=prediction.model_version,
            retries=0,
        )


@dataclass
class _Run:
    state: ReplayRunState
    lease_token: str | None = None
    lease_expires_at: datetime | None = None


@dataclass
class FakeReplaySource:
    """Verified-source stand-in over prebuilt trajectories."""

    trajectories: dict[int, Any]
    fail_with: Exception | None = None

    def replay_asset_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.trajectories))

    def load_trajectory(self, source_asset_id: int) -> Any:
        from turbine_guard.replay.errors import ReplaySourceError

        if self.fail_with is not None:
            raise self.fail_with
        trajectory = self.trajectories.get(source_asset_id)
        if trajectory is None:
            raise ReplaySourceError(f"Source asset {source_asset_id} is not replayable.")
        return trajectory


class InMemoryReplayStateStore:
    """Thread-safe in-memory ``ReplayStateStore`` mirroring PostgreSQL semantics."""

    def __init__(self, system: FakeOnlineSystem, clock: MutableClock | None = None) -> None:
        self.system = system
        self.clock = clock or MutableClock()
        self._runs: dict[uuid.UUID, _Run] = {}
        self._lock = threading.Lock()

    # -- run lifecycle ------------------------------------------------------

    def create_run(self, command: NewReplayRun) -> ReplayRunState:
        with self._lock:
            for run in self._runs.values():
                same_source = (
                    run.state.dataset_name == command.dataset_name
                    and run.state.dataset_subset == command.dataset_subset
                    and run.state.source_asset_id == command.source_asset_id
                    and run.state.attempt == command.attempt
                )
                if same_source or run.state.external_asset_id == command.external_asset_id:
                    from turbine_guard.database.errors import ReplayRunConflictError

                    raise ReplayRunConflictError("Replay run identity already exists.")
            state = ReplayRunState(
                run_id=uuid.uuid4(),
                dataset_name=command.dataset_name,
                dataset_subset=command.dataset_subset,
                source_asset_id=command.source_asset_id,
                attempt=command.attempt,
                external_asset_id=command.external_asset_id,
                asset_id=None,
                final_cycle=command.final_cycle,
                last_confirmed_cycle=0,
                status=command.status,
                mode=command.mode,
                cycle_delay_seconds=command.cycle_delay_seconds,
                simulated_cycle_duration_seconds=command.simulated_cycle_duration_seconds,
                replay_started_at=command.replay_started_at,
                last_advanced_at=None,
                completed_at=None,
                failure_event_id=None,
                labels_backfilled_at=None,
                evaluation_completed_at=None,
                error_message=None,
                metadata=dict(command.metadata),
            )
            self._runs[state.run_id] = _Run(state=state)
            return state

    def get_run(self, run_id: uuid.UUID) -> ReplayRunState | None:
        run = self._runs.get(run_id)
        return None if run is None else run.state

    def latest_run_for_source(
        self, dataset_name: str, dataset_subset: str, source_asset_id: int
    ) -> ReplayRunState | None:
        candidates = [
            run.state
            for run in self._runs.values()
            if run.state.dataset_name == dataset_name
            and run.state.dataset_subset == dataset_subset
            and run.state.source_asset_id == source_asset_id
        ]
        return max(candidates, key=lambda state: state.attempt, default=None)

    def list_runs(self, *, limit: int = 100) -> list[ReplayRunState]:
        return [run.state for run in list(self._runs.values())[:limit]]

    def claim_advance(self, run_id: uuid.UUID, *, lease_seconds: float) -> AdvanceClaim:
        with self._lock:
            run = self._require(run_id)
            now = self.clock()
            if run.state.status not in ACTIVE:
                raise ReplayStateError(
                    f"Replay run {run_id} is {run.state.status.value}; it cannot advance."
                )
            if run.lease_token is not None and (
                run.lease_expires_at is not None and run.lease_expires_at > now
            ):
                raise ReplayConcurrencyError("Another worker holds the advance lease.")
            if run.state.last_confirmed_cycle >= run.state.final_cycle:
                raise ReplayStateError("The final cycle is already ingested.")
            token = uuid.uuid4().hex
            run.lease_token = token
            run.lease_expires_at = now + timedelta(seconds=lease_seconds)
            run.state = replace(run.state, status=ReplayRunStatus.RUNNING)
            return AdvanceClaim(
                token=token,
                next_cycle=run.state.last_confirmed_cycle + 1,
                run=run.state,
            )

    def confirm_advance(
        self, run_id: uuid.UUID, *, token: str, cycle: int, asset_id: uuid.UUID
    ) -> ReplayRunState:
        with self._lock:
            run = self._require(run_id)
            if run.lease_token != token:
                raise ReplayConcurrencyError("The advance lease was lost before confirmation.")
            if cycle != run.state.last_confirmed_cycle + 1:
                raise ReplayStateError(f"Cannot confirm out-of-order cycle {cycle}.")
            run.state = replace(
                run.state,
                last_confirmed_cycle=cycle,
                last_advanced_at=self.clock(),
                asset_id=run.state.asset_id or asset_id,
            )
            run.lease_token = None
            run.lease_expires_at = None
            return run.state

    def release_advance(self, run_id: uuid.UUID, *, token: str) -> None:
        with self._lock:
            run = self._require(run_id)
            if run.lease_token == token:
                run.lease_token = None
                run.lease_expires_at = None

    def request_stop(self, run_id: uuid.UUID) -> ReplayRunState:
        with self._lock:
            run = self._require(run_id)
            if run.state.status in ACTIVE:
                run.state = replace(run.state, status=ReplayRunStatus.PAUSED)
            elif run.state.status is not ReplayRunStatus.PAUSED:
                raise ReplayStateError(
                    f"Replay run {run_id} is {run.state.status.value} and cannot be stopped."
                )
            return run.state

    def mark_running(self, run_id: uuid.UUID) -> ReplayRunState:
        with self._lock:
            run = self._require(run_id)
            if run.state.status in TERMINAL:
                raise ReplayStateError("Terminal runs cannot be resumed.")
            run.state = replace(run.state, status=ReplayRunStatus.RUNNING, error_message=None)
            return run.state

    def mark_failed(self, run_id: uuid.UUID, error: str) -> ReplayRunState:
        with self._lock:
            run = self._require(run_id)
            if run.state.status in TERMINAL:
                return run.state
            run.state = replace(run.state, status=ReplayRunStatus.FAILED, error_message=error)
            run.lease_token = None
            run.lease_expires_at = None
            return run.state

    def cancel_run(self, run_id: uuid.UUID) -> ReplayRunState:
        with self._lock:
            run = self._require(run_id)
            if run.state.status is ReplayRunStatus.COMPLETED:
                raise ReplayStateError("Completed runs are never cancelled.")
            run.state = replace(run.state, status=ReplayRunStatus.CANCELLED)
            run.lease_token = None
            run.lease_expires_at = None
            return run.state

    # -- delayed feedback phases -------------------------------------------

    def record_failure_event(self, run_id: uuid.UUID, command: NewMaintenanceEvent) -> uuid.UUID:
        with self._lock:
            run = self._require(run_id)
            if run.state.failure_event_id is not None:
                return run.state.failure_event_id
            assert command.external_event_id is not None
            stored = self.system.events.get(command.external_event_id)
            if stored is None:
                stored = _StoredEvent(event_id=uuid.uuid4(), command=command)
                self.system.events[command.external_event_id] = stored
            elif stored.command != command:
                from turbine_guard.database.errors import DuplicateExternalIdError

                raise DuplicateExternalIdError("Conflicting failure event data.")
            run.state = replace(run.state, failure_event_id=stored.event_id)
            return stored.event_id

    def record_outcomes(self, run_id: uuid.UUID, commands: list[NewPredictionOutcome]) -> int:
        with self._lock:
            run = self._require(run_id)
            for command in commands:
                key = (command.prediction_id, command.maintenance_event_id)
                existing = self.system.outcomes.get(key)
                if existing is not None and (
                    existing.realized_rul != command.realized_rul or existing.cycle != command.cycle
                ):
                    raise ReplayOutcomeError("Conflicting realized label.")
                self.system.outcomes.setdefault(key, command)
            if run.state.labels_backfilled_at is None:
                run.state = replace(run.state, labels_backfilled_at=self.clock())
            return len(commands)

    def record_evaluations(self, run_id: uuid.UUID, commands: list[NewModelEvaluation]) -> int:
        with self._lock:
            run = self._require(run_id)
            if run.state.evaluation_completed_at is not None:
                return 0
            for command in commands:
                self.system.evaluations.append((run_id, command))
            run.state = replace(run.state, evaluation_completed_at=self.clock())
            return len(commands)

    def record_aggregate_evaluations(self, commands: list[NewModelEvaluation]) -> int:
        with self._lock:
            for command in commands:
                self.system.evaluations.append((None, command))
            return len(commands)

    def complete_run(self, run_id: uuid.UUID) -> ReplayRunState:
        with self._lock:
            run = self._require(run_id)
            if run.state.status is ReplayRunStatus.COMPLETED:
                return run.state
            if (
                run.state.last_confirmed_cycle < run.state.final_cycle
                or run.state.failure_event_id is None
                or run.state.labels_backfilled_at is None
                or run.state.evaluation_completed_at is None
            ):
                raise ReplayStateError("Incomplete phases; cannot complete.")
            run.state = replace(
                run.state, status=ReplayRunStatus.COMPLETED, completed_at=self.clock()
            )
            return run.state

    # -- reads ----------------------------------------------------------------

    def resolve_asset(self, external_asset_id: str) -> uuid.UUID | None:
        return self.system.assets.get(external_asset_id)

    def reading_exists(self, asset_id: uuid.UUID, cycle: int) -> bool:
        return (asset_id, cycle) in self.system.readings

    def stored_predictions(self, asset_id: uuid.UUID) -> tuple[StoredPrediction, ...]:
        return tuple(
            sorted(
                (
                    prediction
                    for prediction in self.system.predictions
                    if self.system.prediction_assets.get(prediction.prediction_id) == asset_id
                ),
                key=lambda prediction: (prediction.cycle, prediction.model_version),
            )
        )

    def outcomes_for_event(self, event_id: uuid.UUID) -> tuple[StoredOutcome, ...]:
        return tuple(
            StoredOutcome(
                prediction_id=command.prediction_id,
                cycle=command.cycle,
                realized_rul=command.realized_rul,
            )
            for (_, outcome_event_id), command in sorted(
                self.system.outcomes.items(), key=lambda item: item[1].cycle
            )
            if outcome_event_id == event_id
        )

    def evaluations_for_run(self, run_id: uuid.UUID) -> tuple[StoredEvaluation, ...]:
        return tuple(
            _stored_evaluation(command)
            for owner, command in self.system.evaluations
            if owner == run_id
        )

    def evaluations_for_model(
        self, model_name: str, model_version: str
    ) -> tuple[StoredEvaluation, ...]:
        return tuple(
            _stored_evaluation(command)
            for _, command in self.system.evaluations
            if command.model_name == model_name and command.model_version == model_version
        )

    # -- helpers ---------------------------------------------------------------

    def _require(self, run_id: uuid.UUID) -> _Run:
        run = self._runs.get(run_id)
        if run is None:
            raise ReplayStateError(f"Replay run {run_id} does not exist.")
        return run


def _stored_evaluation(command: NewModelEvaluation) -> StoredEvaluation:
    return StoredEvaluation(
        model_name=command.model_name,
        model_version=command.model_version,
        evaluation_scope=command.evaluation_scope.value,
        sample_count=command.sample_count,
        mae=command.mae,
        rmse=command.rmse,
        nasa_score=command.nasa_score,
        critical_precision=command.critical_precision,
        critical_recall=command.critical_recall,
        interval_coverage=command.interval_coverage,
        metrics=dict(command.metrics),
        created_at=command.window_end,
    )


def make_replay_system(
    prediction_fn: Callable[[int], float] | None = None,
) -> tuple[FakeOnlineSystem, InMemoryReplayStateStore, FakeIngestionClient, MutableClock]:
    """Wire a shared fake online system, store, client, and clock."""
    system = FakeOnlineSystem(prediction_fn)
    clock = MutableClock()
    store = InMemoryReplayStateStore(system, clock)
    client = FakeIngestionClient(system)
    return system, store, client, clock

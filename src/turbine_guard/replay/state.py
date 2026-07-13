"""Durable replay-run state with lease-based single-writer advancement.

The store is the only component that touches replay persistence. Every method
runs in its own short transaction; no database lock is ever held across an
HTTP wait. Exclusive advancement uses a claim lease on the run row: a worker
claims before sending a cycle, sends outside any transaction, then confirms
with its token. A competing worker cannot claim an active lease, so the same
next cycle is never sent twice; a crashed worker's lease simply expires and
the idempotent resend reconciles progress.

The engine consumes frozen snapshots instead of ORM objects, which keeps the
orchestration logic independent of SQLAlchemy sessions and lets unit tests
substitute an in-memory store implementing :class:`ReplayStateStore`.
"""

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from turbine_guard.database.commands import (
    NewMaintenanceEvent,
    NewModelEvaluation,
    NewPredictionOutcome,
    NewReplayRun,
)
from turbine_guard.database.enums import ReplayMode, ReplayRunStatus
from turbine_guard.database.models import Asset, ModelEvaluation, ReplayRun, SensorReading
from turbine_guard.database.repositories import (
    MaintenanceEventRepository,
    ModelEvaluationRepository,
    PredictionOutcomeRepository,
    PredictionRepository,
    ReplayRunRepository,
)
from turbine_guard.database.session import session_scope
from turbine_guard.replay.errors import ReplayConcurrencyError, ReplayStateError

ACTIVE_STATUSES = frozenset({ReplayRunStatus.CREATED, ReplayRunStatus.RUNNING})
TERMINAL_STATUSES = frozenset({ReplayRunStatus.COMPLETED, ReplayRunStatus.CANCELLED})


@dataclass(frozen=True)
class ReplayRunState:
    """Detached snapshot of one replay run row."""

    run_id: uuid.UUID
    dataset_name: str
    dataset_subset: str
    source_asset_id: int
    attempt: int
    external_asset_id: str
    asset_id: uuid.UUID | None
    final_cycle: int
    last_confirmed_cycle: int
    status: ReplayRunStatus
    mode: ReplayMode
    cycle_delay_seconds: float
    simulated_cycle_duration_seconds: float
    replay_started_at: datetime
    last_advanced_at: datetime | None
    completed_at: datetime | None
    failure_event_id: uuid.UUID | None
    labels_backfilled_at: datetime | None
    evaluation_completed_at: datetime | None
    error_message: str | None
    metadata: dict[str, Any]

    @property
    def ingest_complete(self) -> bool:
        return self.last_confirmed_cycle >= self.final_cycle


@dataclass(frozen=True)
class AdvanceClaim:
    """Exclusive permission to send exactly one next cycle."""

    token: str
    next_cycle: int
    run: ReplayRunState


@dataclass(frozen=True)
class StoredPrediction:
    """Immutable prediction fields needed for backfill and delayed evaluation."""

    prediction_id: uuid.UUID
    cycle: int
    predicted_rul: float
    lower_rul: float | None
    upper_rul: float | None
    risk_level: str
    model_name: str
    model_version: str
    model_run_id: str | None
    prediction_timestamp: datetime


@dataclass(frozen=True)
class StoredOutcome:
    """One realized label linked to a prediction and its outcome event."""

    prediction_id: uuid.UUID
    cycle: int
    realized_rul: int


@dataclass(frozen=True)
class StoredEvaluation:
    """Persisted delayed-evaluation summary for status reporting."""

    model_name: str
    model_version: str
    evaluation_scope: str
    sample_count: int
    mae: float | None
    rmse: float | None
    nasa_score: float | None
    critical_precision: float | None
    critical_recall: float | None
    interval_coverage: float | None
    metrics: dict[str, Any]
    created_at: datetime


class ReplayStateStore(Protocol):
    """Persistence contract the replay orchestrator depends on."""

    def create_run(self, command: NewReplayRun) -> ReplayRunState: ...
    def get_run(self, run_id: uuid.UUID) -> ReplayRunState | None: ...
    def latest_run_for_source(
        self, dataset_name: str, dataset_subset: str, source_asset_id: int
    ) -> ReplayRunState | None: ...
    def list_runs(self, *, limit: int = 100) -> list[ReplayRunState]: ...
    def claim_advance(self, run_id: uuid.UUID, *, lease_seconds: float) -> AdvanceClaim: ...
    def confirm_advance(
        self, run_id: uuid.UUID, *, token: str, cycle: int, asset_id: uuid.UUID
    ) -> ReplayRunState: ...
    def release_advance(self, run_id: uuid.UUID, *, token: str) -> None: ...
    def request_stop(self, run_id: uuid.UUID) -> ReplayRunState: ...
    def mark_running(self, run_id: uuid.UUID) -> ReplayRunState: ...
    def mark_failed(self, run_id: uuid.UUID, error: str) -> ReplayRunState: ...
    def cancel_run(self, run_id: uuid.UUID) -> ReplayRunState: ...
    def record_failure_event(
        self, run_id: uuid.UUID, command: NewMaintenanceEvent
    ) -> uuid.UUID: ...
    def record_outcomes(self, run_id: uuid.UUID, commands: list[NewPredictionOutcome]) -> int: ...
    def record_evaluations(self, run_id: uuid.UUID, commands: list[NewModelEvaluation]) -> int: ...
    def record_aggregate_evaluations(self, commands: list[NewModelEvaluation]) -> int: ...
    def complete_run(self, run_id: uuid.UUID) -> ReplayRunState: ...
    def resolve_asset(self, external_asset_id: str) -> uuid.UUID | None: ...
    def reading_exists(self, asset_id: uuid.UUID, cycle: int) -> bool: ...
    def stored_predictions(self, asset_id: uuid.UUID) -> tuple[StoredPrediction, ...]: ...
    def outcomes_for_event(self, event_id: uuid.UUID) -> tuple[StoredOutcome, ...]: ...
    def evaluations_for_run(self, run_id: uuid.UUID) -> tuple[StoredEvaluation, ...]: ...
    def evaluations_for_model(
        self, model_name: str, model_version: str
    ) -> tuple[StoredEvaluation, ...]: ...


class PostgresReplayStateStore:
    """PostgreSQL-backed :class:`ReplayStateStore` over the Loop 6 repositories."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._sessions = session_factory
        self._clock = clock

    def create_run(self, command: NewReplayRun) -> ReplayRunState:
        with session_scope(self._sessions) as session:
            run = ReplayRunRepository(session).create(command)
            return _snapshot(run)

    def get_run(self, run_id: uuid.UUID) -> ReplayRunState | None:
        with session_scope(self._sessions) as session:
            run = ReplayRunRepository(session).get(run_id)
            return None if run is None else _snapshot(run)

    def latest_run_for_source(
        self, dataset_name: str, dataset_subset: str, source_asset_id: int
    ) -> ReplayRunState | None:
        with session_scope(self._sessions) as session:
            run = ReplayRunRepository(session).latest_for_source(
                dataset_name, dataset_subset, source_asset_id
            )
            return None if run is None else _snapshot(run)

    def list_runs(self, *, limit: int = 100) -> list[ReplayRunState]:
        with session_scope(self._sessions) as session:
            return [_snapshot(run) for run in ReplayRunRepository(session).list_runs(limit=limit)]

    def claim_advance(self, run_id: uuid.UUID, *, lease_seconds: float) -> AdvanceClaim:
        """Claim the exclusive right to send the next cycle, in one short transaction."""
        now = self._clock()
        with session_scope(self._sessions) as session:
            run = _locked(session, run_id)
            if run.status not in ACTIVE_STATUSES:
                raise ReplayStateError(
                    f"Replay run {run_id} is {run.status.value}; it cannot advance. "
                    "Resume it explicitly if it is paused or failed."
                )
            if run.lease_token is not None and (
                run.lease_expires_at is not None and run.lease_expires_at > now
            ):
                raise ReplayConcurrencyError(
                    f"Replay run {run_id} is already being advanced by another worker "
                    f"(lease expires {run.lease_expires_at.isoformat()})."
                )
            if run.last_confirmed_cycle >= run.final_cycle:
                raise ReplayStateError(f"Replay run {run_id} has already ingested its final cycle.")
            token = uuid.uuid4().hex
            run.lease_token = token
            run.lease_expires_at = _lease_expiry(now, lease_seconds)
            run.status = ReplayRunStatus.RUNNING
            session.flush()
            return AdvanceClaim(
                token=token, next_cycle=run.last_confirmed_cycle + 1, run=_snapshot(run)
            )

    def confirm_advance(
        self, run_id: uuid.UUID, *, token: str, cycle: int, asset_id: uuid.UUID
    ) -> ReplayRunState:
        """Record one confirmed cycle and release the lease atomically."""
        with session_scope(self._sessions) as session:
            run = _locked(session, run_id)
            if run.lease_token != token:
                raise ReplayConcurrencyError(
                    f"Advance lease for replay run {run_id} was lost before confirmation; "
                    "another worker may have taken over after expiry."
                )
            if cycle != run.last_confirmed_cycle + 1:
                raise ReplayStateError(
                    f"Cannot confirm cycle {cycle}; expected "
                    f"{run.last_confirmed_cycle + 1} for replay run {run_id}."
                )
            run.last_confirmed_cycle = cycle
            run.last_advanced_at = self._clock()
            if run.asset_id is None:
                run.asset_id = asset_id
            run.lease_token = None
            run.lease_expires_at = None
            session.flush()
            return _snapshot(run)

    def release_advance(self, run_id: uuid.UUID, *, token: str) -> None:
        """Give up an unconfirmed claim; a foreign or expired token is a no-op."""
        with session_scope(self._sessions) as session:
            run = _locked(session, run_id)
            if run.lease_token == token:
                run.lease_token = None
                run.lease_expires_at = None
                session.flush()

    def request_stop(self, run_id: uuid.UUID) -> ReplayRunState:
        """Ask the active worker to stop after the cycle it is currently sending."""
        with session_scope(self._sessions) as session:
            run = _locked(session, run_id)
            if run.status in ACTIVE_STATUSES:
                run.status = ReplayRunStatus.PAUSED
                session.flush()
            elif run.status is not ReplayRunStatus.PAUSED:
                raise ReplayStateError(
                    f"Replay run {run_id} is {run.status.value} and cannot be stopped."
                )
            return _snapshot(run)

    def mark_running(self, run_id: uuid.UUID) -> ReplayRunState:
        """Resume a paused or failed run; clears any recorded retryable error."""
        with session_scope(self._sessions) as session:
            run = _locked(session, run_id)
            if run.status in TERMINAL_STATUSES:
                raise ReplayStateError(
                    f"Replay run {run_id} is {run.status.value} and cannot be resumed. "
                    "Start a new run with an explicit force restart instead."
                )
            run.status = ReplayRunStatus.RUNNING
            run.error_message = None
            session.flush()
            return _snapshot(run)

    def mark_failed(self, run_id: uuid.UUID, error: str) -> ReplayRunState:
        with session_scope(self._sessions) as session:
            run = _locked(session, run_id)
            if run.status in TERMINAL_STATUSES:
                return _snapshot(run)
            run.status = ReplayRunStatus.FAILED
            run.error_message = error
            run.lease_token = None
            run.lease_expires_at = None
            session.flush()
            return _snapshot(run)

    def cancel_run(self, run_id: uuid.UUID) -> ReplayRunState:
        """Supersede an incomplete run (force restart); completed runs stay intact."""
        with session_scope(self._sessions) as session:
            run = _locked(session, run_id)
            if run.status is ReplayRunStatus.COMPLETED:
                raise ReplayStateError(
                    f"Replay run {run_id} is completed and will not be cancelled."
                )
            if run.status is not ReplayRunStatus.CANCELLED:
                run.status = ReplayRunStatus.CANCELLED
                run.lease_token = None
                run.lease_expires_at = None
                session.flush()
            return _snapshot(run)

    def record_failure_event(self, run_id: uuid.UUID, command: NewMaintenanceEvent) -> uuid.UUID:
        """Create the failure event and link it to the run in one transaction.

        Repeated calls return the existing event: deduplication rests on the
        explicit ``external_event_id`` plus the run-row link.
        """
        with session_scope(self._sessions) as session:
            run = _locked(session, run_id)
            if run.failure_event_id is not None:
                return run.failure_event_id
            event = MaintenanceEventRepository(session).create(command)
            run.failure_event_id = event.id
            session.flush()
            return event.id

    def record_outcomes(self, run_id: uuid.UUID, commands: list[NewPredictionOutcome]) -> int:
        """Persist realized labels idempotently and stamp the backfill phase."""
        with session_scope(self._sessions) as session:
            run = _locked(session, run_id)
            repository = PredictionOutcomeRepository(session)
            for command in commands:
                repository.create(command)
            if run.labels_backfilled_at is None:
                run.labels_backfilled_at = self._clock()
            session.flush()
            return len(commands)

    def record_evaluations(self, run_id: uuid.UUID, commands: list[NewModelEvaluation]) -> int:
        """Persist delayed evaluations once; a stamped phase is never repeated."""
        with session_scope(self._sessions) as session:
            run = _locked(session, run_id)
            if run.evaluation_completed_at is not None:
                return 0
            repository = ModelEvaluationRepository(session)
            for command in commands:
                repository.create(command)
            run.evaluation_completed_at = self._clock()
            session.flush()
            return len(commands)

    def record_aggregate_evaluations(self, commands: list[NewModelEvaluation]) -> int:
        """Persist run-spanning aggregate evaluations; idempotency lives above."""
        with session_scope(self._sessions) as session:
            repository = ModelEvaluationRepository(session)
            for command in commands:
                repository.create(command)
            return len(commands)

    def complete_run(self, run_id: uuid.UUID) -> ReplayRunState:
        with session_scope(self._sessions) as session:
            run = _locked(session, run_id)
            if run.status is ReplayRunStatus.COMPLETED:
                return _snapshot(run)
            if (
                run.last_confirmed_cycle < run.final_cycle
                or run.failure_event_id is None
                or run.labels_backfilled_at is None
                or run.evaluation_completed_at is None
            ):
                raise ReplayStateError(
                    f"Replay run {run_id} has incomplete phases and cannot be completed."
                )
            run.status = ReplayRunStatus.COMPLETED
            run.completed_at = self._clock()
            run.lease_token = None
            run.lease_expires_at = None
            session.flush()
            return _snapshot(run)

    def resolve_asset(self, external_asset_id: str) -> uuid.UUID | None:
        with session_scope(self._sessions) as session:
            asset = session.scalar(select(Asset).where(Asset.external_id == external_asset_id))
            return None if asset is None else asset.id

    def reading_exists(self, asset_id: uuid.UUID, cycle: int) -> bool:
        with session_scope(self._sessions) as session:
            reading = session.scalar(
                select(SensorReading.id).where(
                    SensorReading.asset_id == asset_id, SensorReading.cycle == cycle
                )
            )
            return reading is not None

    def stored_predictions(self, asset_id: uuid.UUID) -> tuple[StoredPrediction, ...]:
        with session_scope(self._sessions) as session:
            predictions = PredictionRepository(session).for_asset(asset_id, limit=100_000)
            return tuple(
                StoredPrediction(
                    prediction_id=prediction.id,
                    cycle=prediction.cycle,
                    predicted_rul=prediction.predicted_rul,
                    lower_rul=prediction.lower_rul,
                    upper_rul=prediction.upper_rul,
                    risk_level=prediction.risk_level.value,
                    model_name=prediction.model_name,
                    model_version=prediction.model_version,
                    model_run_id=prediction.model_run_id,
                    prediction_timestamp=prediction.prediction_timestamp,
                )
                for prediction in predictions
            )

    def outcomes_for_event(self, event_id: uuid.UUID) -> tuple[StoredOutcome, ...]:
        with session_scope(self._sessions) as session:
            outcomes = PredictionOutcomeRepository(session).for_event(event_id)
            return tuple(
                StoredOutcome(
                    prediction_id=outcome.prediction_id,
                    cycle=outcome.cycle,
                    realized_rul=outcome.realized_rul,
                )
                for outcome in outcomes
            )

    def evaluations_for_run(self, run_id: uuid.UUID) -> tuple[StoredEvaluation, ...]:
        with session_scope(self._sessions) as session:
            evaluations = ModelEvaluationRepository(session).for_replay_run(run_id)
            return tuple(_evaluation_snapshot(evaluation) for evaluation in evaluations)

    def evaluations_for_model(
        self, model_name: str, model_version: str
    ) -> tuple[StoredEvaluation, ...]:
        with session_scope(self._sessions) as session:
            evaluations = ModelEvaluationRepository(session).for_model(model_name, model_version)
            return tuple(_evaluation_snapshot(evaluation) for evaluation in evaluations)


def _locked(session: Session, run_id: uuid.UUID) -> ReplayRun:
    run = ReplayRunRepository(session).get_for_update(run_id)
    if run is None:
        raise ReplayStateError(f"Replay run {run_id} does not exist.")
    return run


def _lease_expiry(now: datetime, lease_seconds: float) -> datetime:
    if lease_seconds <= 0:
        raise ReplayStateError("Lease duration must be positive.")
    return now + timedelta(seconds=lease_seconds)


def _snapshot(run: ReplayRun) -> ReplayRunState:
    return ReplayRunState(
        run_id=run.id,
        dataset_name=run.dataset_name,
        dataset_subset=run.dataset_subset,
        source_asset_id=run.source_asset_id,
        attempt=run.attempt,
        external_asset_id=run.external_asset_id,
        asset_id=run.asset_id,
        final_cycle=run.final_cycle,
        last_confirmed_cycle=run.last_confirmed_cycle,
        status=run.status,
        mode=run.mode,
        cycle_delay_seconds=run.cycle_delay_seconds,
        simulated_cycle_duration_seconds=run.simulated_cycle_duration_seconds,
        replay_started_at=run.replay_started_at,
        last_advanced_at=run.last_advanced_at,
        completed_at=run.completed_at,
        failure_event_id=run.failure_event_id,
        labels_backfilled_at=run.labels_backfilled_at,
        evaluation_completed_at=run.evaluation_completed_at,
        error_message=run.error_message,
        metadata=dict(run.run_metadata),
    )


def _evaluation_snapshot(evaluation: ModelEvaluation) -> StoredEvaluation:
    return StoredEvaluation(
        model_name=evaluation.model_name,
        model_version=evaluation.model_version,
        evaluation_scope=evaluation.evaluation_scope.value,
        sample_count=evaluation.sample_count,
        mae=evaluation.mae,
        rmse=evaluation.rmse,
        nasa_score=evaluation.nasa_score,
        critical_precision=evaluation.critical_precision,
        critical_recall=evaluation.critical_recall,
        interval_coverage=evaluation.interval_coverage,
        metrics=dict(evaluation.metrics),
        created_at=evaluation.created_at,
    )

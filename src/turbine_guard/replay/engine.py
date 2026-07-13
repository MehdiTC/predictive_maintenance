"""Replay orchestration: modes, exclusive advancement, and delayed feedback.

One advance is a three-part protocol — claim the run's lease in a short
transaction, send exactly one cycle through the real Loop 7 HTTP contract,
then confirm progress with the claim token. Because payloads are
deterministic and ingestion is exactly idempotent, every failure mode
(timeout, crash before confirm, duplicate retry) reconciles by resending the
same cycle. After the final cycle is confirmed the run finalizes through
strictly ordered, individually idempotent phases: failure event, realized
label backfill, delayed evaluation, completion. Resume always continues from
the earliest incomplete phase recorded in durable state.
"""

import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from turbine_guard import __version__
from turbine_guard.database.commands import NewMaintenanceEvent, NewReplayRun
from turbine_guard.database.enums import MaintenanceEventType, ReplayMode, ReplayRunStatus
from turbine_guard.observability.metrics import ReplayMetrics
from turbine_guard.replay.client import ReplayIngestionClient, build_reading_request
from turbine_guard.replay.errors import (
    ReplayIngestionError,
    ReplayOutcomeError,
    ReplaySourceError,
    ReplayStateError,
    ReplayTransientError,
)
from turbine_guard.replay.evaluation import (
    AGGREGATE_AGGREGATION,
    DelayedEvaluationConfig,
    aggregate_evaluations,
    build_outcome_frame,
    per_asset_evaluations,
)
from turbine_guard.replay.feedback import build_outcome_commands
from turbine_guard.replay.source import ReplaySource, ReplayTrajectory
from turbine_guard.replay.state import (
    ReplayRunState,
    ReplayStateStore,
    StoredEvaluation,
)

logger = logging.getLogger(__name__)

FAILURE_EVENT_SOURCE = "replay"


@dataclass(frozen=True)
class ReplayEngineConfig:
    """Behavioral knobs for one orchestrator; values come from typed settings."""

    lease_seconds: float = 120.0
    default_cycle_delay_seconds: float = 1.0
    simulated_cycle_duration_seconds: float = 1.0
    evaluation: DelayedEvaluationConfig = field(default_factory=DelayedEvaluationConfig)

    def __post_init__(self) -> None:
        if self.lease_seconds <= 0 or self.simulated_cycle_duration_seconds <= 0:
            raise ValueError("Lease and simulated cycle duration must be positive.")
        if self.default_cycle_delay_seconds < 0:
            raise ValueError("Default cycle delay must be non-negative.")


@dataclass(frozen=True)
class ReplayStatusReport:
    """One run's durable state plus its persisted delayed evaluations."""

    run: ReplayRunState
    evaluations: tuple[StoredEvaluation, ...]


class ReplayOrchestrator:
    """Drive held-out trajectories through the online system one cycle at a time."""

    def __init__(
        self,
        store: ReplayStateStore,
        source: ReplaySource,
        client: ReplayIngestionClient,
        config: ReplayEngineConfig | None = None,
        metrics: ReplayMetrics | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._store = store
        self._source = source
        self._client = client
        self._config = config or ReplayEngineConfig()
        self._metrics = metrics or ReplayMetrics()
        self._clock = clock
        self._sleep = sleeper
        self._trajectories: dict[tuple[str, int], ReplayTrajectory] = {}

    # ------------------------------------------------------------------ runs

    def start(
        self,
        source_asset_id: int,
        *,
        mode: ReplayMode = ReplayMode.STEP,
        cycle_delay_seconds: float | None = None,
        force_restart: bool = False,
    ) -> ReplayRunState:
        """Create a run for one replay-split asset, or return the existing one.

        Repeated starts are idempotent: an incomplete run is returned (and a
        paused or failed one reactivated) instead of duplicating operational
        assets. A completed run is returned unchanged unless ``force_restart``
        explicitly begins a new attempt with a fresh operational asset; earlier
        attempts and their operational history are never deleted.
        """
        trajectory = self._load_trajectory(source_asset_id)
        latest = self._store.latest_run_for_source(
            trajectory.dataset_name, trajectory.dataset_subset, source_asset_id
        )
        if latest is not None and not force_restart:
            if latest.status is ReplayRunStatus.COMPLETED:
                logger.info(
                    "replay_run_already_completed",
                    extra={"run_id": str(latest.run_id), "source_asset_id": source_asset_id},
                )
                return latest
            if latest.status is not ReplayRunStatus.CANCELLED:
                self._check_run_matches_source(latest, trajectory)
                if latest.status in (ReplayRunStatus.PAUSED, ReplayRunStatus.FAILED):
                    latest = self._store.mark_running(latest.run_id)
                logger.info(
                    "replay_run_resumed_existing",
                    extra={"run_id": str(latest.run_id), "status": latest.status.value},
                )
                return latest
        if (
            latest is not None
            and force_restart
            and latest.status
            not in (
                ReplayRunStatus.COMPLETED,
                ReplayRunStatus.CANCELLED,
            )
        ):
            self._store.cancel_run(latest.run_id)
            logger.info("replay_run_cancelled_for_restart", extra={"run_id": str(latest.run_id)})
        attempt = 1 if latest is None else latest.attempt + 1
        delay = (
            self._config.default_cycle_delay_seconds
            if cycle_delay_seconds is None
            else cycle_delay_seconds
        )
        command = NewReplayRun(
            dataset_name=trajectory.dataset_name,
            dataset_subset=trajectory.dataset_subset,
            source_asset_id=source_asset_id,
            external_asset_id=_external_asset_id(
                trajectory.dataset_subset, source_asset_id, attempt
            ),
            final_cycle=trajectory.final_cycle,
            mode=mode,
            cycle_delay_seconds=delay,
            simulated_cycle_duration_seconds=self._config.simulated_cycle_duration_seconds,
            replay_started_at=self._clock(),
            attempt=attempt,
            metadata={
                "source_checksums": dict(trajectory.source_checksums),
                "created_by": f"turbine-guard {__version__}",
            },
        )
        run = self._store.create_run(command)
        self._metrics.runs_started.inc()
        logger.info(
            "replay_run_started",
            extra={
                "run_id": str(run.run_id),
                "source_asset_id": source_asset_id,
                "attempt": attempt,
                "final_cycle": trajectory.final_cycle,
                "mode": mode.value,
            },
        )
        return run

    def step(self, run_id: uuid.UUID) -> ReplayRunState:
        """Advance exactly one cycle; finalize when it was the final cycle."""
        run = self._require_run(run_id)
        if run.status is ReplayRunStatus.COMPLETED:
            return run
        if run.ingest_complete:
            return self._finalize(run_id)
        state = self._advance_once(run_id)
        if state.ingest_complete:
            state = self._finalize(run_id)
        return state

    def drive(self, run_id: uuid.UUID, *, max_cycles: int | None = None) -> ReplayRunState:
        """Advance until completion, pause, failure, or the cycle budget."""
        if max_cycles is not None and max_cycles <= 0:
            raise ReplayStateError("max_cycles must be positive when provided.")
        advanced = 0
        self._metrics.active_runs.inc()
        try:
            while True:
                state = self._require_run(run_id)
                if state.status is ReplayRunStatus.COMPLETED:
                    return state
                if state.status not in (ReplayRunStatus.CREATED, ReplayRunStatus.RUNNING):
                    logger.info(
                        "replay_drive_stopped",
                        extra={"run_id": str(run_id), "status": state.status.value},
                    )
                    return state
                if state.ingest_complete:
                    return self._finalize(run_id)
                if max_cycles is not None and advanced >= max_cycles:
                    return state
                state = self._advance_once(run_id)
                advanced += 1
                if (
                    not state.ingest_complete
                    and state.mode is ReplayMode.CONTINUOUS
                    and state.cycle_delay_seconds > 0
                ):
                    self._sleep(state.cycle_delay_seconds)
        finally:
            self._metrics.active_runs.dec()

    def resume(self, run_id: uuid.UUID, *, max_cycles: int | None = None) -> ReplayRunState:
        """Reactivate a run and continue from the earliest incomplete phase."""
        run = self._require_run(run_id)
        if run.status is ReplayRunStatus.COMPLETED:
            return run
        run = self._store.mark_running(run_id)
        if run.ingest_complete:
            return self._finalize(run_id)
        budget = (
            max_cycles if max_cycles is not None else (1 if run.mode is ReplayMode.STEP else None)
        )
        return self.drive(run_id, max_cycles=budget)

    def stop(self, run_id: uuid.UUID) -> ReplayRunState:
        """Request a stop after the cycle currently in flight; resumable later."""
        state = self._store.request_stop(run_id)
        logger.info("replay_stop_requested", extra={"run_id": str(run_id)})
        return state

    def status(self, run_id: uuid.UUID) -> ReplayStatusReport:
        run = self._require_run(run_id)
        return ReplayStatusReport(run=run, evaluations=self._store.evaluations_for_run(run_id))

    def list_status(self, *, limit: int = 100) -> list[ReplayStatusReport]:
        return [
            ReplayStatusReport(run=run, evaluations=self._store.evaluations_for_run(run.run_id))
            for run in self._store.list_runs(limit=limit)
        ]

    def replay_asset_ids(self) -> tuple[int, ...]:
        return self._source.replay_asset_ids()

    # ------------------------------------------------------------- advancing

    def _advance_once(self, run_id: uuid.UUID) -> ReplayRunState:
        started = time.perf_counter()
        claim = self._store.claim_advance(run_id, lease_seconds=self._config.lease_seconds)
        run = claim.run
        try:
            trajectory = self._trajectory_for(run)
            request = build_reading_request(
                trajectory,
                claim.next_cycle,
                run_id=run.run_id,
                external_asset_id=run.external_asset_id,
                replay_started_at=run.replay_started_at,
                simulated_cycle_duration_seconds=run.simulated_cycle_duration_seconds,
            )
        except ReplaySourceError as exc:
            self._store.release_advance(run_id, token=claim.token)
            self._fail(run_id, str(exc))
            raise
        self._metrics.cycles_sent.inc()
        try:
            result = self._client.send_reading(request)
        except (ReplayIngestionError, ReplayTransientError) as exc:
            self._store.release_advance(run_id, token=claim.token)
            self._fail(run_id, str(exc))
            raise
        if result.retries:
            self._metrics.retries.inc(result.retries)
        if run.asset_id is not None and result.asset_id != run.asset_id:
            self._store.release_advance(run_id, token=claim.token)
            error = (
                f"Ingestion confirmed a different operational asset ({result.asset_id}) "
                f"than this run's recorded asset ({run.asset_id})."
            )
            self._fail(run_id, error)
            raise ReplayIngestionError(error)
        state = self._store.confirm_advance(
            run_id, token=claim.token, cycle=claim.next_cycle, asset_id=result.asset_id
        )
        self._metrics.cycles_accepted.inc()
        self._metrics.cycle_latency.observe(time.perf_counter() - started)
        logger.info(
            "replay_cycle_confirmed",
            extra={
                "run_id": str(run_id),
                "cycle": claim.next_cycle,
                "final_cycle": state.final_cycle,
                "idempotent": result.idempotent,
                "risk_level": result.risk_level,
            },
        )
        return state

    # ------------------------------------------------------------ finalizing

    def _finalize(self, run_id: uuid.UUID) -> ReplayRunState:
        """Run the delayed-feedback phases from the earliest incomplete one."""
        run = self._require_run(run_id)
        if run.status is ReplayRunStatus.COMPLETED:
            return run
        if not run.ingest_complete:
            raise ReplayStateError(f"Replay run {run_id} has not ingested its final cycle yet.")
        asset_id = run.asset_id or self._store.resolve_asset(run.external_asset_id)
        if asset_id is None:
            raise ReplayStateError(
                f"Operational asset {run.external_asset_id!r} could not be resolved."
            )
        try:
            event_id = run.failure_event_id
            if event_id is None:
                if not self._store.reading_exists(asset_id, run.final_cycle):
                    raise ReplayStateError(
                        "The final cycle reading is not persisted; refusing to emit a "
                        "failure event before ingestion is truly complete."
                    )
                event_id = self._store.record_failure_event(
                    run_id, _failure_event_command(run, asset_id)
                )
                self._metrics.failure_events.inc()
                logger.info(
                    "replay_failure_event_emitted",
                    extra={
                        "run_id": str(run_id),
                        "event_id": str(event_id),
                        "event_cycle": run.final_cycle,
                    },
                )
            run = self._require_run(run_id)
            predictions = self._store.stored_predictions(asset_id)
            if run.labels_backfilled_at is None:
                commands = build_outcome_commands(
                    predictions,
                    final_cycle=run.final_cycle,
                    asset_id=asset_id,
                    maintenance_event_id=event_id,
                    labeled_at=self._clock(),
                )
                self._store.record_outcomes(run_id, commands)
                self._metrics.label_backfills.inc()
                logger.info(
                    "replay_labels_backfilled",
                    extra={"run_id": str(run_id), "label_count": len(commands)},
                )
                run = self._require_run(run_id)
            if run.evaluation_completed_at is None:
                outcomes = self._store.outcomes_for_event(event_id)
                evaluations = per_asset_evaluations(
                    run, predictions, outcomes, self._config.evaluation
                )
                self._store.record_evaluations(run_id, evaluations)
                self._metrics.evaluations.inc()
                logger.info(
                    "replay_evaluation_persisted",
                    extra={"run_id": str(run_id), "model_version_count": len(evaluations)},
                )
        except ReplayOutcomeError as exc:
            self._fail(run_id, str(exc))
            raise
        state = self._store.complete_run(run_id)
        self._metrics.completed_runs.inc()
        logger.info(
            "replay_run_completed",
            extra={"run_id": str(run_id), "final_cycle": state.final_cycle},
        )
        return state

    # -------------------------------------------------------------- aggregate

    def evaluate_aggregate(self) -> list[StoredEvaluation]:
        """Persist one aggregate delayed evaluation per model version.

        Covers every completed replay run; repeated calls over the same set of
        completed runs return the existing rows instead of inserting again.
        """
        runs = [
            run
            for run in self._store.list_runs(limit=1000)
            if run.status is ReplayRunStatus.COMPLETED
        ]
        if not runs:
            raise ReplayStateError("No completed replay runs exist to aggregate.")
        frames = []
        for run in runs:
            asset_id = run.asset_id or self._store.resolve_asset(run.external_asset_id)
            if asset_id is None or run.failure_event_id is None:
                raise ReplayStateError(
                    f"Completed run {run.run_id} is missing its asset or failure event."
                )
            frames.append(
                build_outcome_frame(
                    self._store.stored_predictions(asset_id),
                    self._store.outcomes_for_event(run.failure_event_id),
                    source_asset_id=run.source_asset_id,
                )
            )
        commands = aggregate_evaluations(runs, frames, self._config.evaluation)
        run_ids = sorted(str(run.run_id) for run in runs)
        persisted: list[StoredEvaluation] = []
        new_commands = []
        for command in commands:
            existing = self._existing_aggregate(command.model_name, command.model_version, run_ids)
            if existing is not None:
                persisted.append(existing)
            else:
                new_commands.append(command)
        if new_commands:
            self._store.record_aggregate_evaluations(new_commands)
            self._metrics.evaluations.inc(len(new_commands))
        for command in new_commands:
            refreshed = self._existing_aggregate(command.model_name, command.model_version, run_ids)
            if refreshed is not None:
                persisted.append(refreshed)
        logger.info(
            "replay_aggregate_evaluated",
            extra={"run_count": len(runs), "model_version_count": len(commands)},
        )
        return persisted

    def _existing_aggregate(
        self, model_name: str, model_version: str, run_ids: list[str]
    ) -> StoredEvaluation | None:
        for evaluation in self._store.evaluations_for_model(model_name, model_version):
            metrics = evaluation.metrics
            if (
                metrics.get("aggregation") == AGGREGATE_AGGREGATION
                and sorted(metrics.get("replay_run_ids", [])) == run_ids
            ):
                return evaluation
        return None

    # --------------------------------------------------------------- helpers

    def _require_run(self, run_id: uuid.UUID) -> ReplayRunState:
        run = self._store.get_run(run_id)
        if run is None:
            raise ReplayStateError(f"Replay run {run_id} does not exist.")
        return run

    def _fail(self, run_id: uuid.UUID, error: str) -> None:
        self._metrics.failures.inc()
        self._store.mark_failed(run_id, error)
        logger.error("replay_run_failed", extra={"run_id": str(run_id), "error": error})

    def _load_trajectory(self, source_asset_id: int) -> ReplayTrajectory:
        trajectory = self._source.load_trajectory(source_asset_id)
        self._trajectories[(trajectory.dataset_subset, source_asset_id)] = trajectory
        return trajectory

    def _trajectory_for(self, run: ReplayRunState) -> ReplayTrajectory:
        key = (run.dataset_subset, run.source_asset_id)
        trajectory = self._trajectories.get(key)
        if trajectory is None:
            trajectory = self._load_trajectory(run.source_asset_id)
        self._check_run_matches_source(run, trajectory)
        return trajectory

    def _check_run_matches_source(self, run: ReplayRunState, trajectory: ReplayTrajectory) -> None:
        if run.final_cycle != trajectory.final_cycle:
            raise ReplaySourceError(
                f"Replay run {run.run_id} recorded final cycle {run.final_cycle} but the "
                f"verified source now ends at {trajectory.final_cycle}; the input changed."
            )
        recorded = run.metadata.get("source_checksums")
        if recorded is not None and recorded != trajectory.source_checksums:
            raise ReplaySourceError(
                f"Replay run {run.run_id} was created from different source checksums; "
                "refusing to continue with changed input."
            )


def _external_asset_id(dataset_subset: str, source_asset_id: int, attempt: int) -> str:
    base = f"replay-{dataset_subset}-{source_asset_id:03d}"
    return base if attempt == 1 else f"{base}-r{attempt}"


def _failure_event_command(run: ReplayRunState, asset_id: uuid.UUID) -> NewMaintenanceEvent:
    occurred_at = run.replay_started_at + timedelta(
        seconds=(run.final_cycle - 1) * run.simulated_cycle_duration_seconds
    )
    return NewMaintenanceEvent(
        asset_id=asset_id,
        event_type=MaintenanceEventType.FAILURE,
        occurred_at=occurred_at,
        source=FAILURE_EVENT_SOURCE,
        event_cycle=run.final_cycle,
        external_event_id=f"replay-run:{run.run_id}:failure",
        description=(
            f"Simulated end-of-trajectory failure replayed from "
            f"{run.dataset_name} {run.dataset_subset} unit {run.source_asset_id}."
        ),
        metadata={
            "replay_run_id": str(run.run_id),
            "dataset_name": run.dataset_name,
            "dataset_subset": run.dataset_subset,
            "source_asset_id": run.source_asset_id,
            "attempt": run.attempt,
            "simulated_timestamps": True,
        },
    )

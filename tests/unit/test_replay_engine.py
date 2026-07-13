"""Replay orchestrator lifecycle, isolation, recovery, and concurrency."""

import uuid
from types import SimpleNamespace

import pytest
from tests.conftest import make_trajectory_frame
from tests.unit.replay_fakes import make_replay_system

from turbine_guard.database.enums import ReplayMode, ReplayRunStatus
from turbine_guard.replay.client import build_reading_request
from turbine_guard.replay.engine import ReplayEngineConfig, ReplayOrchestrator
from turbine_guard.replay.errors import (
    ReplayConcurrencyError,
    ReplayIngestionError,
    ReplaySourceError,
    ReplayStateError,
    ReplayTransientError,
)
from turbine_guard.replay.evaluation import AGGREGATE_AGGREGATION
from turbine_guard.replay.source import ReplayTrajectory

SENSOR_PAYLOAD_KEYS = {
    "external_asset_id",
    "cycle",
    "observed_at",
    *{f"operating_setting_{index}" for index in range(1, 4)},
    *{f"sensor_{index:02d}" for index in range(1, 22)},
    "source",
    "ingestion_id",
    "schema_version",
}


def _trajectory(asset_id: int, length: int) -> ReplayTrajectory:
    frame = make_trajectory_frame({asset_id: length})
    return ReplayTrajectory(
        dataset_name="cmapss",
        dataset_subset="FD001",
        source_asset_id=asset_id,
        final_cycle=length,
        frame=frame,
        source_checksums={"trajectory_parquet_sha256": f"sha-{asset_id}"},
    )


def _make(lengths: dict[int, int] | None = None, prediction_fn=None) -> SimpleNamespace:
    from tests.unit.replay_fakes import FakeReplaySource

    lengths = lengths or {9: 6, 13: 4}
    system, store, client, clock = make_replay_system(prediction_fn)
    source = FakeReplaySource({aid: _trajectory(aid, n) for aid, n in lengths.items()})
    sleeps: list[float] = []
    engine = ReplayOrchestrator(
        store,
        source,
        client,
        ReplayEngineConfig(
            lease_seconds=60.0,
            default_cycle_delay_seconds=0.25,
            simulated_cycle_duration_seconds=1.0,
        ),
        clock=clock,
        sleeper=sleeps.append,
    )
    return SimpleNamespace(
        engine=engine,
        system=system,
        store=store,
        client=client,
        clock=clock,
        source=source,
        sleeps=sleeps,
        lengths=lengths,
    )


class TestStartRun:
    def test_start_creates_durable_run_with_source_mapping(self) -> None:
        env = _make()
        run = env.engine.start(9, mode=ReplayMode.ACCELERATED)
        assert run.source_asset_id == 9
        assert run.external_asset_id == "replay-FD001-009"
        assert run.attempt == 1
        assert run.final_cycle == 6
        assert run.last_confirmed_cycle == 0
        assert run.status is ReplayRunStatus.CREATED
        assert run.metadata["source_checksums"] == {"trajectory_parquet_sha256": "sha-9"}
        assert env.store.get_run(run.run_id) is not None

    def test_repeated_start_is_idempotent(self) -> None:
        env = _make()
        first = env.engine.start(9, mode=ReplayMode.ACCELERATED)
        second = env.engine.start(9, mode=ReplayMode.ACCELERATED)
        assert first.run_id == second.run_id
        assert len(env.store.list_runs()) == 1

    def test_non_replay_asset_is_rejected(self) -> None:
        env = _make()
        with pytest.raises(ReplaySourceError):
            env.engine.start(999, mode=ReplayMode.ACCELERATED)


class TestStepMode:
    def test_step_advances_exactly_one_cycle_durably(self) -> None:
        env = _make()
        run = env.engine.start(9, mode=ReplayMode.STEP)
        state = env.engine.step(run.run_id)
        assert state.last_confirmed_cycle == 1
        assert state.status is ReplayRunStatus.RUNNING
        assert len(env.system.readings) == 1
        reloaded = env.store.get_run(run.run_id)
        assert reloaded is not None
        assert reloaded.last_confirmed_cycle == 1

    def test_steps_send_cycles_in_order_without_gaps(self) -> None:
        env = _make()
        run = env.engine.start(9, mode=ReplayMode.STEP)
        for expected in (1, 2, 3):
            state = env.engine.step(run.run_id)
            assert state.last_confirmed_cycle == expected
        sent = [payload["cycle"] for payload in env.system.sent_payloads]
        assert sent == [1, 2, 3]


class TestCompletion:
    def test_accelerated_drive_completes_full_lifecycle(self) -> None:
        env = _make()
        run = env.engine.start(9, mode=ReplayMode.ACCELERATED)
        state = env.engine.drive(run.run_id)
        assert state.status is ReplayRunStatus.COMPLETED
        assert state.last_confirmed_cycle == 6
        assert state.completed_at is not None
        assert env.sleeps == []  # accelerated mode never waits

        # one failure event, at the final cycle, with the deterministic external ID
        assert len(env.system.events) == 1
        external_id, stored = next(iter(env.system.events.items()))
        assert external_id == f"replay-run:{run.run_id}:failure"
        assert stored.command.event_cycle == 6
        assert stored.command.metadata["source_asset_id"] == 9

        # labels: one per prediction, final cycle realizes zero, minus one per cycle
        realized = sorted(
            (command.cycle, command.realized_rul) for command in env.system.outcomes.values()
        )
        assert realized == [(1, 5), (2, 4), (3, 3), (4, 2), (5, 1), (6, 0)]

        # evaluation persisted and linked to the run
        evaluations = env.store.evaluations_for_run(run.run_id)
        assert len(evaluations) == 1
        assert evaluations[0].sample_count == 6

    def test_continuous_drive_waits_between_cycles(self) -> None:
        env = _make(lengths={13: 4})
        run = env.engine.start(13, mode=ReplayMode.CONTINUOUS, cycle_delay_seconds=0.5)
        state = env.engine.drive(run.run_id)
        assert state.status is ReplayRunStatus.COMPLETED
        assert env.sleeps == [0.5, 0.5, 0.5]  # no wait after the final cycle

    def test_max_cycles_bounds_advancement_without_finalizing(self) -> None:
        env = _make()
        run = env.engine.start(9, mode=ReplayMode.ACCELERATED)
        state = env.engine.drive(run.run_id, max_cycles=2)
        assert state.last_confirmed_cycle == 2
        assert state.status is ReplayRunStatus.RUNNING
        assert env.system.events == {}
        assert env.system.outcomes == {}

    def test_completed_run_is_idempotent_for_drive_start_and_step(self) -> None:
        env = _make()
        run = env.engine.start(9, mode=ReplayMode.ACCELERATED)
        env.engine.drive(run.run_id)
        events = dict(env.system.events)
        outcomes = dict(env.system.outcomes)
        evaluation_count = len(env.system.evaluations)
        readings = dict(env.system.readings)

        assert env.engine.drive(run.run_id).status is ReplayRunStatus.COMPLETED
        assert env.engine.step(run.run_id).status is ReplayRunStatus.COMPLETED
        restart = env.engine.start(9, mode=ReplayMode.ACCELERATED)
        assert restart.run_id == run.run_id
        assert env.system.events == events
        assert env.system.outcomes == outcomes
        assert len(env.system.evaluations) == evaluation_count
        assert env.system.readings == readings


class TestGroundTruthIsolation:
    def test_only_schema_fields_are_ever_sent(self) -> None:
        env = _make()
        run = env.engine.start(9, mode=ReplayMode.ACCELERATED)
        env.engine.drive(run.run_id)
        for payload in env.system.sent_payloads:
            assert set(payload) <= SENSOR_PAYLOAD_KEYS

    def test_cycles_are_never_sent_ahead_of_confirmed_progress(self) -> None:
        env = _make()
        run = env.engine.start(9, mode=ReplayMode.ACCELERATED)
        env.engine.drive(run.run_id)
        sent = [payload["cycle"] for payload in env.system.sent_payloads]
        assert sent == sorted(sent)
        assert sent == list(range(1, 7))

    def test_no_failure_event_or_labels_before_final_cycle(self) -> None:
        env = _make()
        run = env.engine.start(9, mode=ReplayMode.STEP)
        for _ in range(5):  # everything except the final cycle
            env.engine.step(run.run_id)
        assert env.system.events == {}
        assert env.system.outcomes == {}
        state = env.store.get_run(run.run_id)
        assert state is not None
        assert state.failure_event_id is None

    def test_future_source_mutation_cannot_change_already_sent_payloads(self) -> None:
        env = _make()
        run = env.engine.start(9, mode=ReplayMode.STEP)
        for _ in range(3):
            env.engine.step(run.run_id)
        sent_before = [dict(payload) for payload in env.system.sent_payloads]
        trajectory = env.source.trajectories[9]
        future = trajectory.frame["cycle"] > 3
        trajectory.frame.loc[future, "sensor_04"] = 123_456.0
        env.engine.drive(run.run_id)
        assert env.system.sent_payloads[:3] == sent_before


class TestPauseResumeStop:
    def test_stop_pauses_after_current_cycle_and_resume_continues(self) -> None:
        env = _make()
        run = env.engine.start(9, mode=ReplayMode.CONTINUOUS, cycle_delay_seconds=0.1)
        stopping = env.engine

        def stop_during_sleep(_: float) -> None:
            stopping.stop(run.run_id)

        env.engine._sleep = stop_during_sleep  # stop request lands mid-run
        state = env.engine.drive(run.run_id)
        assert state.status is ReplayRunStatus.PAUSED
        assert 0 < state.last_confirmed_cycle < state.final_cycle
        confirmed = state.last_confirmed_cycle

        env.engine._sleep = lambda _: None
        resumed = env.engine.resume(run.run_id)
        assert resumed.status is ReplayRunStatus.COMPLETED
        assert resumed.last_confirmed_cycle == 6
        assert confirmed < resumed.last_confirmed_cycle

    def test_paused_run_cannot_step_without_resume(self) -> None:
        env = _make()
        run = env.engine.start(9, mode=ReplayMode.STEP)
        env.engine.step(run.run_id)
        env.engine.stop(run.run_id)
        with pytest.raises(ReplayStateError, match="cannot advance"):
            env.engine.step(run.run_id)

    def test_resume_on_step_mode_advances_one_cycle(self) -> None:
        env = _make()
        run = env.engine.start(9, mode=ReplayMode.STEP)
        env.engine.step(run.run_id)
        env.engine.stop(run.run_id)
        state = env.engine.resume(run.run_id)
        assert state.last_confirmed_cycle == 2
        assert state.status is ReplayRunStatus.RUNNING


class TestForceRestart:
    def test_force_restart_cancels_incomplete_run_and_creates_new_attempt(self) -> None:
        env = _make()
        first = env.engine.start(9, mode=ReplayMode.STEP)
        env.engine.step(first.run_id)
        second = env.engine.start(9, mode=ReplayMode.ACCELERATED, force_restart=True)
        assert second.run_id != first.run_id
        assert second.attempt == 2
        assert second.external_asset_id == "replay-FD001-009-r2"
        cancelled = env.store.get_run(first.run_id)
        assert cancelled is not None
        assert cancelled.status is ReplayRunStatus.CANCELLED

    def test_force_restart_preserves_completed_run_and_its_data(self) -> None:
        env = _make()
        first = env.engine.start(9, mode=ReplayMode.ACCELERATED)
        env.engine.drive(first.run_id)
        events = dict(env.system.events)
        second = env.engine.start(9, mode=ReplayMode.ACCELERATED, force_restart=True)
        env.engine.drive(second.run_id)
        preserved = env.store.get_run(first.run_id)
        assert preserved is not None
        assert preserved.status is ReplayRunStatus.COMPLETED
        for external_id in events:
            assert external_id in env.system.events
        assert len(env.system.events) == 2  # one failure per attempt
        assert "replay-FD001-009" in env.system.assets
        assert "replay-FD001-009-r2" in env.system.assets

    def test_changed_source_is_refused_on_resume(self) -> None:
        env = _make()
        run = env.engine.start(9, mode=ReplayMode.STEP)
        env.engine.step(run.run_id)
        env.source.trajectories[9] = _trajectory(9, 6)  # same length, new checksums
        env.source.trajectories[9].source_checksums["trajectory_parquet_sha256"] = "sha-tampered"
        env.engine._trajectories.clear()
        with pytest.raises(ReplaySourceError, match="different source checksums"):
            env.engine.step(run.run_id)


class TestRecovery:
    def test_api_accepted_but_progress_not_updated_reconciles_idempotently(self) -> None:
        env = _make()
        run = env.engine.start(9, mode=ReplayMode.ACCELERATED)
        env.engine.drive(run.run_id, max_cycles=1)

        # simulate a crash: cycle 2 accepted by the API, no confirmation recorded
        claim = env.store.claim_advance(run.run_id, lease_seconds=60)
        request = build_reading_request(
            env.source.trajectories[9],
            claim.next_cycle,
            run_id=run.run_id,
            external_asset_id=run.external_asset_id,
            replay_started_at=run.replay_started_at,
            simulated_cycle_duration_seconds=run.simulated_cycle_duration_seconds,
        )
        env.client.send_reading(request)
        assert len(env.system.readings) == 2

        env.clock.advance(120)  # crashed worker's lease expires
        state = env.engine.resume(run.run_id)
        assert state.status is ReplayRunStatus.COMPLETED
        assert len(env.system.readings) == 6  # no duplicates, no gaps
        idempotent_resend = [payload["cycle"] for payload in env.system.sent_payloads].count(2)
        assert idempotent_resend == 2  # original send plus reconciling resend

    def test_failure_event_exists_but_backfill_missing_resumes_at_backfill(self) -> None:
        env = _make()
        run = env.engine.start(9, mode=ReplayMode.ACCELERATED)
        original = env.store.record_outcomes
        env.store.record_outcomes = lambda *_a, **_k: (_ for _ in ()).throw(
            ReplayTransientError("simulated crash after failure event")
        )
        with pytest.raises(ReplayTransientError):
            env.engine.drive(run.run_id)
        interrupted = env.store.get_run(run.run_id)
        assert interrupted is not None
        assert interrupted.failure_event_id is not None
        assert interrupted.labels_backfilled_at is None
        assert len(env.system.events) == 1

        env.store.record_outcomes = original
        state = env.engine.resume(run.run_id)
        assert state.status is ReplayRunStatus.COMPLETED
        assert len(env.system.events) == 1  # event was not duplicated
        assert len(env.system.outcomes) == 6

    def test_backfill_done_but_evaluation_missing_resumes_at_evaluation(self) -> None:
        env = _make()
        run = env.engine.start(9, mode=ReplayMode.ACCELERATED)
        original = env.store.record_evaluations
        env.store.record_evaluations = lambda *_a, **_k: (_ for _ in ()).throw(
            ReplayTransientError("simulated crash after backfill")
        )
        with pytest.raises(ReplayTransientError):
            env.engine.drive(run.run_id)
        interrupted = env.store.get_run(run.run_id)
        assert interrupted is not None
        assert interrupted.labels_backfilled_at is not None
        assert interrupted.evaluation_completed_at is None
        outcome_count = len(env.system.outcomes)

        env.store.record_evaluations = original
        state = env.engine.resume(run.run_id)
        assert state.status is ReplayRunStatus.COMPLETED
        assert len(env.system.outcomes) == outcome_count  # backfill not repeated
        assert len(env.store.evaluations_for_run(run.run_id)) == 1

    def test_transient_exhaustion_marks_run_failed_and_resume_recovers(self) -> None:
        env = _make()
        run = env.engine.start(9, mode=ReplayMode.ACCELERATED)
        env.client.fail_next.append(ReplayTransientError("API kept returning 503"))
        with pytest.raises(ReplayTransientError):
            env.engine.drive(run.run_id)
        failed = env.store.get_run(run.run_id)
        assert failed is not None
        assert failed.status is ReplayRunStatus.FAILED
        assert failed.error_message is not None
        assert "503" in failed.error_message

        state = env.engine.resume(run.run_id)
        assert state.status is ReplayRunStatus.COMPLETED
        assert state.error_message is None

    def test_permanent_conflict_marks_run_failed(self) -> None:
        env = _make()
        run = env.engine.start(9, mode=ReplayMode.ACCELERATED)
        env.client.fail_next.append(
            ReplayIngestionError("HTTP 409 sensor_reading_conflict: different data")
        )
        with pytest.raises(ReplayIngestionError):
            env.engine.drive(run.run_id)
        failed = env.store.get_run(run.run_id)
        assert failed is not None
        assert failed.status is ReplayRunStatus.FAILED


class TestConcurrency:
    def test_competing_worker_cannot_send_the_same_next_cycle(self) -> None:
        env = _make()
        run = env.engine.start(9, mode=ReplayMode.STEP)
        env.store.claim_advance(run.run_id, lease_seconds=60)  # a rival worker holds the lease
        payloads_before = len(env.system.sent_payloads)
        with pytest.raises(ReplayConcurrencyError, match="lease"):
            env.engine.step(run.run_id)
        assert len(env.system.sent_payloads) == payloads_before  # nothing was sent

    def test_expired_lease_can_be_reclaimed_and_stale_confirm_is_rejected(self) -> None:
        env = _make()
        run = env.engine.start(9, mode=ReplayMode.STEP)
        stale = env.store.claim_advance(run.run_id, lease_seconds=1)
        env.clock.advance(5)
        state = env.engine.step(run.run_id)  # reclaims after expiry
        assert state.last_confirmed_cycle == 1
        with pytest.raises(ReplayConcurrencyError):
            env.store.confirm_advance(run.run_id, token=stale.token, cycle=2, asset_id=uuid.uuid4())


class TestMultipleModelVersions:
    def test_champion_change_mid_life_produces_per_version_evaluations(self) -> None:
        env = _make()
        run = env.engine.start(9, mode=ReplayMode.STEP)
        for _ in range(3):
            env.engine.step(run.run_id)
        env.system.model_version = "2"  # champion promoted mid-lifecycle
        state = env.engine.drive(run.run_id)
        assert state.status is ReplayRunStatus.COMPLETED
        evaluations = env.store.evaluations_for_run(run.run_id)
        by_version = {evaluation.model_version: evaluation for evaluation in evaluations}
        assert set(by_version) == {"1", "2"}
        assert by_version["1"].sample_count == 3
        assert by_version["2"].sample_count == 3


class TestAggregate:
    def test_aggregate_covers_completed_runs_and_repeats_idempotently(self) -> None:
        env = _make()
        for asset_id in (9, 13):
            run = env.engine.start(asset_id, mode=ReplayMode.ACCELERATED)
            env.engine.drive(run.run_id)
        first = env.engine.evaluate_aggregate()
        assert len(first) == 1
        assert first[0].metrics["aggregation"] == AGGREGATE_AGGREGATION
        assert first[0].sample_count == 10  # 6 + 4 labeled cycles
        stored_count = len(env.system.evaluations)
        second = env.engine.evaluate_aggregate()
        assert len(second) == 1
        assert len(env.system.evaluations) == stored_count  # nothing inserted twice

    def test_aggregate_requires_completed_runs(self) -> None:
        env = _make()
        with pytest.raises(ReplayStateError, match="No completed replay runs"):
            env.engine.evaluate_aggregate()

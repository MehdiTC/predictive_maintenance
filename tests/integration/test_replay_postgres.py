"""End-to-end replay over real PostgreSQL and the real Loop 7 HTTP contract."""

import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session, sessionmaker
from tests.integration.test_online_inference_postgres import (
    FakeLoader,
    _client,
    _settings,
)

from turbine_guard.database.commands import (
    NewAsset,
    NewMaintenanceEvent,
    NewPredictionOutcome,
    NewSensorReading,
)
from turbine_guard.database.enums import (
    MaintenanceEventType,
    ReplayMode,
    ReplayRunStatus,
    RiskLevel,
)
from turbine_guard.database.errors import PredictionOutcomeConflictError
from turbine_guard.database.models import (
    MaintenanceEvent,
    Prediction,
    PredictionOutcome,
    SensorReading,
)
from turbine_guard.database.repositories import (
    AssetRepository,
    MaintenanceEventRepository,
    PredictionOutcomeRepository,
    PredictionRepository,
    SensorReadingRepository,
)
from turbine_guard.replay.client import (
    ReplayClientConfig,
    ReplayIngestionClient,
    build_reading_request,
)
from turbine_guard.replay.engine import ReplayEngineConfig, ReplayOrchestrator
from turbine_guard.replay.errors import ReplayConcurrencyError, ReplayTransientError
from turbine_guard.replay.source import ReplaySource, ReplaySourceConfig
from turbine_guard.replay.state import PostgresReplayStateStore

pytestmark = pytest.mark.postgres

NOW = datetime(2026, 7, 13, 12, 0, tzinfo=UTC)


class SettableClock:
    def __init__(self) -> None:
        self.now = NOW

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


def _factory(db_session: Session) -> sessionmaker[Session]:
    """Sessions joining the test's outer transaction through savepoints."""
    return sessionmaker(
        bind=db_session.connection(),
        class_=Session,
        expire_on_commit=False,
        autoflush=False,
        join_transaction_mode="create_savepoint",
    )


def _orchestrator(
    data_dir: Path,
    db_session: Session,
    http: TestClient,
    clock: SettableClock,
) -> tuple[ReplayOrchestrator, PostgresReplayStateStore]:
    store = PostgresReplayStateStore(_factory(db_session), clock=clock)
    source = ReplaySource(ReplaySourceConfig(data_dir=data_dir))
    client = ReplayIngestionClient(http, ReplayClientConfig(max_attempts=3, backoff_seconds=0.0))
    engine = ReplayOrchestrator(
        store,
        source,
        client,
        ReplayEngineConfig(
            lease_seconds=30.0,
            default_cycle_delay_seconds=0.0,
            simulated_cycle_duration_seconds=1.0,
        ),
        clock=clock,
    )
    return engine, store


def test_complete_replay_lifecycle_through_http(
    db_session: Session, postgres_engine: Engine, feature_data_dir: Path
) -> None:
    loader = FakeLoader()
    settings = _settings(postgres_engine)
    clock = SettableClock()
    with _client(db_session, settings, loader) as http:
        engine, store = _orchestrator(feature_data_dir, db_session, http, clock)
        source_asset = engine.replay_asset_ids()[0]

        # -- partial replay: nothing about the future is visible anywhere
        run = engine.start(source_asset, mode=ReplayMode.STEP)
        for _ in range(3):
            state = engine.step(run.run_id)
        assert state.last_confirmed_cycle == 3
        asset_uuid = store.resolve_asset(run.external_asset_id)
        assert asset_uuid is not None
        detail = http.get(f"/v1/assets/{asset_uuid}").json()
        assert detail["latest_reading"]["cycle"] == 3
        assert detail["recent_maintenance_events"] == []
        assert db_session.scalars(select(PredictionOutcome)).all() == []
        assert db_session.scalars(select(MaintenanceEvent)).all() == []
        early_predictions = {
            prediction.cycle: (prediction.id, prediction.predicted_rul)
            for prediction in db_session.scalars(
                select(Prediction).where(Prediction.asset_id == asset_uuid)
            )
        }
        assert set(early_predictions) == {1, 2, 3}

        # -- run to completion: failure, backfill, evaluation, completed state
        final = engine.drive(run.run_id)
        assert final.status is ReplayRunStatus.COMPLETED
        assert final.last_confirmed_cycle == final.final_cycle
        assert final.completed_at is not None

        readings = db_session.scalars(
            select(SensorReading).where(SensorReading.asset_id == asset_uuid)
        ).all()
        assert sorted(reading.cycle for reading in readings) == list(
            range(1, final.final_cycle + 1)
        )

        events = db_session.scalars(
            select(MaintenanceEvent).where(MaintenanceEvent.asset_id == asset_uuid)
        ).all()
        assert len(events) == 1
        event = events[0]
        assert event.event_type is MaintenanceEventType.FAILURE
        assert event.event_cycle == final.final_cycle
        assert event.external_event_id == f"replay-run:{run.run_id}:failure"

        outcomes = db_session.scalars(
            select(PredictionOutcome).where(PredictionOutcome.asset_id == asset_uuid)
        ).all()
        realized = {outcome.cycle: outcome.realized_rul for outcome in outcomes}
        assert realized == {
            cycle: final.final_cycle - cycle for cycle in range(1, final.final_cycle + 1)
        }
        assert realized[final.final_cycle] == 0

        # original predictions remain byte-identical after backfill
        for prediction in db_session.scalars(
            select(Prediction).where(Prediction.asset_id == asset_uuid)
        ):
            if prediction.cycle in early_predictions:
                identity, value = early_predictions[prediction.cycle]
                assert prediction.id == identity
                assert prediction.predicted_rul == value

        evaluations = store.evaluations_for_run(run.run_id)
        assert len(evaluations) == 1
        evaluation = evaluations[0]
        assert evaluation.evaluation_scope == "replay"
        assert evaluation.sample_count == final.final_cycle
        assert evaluation.metrics["source_asset_id"] == source_asset

        # -- completed-run idempotency: repeating adds nothing
        assert engine.drive(run.run_id).status is ReplayRunStatus.COMPLETED
        assert engine.start(source_asset, mode=ReplayMode.STEP).run_id == run.run_id
        assert (
            len(
                db_session.scalars(
                    select(MaintenanceEvent).where(MaintenanceEvent.asset_id == asset_uuid)
                ).all()
            )
            == 1
        )

        # -- aggregate evaluation over the completed run set
        aggregates = engine.evaluate_aggregate()
        assert len(aggregates) == 1
        assert aggregates[0].metrics["aggregation"] == "replay_aggregate"
        again = engine.evaluate_aggregate()
        assert len(again) == 1


def test_uncertain_outcome_recovers_by_idempotent_resend(
    db_session: Session, postgres_engine: Engine, feature_data_dir: Path
) -> None:
    loader = FakeLoader()
    clock = SettableClock()
    with _client(db_session, _settings(postgres_engine), loader) as http:
        engine, store = _orchestrator(feature_data_dir, db_session, http, clock)
        source_asset = engine.replay_asset_ids()[0]
        run = engine.start(source_asset, mode=ReplayMode.ACCELERATED)
        engine.drive(run.run_id, max_cycles=2)

        # crash simulation: the API accepted cycle 3 but progress was never updated
        claim = store.claim_advance(run.run_id, lease_seconds=30)
        trajectory = ReplaySource(ReplaySourceConfig(data_dir=feature_data_dir)).load_trajectory(
            source_asset
        )
        request = build_reading_request(
            trajectory,
            claim.next_cycle,
            run_id=run.run_id,
            external_asset_id=run.external_asset_id,
            replay_started_at=run.replay_started_at,
            simulated_cycle_duration_seconds=run.simulated_cycle_duration_seconds,
        )
        ReplayIngestionClient(http).send_reading(request)
        stale = store.get_run(run.run_id)
        assert stale is not None
        assert stale.last_confirmed_cycle == 2

        # the lease of the crashed worker blocks rivals until it expires
        with pytest.raises(ReplayConcurrencyError):
            store.claim_advance(run.run_id, lease_seconds=30)
        clock.advance(60)

        final = engine.resume(run.run_id)
        assert final.status is ReplayRunStatus.COMPLETED
        asset_uuid = store.resolve_asset(run.external_asset_id)
        assert asset_uuid is not None
        cycles = [
            reading.cycle
            for reading in db_session.scalars(
                select(SensorReading).where(SensorReading.asset_id == asset_uuid)
            )
        ]
        assert sorted(cycles) == list(range(1, final.final_cycle + 1))
        assert len(cycles) == len(set(cycles))


def test_partial_finalize_phases_resume_without_duplication(
    db_session: Session, postgres_engine: Engine, feature_data_dir: Path
) -> None:
    loader = FakeLoader()
    clock = SettableClock()
    with _client(db_session, _settings(postgres_engine), loader) as http:
        engine, store = _orchestrator(feature_data_dir, db_session, http, clock)
        source_asset = engine.replay_asset_ids()[0]
        run = engine.start(source_asset, mode=ReplayMode.ACCELERATED)

        original_record_outcomes = store.record_outcomes
        store.record_outcomes = lambda *_a, **_k: (_ for _ in ()).throw(  # type: ignore[method-assign]
            ReplayTransientError("simulated crash between phases")
        )
        with pytest.raises(ReplayTransientError):
            engine.drive(run.run_id)
        interrupted = store.get_run(run.run_id)
        assert interrupted is not None
        assert interrupted.failure_event_id is not None
        assert interrupted.labels_backfilled_at is None
        assert db_session.scalars(select(PredictionOutcome)).all() == []

        store.record_outcomes = original_record_outcomes  # type: ignore[method-assign]
        final = engine.resume(run.run_id)
        assert final.status is ReplayRunStatus.COMPLETED
        events = db_session.scalars(select(MaintenanceEvent)).all()
        assert len(events) == 1  # the failure event was reused, not duplicated
        assert len(store.evaluations_for_run(run.run_id)) == 1


def test_force_restart_uses_fresh_operational_asset_and_preserves_history(
    db_session: Session, postgres_engine: Engine, feature_data_dir: Path
) -> None:
    loader = FakeLoader()
    clock = SettableClock()
    with _client(db_session, _settings(postgres_engine), loader) as http:
        engine, store = _orchestrator(feature_data_dir, db_session, http, clock)
        source_asset = engine.replay_asset_ids()[0]
        first = engine.start(source_asset, mode=ReplayMode.ACCELERATED)
        engine.drive(first.run_id)

        second = engine.start(source_asset, mode=ReplayMode.ACCELERATED, force_restart=True)
        assert second.attempt == 2
        assert second.external_asset_id.endswith("-r2")
        engine.drive(second.run_id)

        preserved = store.get_run(first.run_id)
        assert preserved is not None
        assert preserved.status is ReplayRunStatus.COMPLETED
        first_asset = store.resolve_asset(first.external_asset_id)
        second_asset = store.resolve_asset(second.external_asset_id)
        assert first_asset is not None
        assert second_asset is not None
        assert first_asset != second_asset
        events = db_session.scalars(select(MaintenanceEvent)).all()
        assert len(events) == 2


def test_prediction_outcome_repository_is_idempotent_and_detects_conflicts(
    db_session: Session,
) -> None:
    assets = AssetRepository(db_session)
    asset = assets.create(NewAsset(external_id="outcome-asset"))
    reading = SensorReadingRepository(db_session).insert(
        NewSensorReading(
            asset_id=asset.id,
            cycle=1,
            observed_at=NOW,
            operating_settings=(0.1, 0.2, 100.0),
            sensor_values=tuple(float(index) for index in range(1, 22)),  # type: ignore[arg-type]
            schema_version="1",
            source="replay",
        )
    )
    prediction = PredictionRepository(db_session).create(_new_prediction(asset.id, reading.id))
    event = MaintenanceEventRepository(db_session).create(
        NewMaintenanceEvent(
            asset_id=asset.id,
            event_type=MaintenanceEventType.FAILURE,
            occurred_at=NOW,
            source="replay",
            event_cycle=5,
            external_event_id="replay-run:test:failure",
        )
    )
    repository = PredictionOutcomeRepository(db_session)
    outcome = repository.create(
        NewPredictionOutcome(
            prediction_id=prediction.id,
            maintenance_event_id=event.id,
            asset_id=asset.id,
            cycle=1,
            realized_rul=4,
            labeled_at=NOW,
        )
    )
    retry = repository.create(
        NewPredictionOutcome(
            prediction_id=prediction.id,
            maintenance_event_id=event.id,
            asset_id=asset.id,
            cycle=1,
            realized_rul=4,
            labeled_at=NOW + timedelta(minutes=5),  # different retry time is still exact
        )
    )
    assert retry.id == outcome.id
    with pytest.raises(PredictionOutcomeConflictError):
        repository.create(
            NewPredictionOutcome(
                prediction_id=prediction.id,
                maintenance_event_id=event.id,
                asset_id=asset.id,
                cycle=1,
                realized_rul=3,
                labeled_at=NOW,
            )
        )


def _new_prediction(asset_id: uuid.UUID, reading_id: uuid.UUID) -> object:
    from turbine_guard.database.commands import NewPrediction

    return NewPrediction(
        asset_id=asset_id,
        sensor_reading_id=reading_id,
        cycle=1,
        predicted_rul=4.0,
        risk_level=RiskLevel.CRITICAL,
        model_name="fake-rul",
        model_version="1",
        feature_version="1",
        prediction_timestamp=NOW,
    )


REPO_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_REAL_ARTIFACTS = (
    REPO_DATA_DIR / "features" / "cmapss" / "FD001" / "split_manifest.json",
    REPO_DATA_DIR / "mlflow" / "mlflow.db",
)


@pytest.mark.real_data
@pytest.mark.skipif(
    not all(path.exists() for path in _REAL_ARTIFACTS),
    reason="FD001 artifacts or MLflow registry unavailable",
)
def test_real_fd001_replay_full_lifecycle_with_registered_champion(
    db_session: Session, postgres_engine: Engine
) -> None:
    """Replay one real held-out engine end to end with the registered champion."""
    from turbine_guard.serving.model_loader import ChampionModelLoader

    settings = _settings(postgres_engine).model_copy(
        update={
            "data_dir": REPO_DATA_DIR,
            "mlflow_tracking_uri": f"sqlite:///{REPO_DATA_DIR / 'mlflow' / 'mlflow.db'}",
        }
    )
    loader = ChampionModelLoader(settings)
    clock = SettableClock()
    with _client(db_session, settings, loader) as http:
        engine, store = _orchestrator(REPO_DATA_DIR, db_session, http, clock)
        source_asset = engine.replay_asset_ids()[0]
        run = engine.start(source_asset, mode=ReplayMode.ACCELERATED)
        final = engine.drive(run.run_id)

        assert final.status is ReplayRunStatus.COMPLETED
        assert final.last_confirmed_cycle == final.final_cycle
        asset_uuid = store.resolve_asset(run.external_asset_id)
        assert asset_uuid is not None
        outcomes = db_session.scalars(
            select(PredictionOutcome).where(PredictionOutcome.asset_id == asset_uuid)
        ).all()
        assert len(outcomes) == final.final_cycle
        assert min(outcome.realized_rul for outcome in outcomes) == 0

        evaluations = store.evaluations_for_run(run.run_id)
        assert len(evaluations) == 1
        evaluation = evaluations[0]
        assert evaluation.sample_count == final.final_cycle
        assert evaluation.mae is not None
        assert evaluation.mae >= 0
        assert evaluation.interval_coverage is not None
        assert 0.0 <= evaluation.interval_coverage <= 1.0
        assert evaluation.metrics["critical"]["missed_failures"] in (0, 1)

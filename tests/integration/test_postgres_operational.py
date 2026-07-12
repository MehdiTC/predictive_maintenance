"""PostgreSQL-only migration, constraint, repository, and transaction coverage."""

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from sqlalchemy import Engine, inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from turbine_guard.database.commands import (
    NewAsset,
    NewDriftReport,
    NewMaintenanceEvent,
    NewModelEvaluation,
    NewPipelineRun,
    NewPrediction,
    NewSensorReading,
)
from turbine_guard.database.enums import (
    AssetStatus,
    DriftStatus,
    EvaluationScope,
    MaintenanceEventType,
    PipelineRunStatus,
    PipelineRunType,
    RiskLevel,
)
from turbine_guard.database.errors import PredictionConflictError, SensorReadingConflictError
from turbine_guard.database.models import Asset, SensorReading
from turbine_guard.database.repositories import (
    AssetRepository,
    DriftReportRepository,
    MaintenanceEventRepository,
    ModelEvaluationRepository,
    PipelineRunRepository,
    PredictionRepository,
    SensorReadingRepository,
)
from turbine_guard.database.session import check_database_connection, session_scope

pytestmark = pytest.mark.postgres
NOW = datetime(2026, 7, 12, 12, tzinfo=UTC)


def reading(asset_id: uuid.UUID, cycle: int, *, first_sensor: float = 1.0) -> NewSensorReading:
    return NewSensorReading(
        asset_id=asset_id,
        cycle=cycle,
        observed_at=NOW + timedelta(minutes=cycle),
        operating_settings=(1.0, 2.0, 3.0),
        sensor_values=(first_sensor, *(float(value) for value in range(2, 22))),
        schema_version="1",
        source="integration-test",
    )


def prediction(
    asset_id: uuid.UUID, reading_id: uuid.UUID, cycle: int, **changes: object
) -> NewPrediction:
    values: dict[str, object] = {
        "asset_id": asset_id,
        "sensor_reading_id": reading_id,
        "cycle": cycle,
        "predicted_rul": 20.0,
        "lower_rul": 15.0,
        "upper_rul": 25.0,
        "risk_level": RiskLevel.CRITICAL,
        "failure_within_30": True,
        "failure_within_50": True,
        "model_name": "TurbineGuard-FD001-RUL",
        "model_version": "1",
        "model_alias": "champion",
        "model_run_id": "mlflow-run",
        "feature_version": "1",
        "prediction_timestamp": NOW,
        "latency_ms": 2.5,
    }
    values.update(changes)
    return NewPrediction(**values)  # type: ignore[arg-type]


def test_migration_is_current_and_schema_has_expected_constraints(postgres_engine: Engine) -> None:
    expected_tables = {
        "alembic_version",
        "assets",
        "sensor_readings",
        "predictions",
        "maintenance_events",
        "model_evaluations",
        "drift_reports",
        "pipeline_runs",
    }
    inspector = inspect(postgres_engine)
    assert check_database_connection(postgres_engine) is True
    assert expected_tables <= set(inspector.get_table_names())
    unique_names = {item["name"] for item in inspector.get_unique_constraints("sensor_readings")}
    index_names = {item["name"] for item in inspector.get_indexes("predictions")}
    assert "uq_sensor_readings_asset_cycle" in unique_names
    assert "ix_predictions_asset_timestamp" in index_names
    with postgres_engine.connect() as connection:
        assert MigrationContext.configure(connection).get_current_revision() == "20260712_0001"
    command.check(Config("alembic.ini"))


def test_asset_sensor_and_prediction_idempotency_and_queries(db_session: Session) -> None:
    assets = AssetRepository(db_session)
    sensors = SensorReadingRepository(db_session)
    predictions = PredictionRepository(db_session)
    asset = assets.create(
        NewAsset(
            external_id="cmapss-fd001-1",
            dataset_name="NASA C-MAPSS",
            dataset_subset="FD001",
            source_asset_id="1",
        )
    )
    assert assets.get(asset.id) is asset
    assert assets.get_by_external_id(asset.external_id) is asset
    assert assets.list() == [asset]
    assets.update_status(asset, AssetStatus.MAINTENANCE)
    assert asset.status is AssetStatus.MAINTENANCE

    inserted = sensors.insert(reading(asset.id, 1))
    assert sensors.insert(reading(asset.id, 1)).id == inserted.id
    with pytest.raises(SensorReadingConflictError):
        sensors.insert(reading(asset.id, 1, first_sensor=999.0))
    batch = sensors.insert_many([reading(asset.id, 2), reading(asset.id, 3)])
    assert [item.cycle for item in batch] == [2, 3]
    assert [item.cycle for item in sensors.history_through(asset.id, 2)] == [1, 2]
    assert sensors.latest(asset.id).cycle == 3  # type: ignore[union-attr]

    created = predictions.create(prediction(asset.id, inserted.id, 1))
    assert created.model_run_id == "mlflow-run"
    assert predictions.create(prediction(asset.id, inserted.id, 1)).id == created.id
    with pytest.raises(PredictionConflictError):
        predictions.create(prediction(asset.id, inserted.id, 1, predicted_rul=19.0))
    assert predictions.latest(asset.id).id == created.id  # type: ignore[union-attr]
    assert predictions.for_asset(asset.id) == [created]
    assert predictions.recent(limit=1) == [created]


def test_database_constraints_and_batch_savepoint_prevent_partial_data(db_session: Session) -> None:
    asset = AssetRepository(db_session).create(NewAsset(external_id="asset-constraints"))
    sensors = SensorReadingRepository(db_session)
    sensors.insert(reading(asset.id, 2))
    with pytest.raises(SensorReadingConflictError):
        sensors.insert_many([reading(asset.id, 1), reading(asset.id, 2, first_sensor=99.0)])
    assert sensors.get_by_asset_cycle(asset.id, 1) is None

    invalid = SensorReading(
        **{
            **{
                "asset_id": asset.id,
                "cycle": 0,
                "observed_at": NOW,
                "schema_version": "1",
                "source": "raw-orm-test",
            },
            **{f"operating_setting_{index}": 1.0 for index in range(1, 4)},
            **{f"sensor_{index:02d}": 1.0 for index in range(1, 22)},
        }
    )

    def persist_invalid() -> None:
        with db_session.begin_nested():
            db_session.add(invalid)
            db_session.flush()

    with pytest.raises(IntegrityError):
        persist_invalid()


def test_maintenance_supporting_tables_and_jsonb(db_session: Session) -> None:
    asset = AssetRepository(db_session).create(NewAsset(external_id="asset-events"))
    events = MaintenanceEventRepository(db_session)
    event = events.create(
        NewMaintenanceEvent(
            asset_id=asset.id,
            event_type=MaintenanceEventType.FAILURE,
            occurred_at=NOW - timedelta(days=1),
            event_cycle=100,
            source="delayed-replay",
            external_event_id="failure-1",
            metadata={"cause": "simulated_end_of_life"},
        )
    )
    assert event.event_metadata["cause"] == "simulated_end_of_life"
    assert (
        events.create(
            NewMaintenanceEvent(
                asset_id=asset.id,
                event_type=MaintenanceEventType.FAILURE,
                occurred_at=NOW - timedelta(days=1),
                event_cycle=100,
                source="delayed-replay",
                external_event_id="failure-1",
                metadata={"cause": "simulated_end_of_life"},
            )
        ).id
        == event.id
    )
    assert events.for_asset(asset.id) == [event]
    assert events.latest_failure_or_maintenance(asset.id) is event

    evaluation = ModelEvaluationRepository(db_session).create(
        NewModelEvaluation(
            model_name="model",
            model_version="1",
            evaluation_scope=EvaluationScope.REPLAY,
            window_start=NOW - timedelta(days=1),
            window_end=NOW,
            sample_count=10,
            mae=2.0,
            metrics={"secondary": 1.0},
        )
    )
    assert evaluation.metrics == {"secondary": 1.0}
    drift = DriftReportRepository(db_session).create(
        NewDriftReport(
            model_name="model",
            model_version="1",
            feature_version="1",
            window_start=NOW - timedelta(days=1),
            window_end=NOW,
            status=DriftStatus.NOT_DETECTED,
            drifted_feature_count=0,
            details={"features": {}},
        )
    )
    assert drift.details == {"features": {}}
    runs = PipelineRunRepository(db_session)
    run = runs.create(
        NewPipelineRun(
            run_type=PipelineRunType.INGESTION,
            status=PipelineRunStatus.FAILED,
            started_at=NOW - timedelta(minutes=1),
            finished_at=NOW,
            trigger="manual",
            error_message="source unavailable",
            metadata={"retryable": True},
        )
    )
    assert run.error_message == "source unavailable"
    running = runs.create(
        NewPipelineRun(
            run_type=PipelineRunType.MONITORING,
            status=PipelineRunStatus.RUNNING,
            started_at=NOW - timedelta(minutes=1),
            trigger="schedule",
        )
    )
    runs.finish(
        running,
        status=PipelineRunStatus.SUCCEEDED,
        finished_at=NOW,
        output_manifest_checksum="a" * 64,
    )
    assert runs.get(running.id).status is PipelineRunStatus.SUCCEEDED  # type: ignore[union-attr]
    assert runs.recent()[0] in {run, running}


def test_repository_operations_join_external_transaction(
    db_session: Session, postgres_engine: Engine
) -> None:
    AssetRepository(db_session).create(NewAsset(external_id="rolled-back"))
    with Session(postgres_engine) as independent_session:
        assert (
            independent_session.scalar(select(Asset).where(Asset.external_id == "rolled-back"))
            is None
        )


def test_session_scope_commits_and_rolls_back(postgres_engine: Engine) -> None:
    factory = sessionmaker(bind=postgres_engine, class_=Session, expire_on_commit=False)
    committed_external_id = f"committed-{uuid.uuid4()}"
    rolled_back_external_id = f"rolled-back-{uuid.uuid4()}"
    try:
        with session_scope(factory) as session:
            AssetRepository(session).create(NewAsset(external_id=committed_external_id))
        with factory() as session:
            assert (
                session.scalar(select(Asset).where(Asset.external_id == committed_external_id))
                is not None
            )

        def force_rollback() -> None:
            with session_scope(factory) as session:
                AssetRepository(session).create(NewAsset(external_id=rolled_back_external_id))
                raise RuntimeError("force rollback")

        with pytest.raises(RuntimeError, match="force rollback"):
            force_rollback()
        with factory() as session:
            assert (
                session.scalar(select(Asset).where(Asset.external_id == rolled_back_external_id))
                is None
            )
    finally:
        with factory.begin() as session:
            committed = session.scalar(
                select(Asset).where(Asset.external_id == committed_external_id)
            )
            if committed is not None:
                session.delete(committed)

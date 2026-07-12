"""Fast validation tests for persistence commands and configuration."""

import math
import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import Engine

from turbine_guard.database.commands import NewPipelineRun, NewPrediction, NewSensorReading
from turbine_guard.database.enums import PipelineRunStatus, PipelineRunType, RiskLevel
from turbine_guard.database.session import (
    DatabaseConfig,
    check_database_connection,
    create_database_engine,
)

NOW = datetime(2026, 7, 12, tzinfo=UTC)


def sensor_command(*, cycle: int = 1, first_sensor: float = 1.0) -> NewSensorReading:
    sensors = (first_sensor, *(float(index) for index in range(2, 22)))
    return NewSensorReading(
        asset_id=uuid.uuid4(),
        cycle=cycle,
        observed_at=NOW,
        operating_settings=(1.0, 2.0, 3.0),
        sensor_values=sensors,
        schema_version="1",
        source="unit-test",
    )


def test_sensor_command_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        sensor_command(first_sensor=math.nan)


def test_sensor_command_rejects_non_positive_cycle() -> None:
    with pytest.raises(ValueError, match="positive"):
        sensor_command(cycle=0)


def test_prediction_validates_interval_and_utc_timestamp() -> None:
    reading_id = uuid.uuid4()
    asset_id = uuid.uuid4()
    with pytest.raises(ValueError, match="interval"):
        NewPrediction(
            asset_id=asset_id,
            sensor_reading_id=reading_id,
            cycle=1,
            predicted_rul=10,
            lower_rul=11,
            upper_rul=12,
            risk_level=RiskLevel.CRITICAL,
            model_name="model",
            model_version="1",
            feature_version="1",
            prediction_timestamp=NOW,
        )


def test_failed_pipeline_run_requires_error_and_finish_time() -> None:
    with pytest.raises(ValueError, match="finished_at"):
        NewPipelineRun(
            run_type=PipelineRunType.INGESTION,
            status=PipelineRunStatus.FAILED,
            started_at=NOW,
            trigger="manual",
            error_message="failed",
        )


def test_engine_creation_is_lazy() -> None:
    config = DatabaseConfig(url="postgresql+psycopg://user:password@localhost:5432/database")
    engine = create_database_engine(config)
    assert isinstance(engine, Engine)
    assert engine.pool.checkedout() == 0
    engine.dispose()


def test_connection_check_success_and_error() -> None:
    class Connection:
        def __enter__(self) -> "Connection":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def execute(self, statement: object) -> None:
            del statement

    class AvailableEngine:
        def connect(self) -> Connection:
            return Connection()

    class UnavailableEngine:
        def connect(self) -> Connection:
            raise TimeoutError

    assert check_database_connection(AvailableEngine()) is True  # type: ignore[arg-type]
    assert check_database_connection(UnavailableEngine()) is False  # type: ignore[arg-type]

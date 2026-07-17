"""Dashboard projections over real migrated PostgreSQL state."""

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest
from sqlalchemy.orm import Session, sessionmaker

from turbine_guard.config.settings import Environment, Settings
from turbine_guard.database.commands import (
    NewAsset,
    NewDriftReport,
    NewModelEvaluation,
    NewPrediction,
    NewSensorReading,
)
from turbine_guard.database.enums import DriftStatus, EvaluationScope, RiskLevel
from turbine_guard.database.repositories import (
    AssetRepository,
    DriftReportRepository,
    ModelEvaluationRepository,
    PredictionRepository,
    SensorReadingRepository,
)
from turbine_guard.services.dashboard import DashboardService
from turbine_guard.serving.model_loader import ChampionModelLoader

pytestmark = pytest.mark.postgres
NOW = datetime.now(UTC).replace(microsecond=0)


class UnavailableLoader:
    def get(self) -> Any:
        raise RuntimeError("MLflow unavailable in focused read-side integration test")


def _reading(asset_id: uuid.UUID, cycle: int) -> NewSensorReading:
    return NewSensorReading(
        asset_id=asset_id,
        cycle=cycle,
        observed_at=NOW - timedelta(seconds=cycle),
        operating_settings=(1.0, 2.0, 3.0),
        sensor_values=tuple(float(index + cycle) for index in range(1, 22)),  # type: ignore[arg-type]
        schema_version="1",
        source="dashboard-integration",
    )


def _prediction(
    asset_id: uuid.UUID,
    reading_id: uuid.UUID,
    cycle: int,
    risk: RiskLevel,
    point: float,
) -> NewPrediction:
    return NewPrediction(
        asset_id=asset_id,
        sensor_reading_id=reading_id,
        cycle=cycle,
        predicted_rul=point,
        lower_rul=max(0, point - 5),
        upper_rul=point + 5,
        risk_level=risk,
        failure_within_30=point <= 30,
        failure_within_50=point <= 50,
        model_name="TurbineGuard-FD001-RUL",
        model_version="7",
        model_alias="champion",
        model_run_id="run-7",
        feature_version="1",
        prediction_timestamp=NOW + timedelta(seconds=cycle),
        latency_ms=1.5 + cycle,
    )


def test_dashboard_reads_real_postgres_state_without_raw_models(db_session: Session) -> None:
    assets = AssetRepository(db_session)
    sensors = SensorReadingRepository(db_session)
    predictions = PredictionRepository(db_session)
    critical = assets.create(NewAsset(external_id="dashboard-critical"))
    healthy = assets.create(NewAsset(external_id="dashboard-healthy"))
    critical_readings = [sensors.insert(_reading(critical.id, cycle)) for cycle in (1, 2)]
    healthy_reading = sensors.insert(_reading(healthy.id, 1))
    predictions.create(_prediction(critical.id, critical_readings[0].id, 1, RiskLevel.WARNING, 45))
    predictions.create(_prediction(critical.id, critical_readings[1].id, 2, RiskLevel.CRITICAL, 20))
    predictions.create(_prediction(healthy.id, healthy_reading.id, 1, RiskLevel.HEALTHY, 80))
    DriftReportRepository(db_session).create(
        NewDriftReport(
            model_name="TurbineGuard-FD001-RUL",
            model_version="7",
            feature_version="1",
            window_start=NOW - timedelta(days=1),
            window_end=NOW,
            status=DriftStatus.WARNING,
            max_psi=0.2,
            max_wasserstein=0.3,
            drifted_feature_count=1,
            details={
                "features": [
                    {
                        "feature": "sensor_04_current",
                        "psi": 0.2,
                        "wasserstein": 0.3,
                        "normalized_wasserstein": 0.4,
                        "missingness_shift": 0,
                        "drifted": True,
                        "warning": True,
                    }
                ]
            },
        )
    )
    ModelEvaluationRepository(db_session).create(
        NewModelEvaluation(
            model_name="TurbineGuard-FD001-RUL",
            model_version="7",
            evaluation_scope=EvaluationScope.ONLINE,
            dataset_subset="FD001",
            window_start=NOW - timedelta(days=1),
            window_end=NOW,
            sample_count=2,
            mae=4,
            rmse=5,
            nasa_score=10,
            critical_precision=0.8,
            critical_recall=0.9,
            interval_coverage=1,
            metrics={
                "status": "available",
                "asset_count": 1,
                "critical": {
                    "f1": 0.85,
                    "false_alarms_per_1000_cycles": 0,
                    "mean_first_alert_lead_time": 20,
                    "timely_warning_asset_percentage": 100,
                },
                "interval": {"average_width": 10},
            },
        )
    )
    db_session.flush()
    sessions = sessionmaker(
        bind=db_session.connection(), class_=Session, expire_on_commit=False, autoflush=False
    )
    service = DashboardService(
        sessions,
        cast(ChampionModelLoader, UnavailableLoader()),
        Settings(environment=Environment.TESTING, online_inference_enabled=False),
    )

    fleet = service.fleet(limit=10, offset=0)
    assert [item.external_asset_id for item in fleet.items] == [
        "dashboard-critical",
        "dashboard-healthy",
    ]
    assert fleet.critical_count == 1
    assert fleet.healthy_count == 1
    assert fleet.items[0].predicted_rul == 20
    assert fleet.items[0].lower_rul == 15
    assert fleet.items[0].model_version == "7"

    detail = service.asset(critical.id, sensor_columns=("sensor_02", "sensor_04"), limit=10)
    assert [item.cycle for item in detail.predictions] == [1, 2]
    assert detail.sensor_history[-1].values == {"sensor_02": 4.0, "sensor_04": 6.0}
    assert detail.failure_within_30 is True

    alerts = service.alerts(limit=10)
    assert alerts.items[0].first_warning_cycle == 1
    assert alerts.items[0].first_critical_cycle == 2
    assert alerts.items[0].current_risk_level == "critical"
    assert service.drift().top_features[0].feature == "sensor_04_current"
    performance = service.performance()
    assert performance.target_label.startswith("Uncapped realized RUL")
    assert performance.critical_f1 == 0.85

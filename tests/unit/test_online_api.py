"""Versioned route, error, request-ID, and metrics tests with an injected service."""

import uuid
from datetime import UTC, datetime
from typing import cast

from fastapi.testclient import TestClient

from turbine_guard.api.app import create_app
from turbine_guard.api.schemas.online import (
    AssetDetailResponse,
    AssetHealthResponse,
    AssetListResponse,
    AssetSummaryResponse,
    CurrentModelResponse,
    MonitoringSummaryResponse,
    PredictionResponse,
    RecentPredictionsResponse,
    SensorIngestionResponse,
)
from turbine_guard.config.settings import Environment, Settings
from turbine_guard.database.errors import SensorReadingConflictError
from turbine_guard.observability.metrics import OnlineMetrics
from turbine_guard.services.inference import OnlineInferenceService, SensorObservation

NOW = datetime(2026, 7, 12, 12, tzinfo=UTC)
ASSET_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")
READING_ID = uuid.UUID("00000000-0000-0000-0000-000000000002")


def payload() -> dict[str, object]:
    return {
        "external_asset_id": "asset-1",
        "cycle": 1,
        "observed_at": NOW.isoformat(),
        "operating_setting_1": 1.0,
        "operating_setting_2": 2.0,
        "operating_setting_3": 3.0,
        **{f"sensor_{index:02d}": float(index) for index in range(1, 22)},
        "source": "unit-test",
        "schema_version": "1",
    }


def prediction() -> PredictionResponse:
    return PredictionResponse(
        predicted_rul=40,
        lower_rul=30,
        upper_rul=50,
        risk_level="warning",
        failure_within_30=False,
        failure_within_50=True,
        model_name="model",
        model_version="1",
        model_alias="champion",
        model_run_id="run",
        feature_version="1",
        prediction_timestamp=NOW,
        latency_ms=2.0,
    )


class FakeService:
    conflict = False

    def ingest(self, observation: SensorObservation, request_id: str) -> SensorIngestionResponse:
        if self.conflict:
            raise SensorReadingConflictError("Cycle already has different data.")
        return SensorIngestionResponse(
            request_id=request_id,
            asset_id=ASSET_ID,
            external_asset_id=observation.external_asset_id,
            cycle=observation.cycle,
            reading_id=READING_ID,
            prediction=prediction(),
            idempotent=False,
            reading_idempotent=False,
            prediction_idempotent=False,
        )

    def list_assets(self, *, limit: int, offset: int) -> AssetListResponse:
        return AssetListResponse(
            items=[
                AssetSummaryResponse(
                    asset_id=ASSET_ID,
                    external_asset_id="asset-1",
                    status="active",
                    latest_cycle=1,
                    latest_risk_level="warning",
                    latest_predicted_rul=40,
                    last_observed_at=NOW,
                )
            ],
            limit=limit,
            offset=offset,
        )

    def get_asset(self, asset_id: uuid.UUID) -> AssetDetailResponse:
        return AssetDetailResponse(
            asset_id=asset_id,
            external_asset_id="asset-1",
            dataset_name=None,
            dataset_subset=None,
            source_asset_id=None,
            status="active",
            created_at=NOW,
            updated_at=NOW,
            latest_reading=None,
            latest_prediction=prediction(),
            recent_maintenance_events=[],
        )

    def get_asset_health(self, asset_id: uuid.UUID) -> AssetHealthResponse:
        return AssetHealthResponse(
            asset_id=asset_id,
            external_asset_id="asset-1",
            latest_cycle=1,
            predicted_rul=40,
            lower_rul=30,
            upper_rul=50,
            risk_level="warning",
            failure_within_30=False,
            failure_within_50=True,
            prediction_trend=[],
            latest_observation_at=NOW,
            model_version="1",
            stale=False,
            data_quality_status="valid",
        )

    def recent_predictions(
        self, *, limit: int, asset_id: uuid.UUID | None = None
    ) -> RecentPredictionsResponse:
        del asset_id
        return RecentPredictionsResponse(items=[], limit=limit)

    def current_model(self) -> CurrentModelResponse:
        return CurrentModelResponse(
            model_name="model",
            registry_version="1",
            alias="champion",
            source_run_id="run",
            target_definition="capped_125",
            rul_cap=125,
            feature_count=552,
            feature_version="1",
            validation_rmse=1,
            replay_rmse=2,
            official_test_rmse=3,
            conformal_coverage_target=0.9,
            model_load_timestamp=NOW,
            model_checksum="abc",
            lineage_id="lineage",
        )

    def monitoring_summary(self) -> MonitoringSummaryResponse:
        return MonitoringSummaryResponse(
            request_count=1,
            prediction_count=1,
            validation_failures=0,
            database_failures=0,
            model_load_failures=0,
            prediction_failures=0,
            conflict_count=0,
            average_prediction_latency_ms=2,
            current_model_version="1",
            recent_risk_distribution={"warning": 1},
            reading_count=1,
            stored_prediction_count=1,
            latest_ingestion_time=NOW,
        )


def _client(service: FakeService | None = None) -> TestClient:
    settings = Settings(
        environment=Environment.TESTING,
        log_level="WARNING",
        online_inference_enabled=False,
    )
    return TestClient(
        create_app(
            settings,
            online_service=cast(OnlineInferenceService, service or FakeService()),
        )
    )


def test_sensor_ingestion_contract_request_id_and_openapi() -> None:
    with _client() as client:
        response = client.post(
            "/v1/sensor-readings", json=payload(), headers={"X-Request-ID": "request-123"}
        )
        assert response.status_code == 201
        assert response.headers["X-Request-ID"] == "request-123"
        assert response.json()["prediction"]["risk_level"] == "warning"
        assert response.json()["prediction"]["prediction_timestamp"].endswith("Z")
        paths = client.get("/openapi.json").json()["paths"]
        assert "/v1/sensor-readings" in paths
        assert "/v1/predictions" not in paths


def test_validation_and_conflict_errors_are_structured() -> None:
    service = FakeService()
    with _client(service) as client:
        invalid = payload()
        invalid["cycle"] = 0
        response = client.post("/v1/sensor-readings", json=invalid)
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "request_validation_failed"
        assert "request_id" in response.json()["error"]
        service.conflict = True
        conflict = client.post("/v1/sensor-readings", json=payload())
        assert conflict.status_code == 409
        assert conflict.json()["error"]["code"] == "sensor_reading_conflict"
        assert "different data" in conflict.text


def test_asset_prediction_model_monitoring_and_metrics_endpoints() -> None:
    with _client() as client:
        assert client.get("/v1/assets?limit=1&offset=0").status_code == 200
        assert client.get(f"/v1/assets/{ASSET_ID}").status_code == 200
        assert client.get(f"/v1/assets/{ASSET_ID}/health").status_code == 200
        assert client.get("/v1/predictions/recent").status_code == 200
        assert client.get("/v1/models/current").json()["registry_version"] == "1"
        assert client.get("/v1/monitoring/summary").status_code == 200
        metrics = client.get("/metrics")
        assert metrics.status_code == 200
        assert "turbine_guard_http_requests_total" in metrics.text
        assert "external_asset_id" not in metrics.text
        assert "request_id" not in metrics.text


def test_repeated_app_creation_has_isolated_metric_registries() -> None:
    first = create_app(
        Settings(environment=Environment.TESTING, online_inference_enabled=False),
        metrics=OnlineMetrics(),
    )
    second = create_app(
        Settings(environment=Environment.TESTING, online_inference_enabled=False),
        metrics=OnlineMetrics(),
    )
    assert first.state.metrics.registry is not second.state.metrics.registry


def test_readiness_reports_individual_online_dependency_failures() -> None:
    settings = Settings(environment=Environment.TESTING, online_inference_enabled=False)
    app = create_app(
        settings,
        online_service=cast(OnlineInferenceService, FakeService()),
        readiness_checks={
            "database": lambda: True,
            "model": lambda: False,
            "feature_contract": lambda: True,
        },
    )
    with TestClient(app) as client:
        response = client.get("/health/ready")
        assert response.status_code == 503
        assert response.json()["checks"] == {
            "database": True,
            "model": False,
            "feature_contract": True,
        }

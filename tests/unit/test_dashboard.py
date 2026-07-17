"""Dashboard rendering, data-contract, degraded-state, and security tests."""

import asyncio
import hmac
import uuid
from datetime import UTC, datetime
from hashlib import sha256
from types import SimpleNamespace
from typing import cast

from fastapi.testclient import TestClient

from turbine_guard.api.app import create_app
from turbine_guard.api.schemas.dashboard import (
    AlertAssetItem,
    AlertSummaryResponse,
    AssetDashboardResponse,
    DemoPredictionPoint,
    DemoStateResponse,
    DriftDetailResponse,
    FleetAssetItem,
    FleetOverviewResponse,
    ModelOverviewResponse,
    PerformanceDetailResponse,
    PredictionHistoryItem,
    PredictionHistoryResponse,
    ReplayStatusResponse,
)
from turbine_guard.config.settings import Environment, Settings
from turbine_guard.services.dashboard import DashboardService
from turbine_guard.services.errors import AssetNotFoundError, ReplayControlDisabledError
from turbine_guard.services.replay_control import ReplayControlService

NOW = datetime(2026, 7, 13, 12, tzinfo=UTC)
ASSET_ID = uuid.UUID("00000000-0000-0000-0000-000000000011")
PREDICTION_ID = uuid.UUID("00000000-0000-0000-0000-000000000012")


def _prediction() -> PredictionHistoryItem:
    return PredictionHistoryItem(
        prediction_id=PREDICTION_ID,
        asset_id=ASSET_ID,
        external_asset_id="demo-asset-9",
        cycle=42,
        predicted_rul=24.5,
        lower_rul=15.0,
        upper_rul=34.0,
        risk_level="critical",
        model_name="TurbineGuard-FD001-RUL",
        model_version="7",
        feature_version="1",
        prediction_timestamp=NOW,
        latency_ms=2.25,
    )


class FakeDashboard:
    empty = False
    fail = False

    def fleet(self, *, limit: int, offset: int) -> FleetOverviewResponse:
        if self.fail:
            raise RuntimeError("postgresql+psycopg://secret@internal/db")
        items = (
            []
            if self.empty
            else [
                FleetAssetItem(
                    asset_id=ASSET_ID,
                    external_asset_id="demo-asset-9",
                    asset_status="active",
                    latest_cycle=42,
                    predicted_rul=24.5,
                    lower_rul=15,
                    upper_rul=34,
                    risk_level="critical",
                    latest_observation_at=NOW,
                    prediction_timestamp=NOW,
                    model_version="7",
                    feature_version="1",
                    stale=False,
                )
            ]
        )
        return FleetOverviewResponse(
            total_assets=len(items),
            active_assets=len(items),
            latest_observation_at=NOW if items else None,
            healthy_count=0,
            warning_count=0,
            critical_count=len(items),
            assets_without_recent_predictions=0,
            current_model_version="7",
            drift_status="not_detected",
            performance_status="available",
            replay_status="running",
            items=items,
            limit=limit,
            offset=offset,
        )

    def alerts(self, *, limit: int) -> AlertSummaryResponse:
        return AlertSummaryResponse(
            warning_count=0,
            critical_count=0 if self.empty else 1,
            items=[]
            if self.empty
            else [
                AlertAssetItem(
                    asset_id=ASSET_ID,
                    external_asset_id="demo-asset-9",
                    current_risk_level="critical",
                    first_warning_cycle=30,
                    first_critical_cycle=40,
                    latest_predicted_rul=24.5,
                    latest_prediction_at=NOW,
                    alert_age_seconds=10,
                    model_version="7",
                    outcome=None,
                )
            ],
            limit=limit,
        )

    def prediction_history(self, **_: object) -> PredictionHistoryResponse:
        return PredictionHistoryResponse(
            items=[] if self.empty else [_prediction()], limit=50, offset=0
        )

    def asset(
        self, asset_id: uuid.UUID, *, sensor_columns: tuple[str, ...], limit: int
    ) -> AssetDashboardResponse:
        del limit
        if asset_id != ASSET_ID:
            raise AssetNotFoundError("missing")
        return AssetDashboardResponse(
            asset_id=asset_id,
            external_asset_id="demo-asset-9",
            asset_status="active",
            dataset_name="NASA C-MAPSS",
            dataset_subset="FD001",
            source_asset_id="9",
            latest_cycle=42,
            predicted_rul=24.5,
            lower_rul=15,
            upper_rul=34,
            risk_level="critical",
            failure_within_30=True,
            failure_within_50=True,
            model_version="7",
            feature_version="1",
            latest_observation_at=NOW,
            stale=False,
            data_quality_warnings=[],
            predictions=[_prediction()],
            sensor_columns=list(sensor_columns),
            available_sensor_columns=[f"sensor_{index:02d}" for index in range(1, 22)],
            sensor_history=[],
            maintenance_events=[],
            replay=None,
        )

    def drift(self) -> DriftDetailResponse:
        return DriftDetailResponse(
            status="not_detected",
            available=True,
            model_name="TurbineGuard-FD001-RUL",
            model_version="7",
            feature_version="1",
            window_start=NOW,
            window_end=NOW,
            drifted_feature_count=0,
            max_psi=0.04,
            max_wasserstein=0.2,
            report_timestamp=NOW,
            top_features=[],
            note="Drift does not prove model error.",
        )

    def performance(self) -> PerformanceDetailResponse:
        return PerformanceDetailResponse(
            status="available",
            available=True,
            target_label="Uncapped realized RUL (cycles)",
            model_name="TurbineGuard-FD001-RUL",
            model_version="7",
            window_start=NOW,
            window_end=NOW,
            labeled_rows=42,
            completed_assets=1,
            mae=8,
            rmse=10,
            nasa_score=100,
            critical_precision=0.8,
            critical_recall=0.9,
            critical_f1=0.85,
            false_alarms_per_1000_cycles=12,
            mean_alert_lead_time=20,
            timely_alert_rate=100,
            interval_coverage=0.9,
            average_interval_width=30,
            report_timestamp=NOW,
        )

    def demo(self) -> DemoStateResponse:
        if self.fail:
            raise RuntimeError("postgresql+psycopg://secret@internal/db")
        return DemoStateResponse(
            enabled=True,
            demo_source_asset_id=9,
            run=None,
            series=[
                DemoPredictionPoint(
                    cycle=42,
                    predicted_rul=24.5,
                    lower_rul=15.0,
                    upper_rul=34.0,
                    risk_level="critical",
                )
            ],
            model_version="7",
            max_attempts=25,
            max_cycles_per_request=20,
            cooldown_seconds=1.0,
        )

    def model(self) -> ModelOverviewResponse:
        return ModelOverviewResponse(
            available=True,
            registry_source="exported_snapshot",
            registered_model_name="TurbineGuard-FD001-RUL",
            registry_version="7",
            alias="champion",
            aliases={"champion": "7", "candidate": "8"},
            model_family="ridge",
            target_definition="capped_125",
            rul_cap=125,
            feature_count=552,
            feature_version="1",
            validation_rmse=12,
            replay_rmse=13,
            official_benchmark_rmse=14,
            conformal_coverage_target=0.9,
            source_run_id="source-run",
            model_load_timestamp=NOW,
            git_sha="abc123",
            manifest_lineage={"lineage_id": "safe-lineage"},
            latest_lifecycle=[],
            latest_event=None,
        )


class FakeReplay:
    def status(self, *, limit: int) -> ReplayStatusResponse:
        del limit
        return ReplayStatusResponse(
            enabled=False,
            writable=False,
            public_demo_mode=True,
            allowed_source_asset_ids=[9],
            restrictions=["This deployment exposes replay status as read-only."],
            runs=[],
        )

    def perform(self, *_: object, **__: object) -> object:
        raise ReplayControlDisabledError("read-only")


def _client(dashboard: FakeDashboard | None = None) -> TestClient:
    settings = Settings(
        environment=Environment.PRODUCTION,
        online_inference_enabled=False,
        trusted_hosts=("testserver",),
        database_url="postgresql+psycopg://secret:secret@internal-db/demo",
        mlflow_tracking_uri="http://internal-mlflow:5000",
        replay_admin_token=None,
    )
    app = create_app(
        settings,
        dashboard_service=cast(DashboardService, dashboard or FakeDashboard()),
        replay_control=cast(ReplayControlService, FakeReplay()),
    )
    return TestClient(app)


def test_fleet_page_and_json_render_correct_values() -> None:
    with _client() as client:
        page = client.get("/dashboard")
        api = client.get("/v1/fleet")
    assert page.status_code == 200
    assert "demo-asset-9" in page.text
    assert "24.5" in page.text
    assert "15.0–34.0" in page.text  # noqa: RUF001 - matches rendered interval typography
    assert "critical" in page.text
    assert api.json()["items"][0]["model_version"] == "7"
    assert "independently developed" in page.text


def test_empty_fleet_has_intentional_empty_state() -> None:
    dashboard = FakeDashboard()
    dashboard.empty = True
    with _client(dashboard) as client:
        response = client.get("/dashboard")
    assert response.status_code == 200
    assert "No assets yet" in response.text


def test_asset_detail_chart_data_and_missing_asset() -> None:
    with _client() as client:
        found = client.get(f"/dashboard/assets/{ASSET_ID}")
        missing = client.get("/dashboard/assets/00000000-0000-0000-0000-000000000099")
    assert found.status_code == 200
    assert "RUL and calibrated interval" in found.text
    assert '"predicted_rul": 24.5' in found.text
    assert "No physical meanings assigned" in found.text
    assert missing.status_code == 404
    assert "Asset not found" in missing.text


def test_prediction_model_drift_and_performance_pages() -> None:
    with _client() as client:
        predictions = client.get("/dashboard/predictions")
        model = client.get("/dashboard/models")
        monitoring = client.get("/dashboard/monitoring")
    assert "2.25 ms" in predictions.text
    assert "ridge" in model.text
    assert "Official benchmark RMSE" in model.text
    assert "immutable exported champion snapshot" in model.text
    assert "Uncapped realized RUL" in monitoring.text
    assert "Drift does not prove model error" in monitoring.text


def test_secrets_and_internal_urls_never_reach_html() -> None:
    with _client() as client:
        body = client.get("/dashboard").text
    assert "secret:secret" not in body
    assert "internal-db" not in body
    assert "internal-mlflow" not in body


def test_degraded_page_hides_exception_and_production_traceback() -> None:
    dashboard = FakeDashboard()
    dashboard.fail = True
    with _client(dashboard) as client:
        response = client.get("/dashboard")
    assert response.status_code == 503
    assert "temporarily unavailable" in response.text
    assert "postgresql+psycopg" not in response.text
    assert "Traceback" not in response.text


def test_security_headers_and_trusted_host() -> None:
    with _client() as client:
        response = client.get("/dashboard")
        rejected = client.get("/dashboard", headers={"host": "attacker.example"})
    assert response.headers["x-frame-options"] == "DENY"
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
    assert response.headers["strict-transport-security"].startswith("max-age=")
    assert rejected.status_code == 400


def test_public_replay_action_is_blocked_in_read_only_mode() -> None:
    with _client() as client:
        response = client.post("/v1/replay/actions", json={"action": "start", "source_asset_id": 9})
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "replay_control_disabled"


def test_production_cors_allows_only_configured_origin() -> None:
    settings = Settings(
        environment=Environment.PRODUCTION,
        online_inference_enabled=False,
        trusted_hosts=("testserver",),
        cors_allowed_origins=("https://dashboard.example",),
    )
    app = create_app(
        settings,
        dashboard_service=cast(DashboardService, FakeDashboard()),
        replay_control=cast(ReplayControlService, FakeReplay()),
    )
    with TestClient(app) as client:
        allowed = client.options(
            "/v1/fleet",
            headers={
                "Origin": "https://dashboard.example",
                "Access-Control-Request-Method": "GET",
            },
        )
        denied = client.options(
            "/v1/fleet",
            headers={
                "Origin": "https://attacker.example",
                "Access-Control-Request-Method": "GET",
            },
        )
    assert allowed.headers["access-control-allow-origin"] == "https://dashboard.example"
    assert "access-control-allow-origin" not in denied.headers


def test_replay_form_action_runs_off_the_event_loop() -> None:
    """The control service blocks on HTTP calls back to this same server.

    If the form handler ran it on the event loop, the process would deadlock
    in production (observed as hung requests and a killed instance).
    """
    observed: dict[str, bool] = {}

    class LoopProbeReplay(FakeReplay):
        def perform(self, *_: object, **__: object) -> object:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                observed["on_event_loop"] = False
            else:
                observed["on_event_loop"] = True
            return SimpleNamespace(message="probe completed")

    secret = "dashboard-secret-for-tests"
    settings = Settings(
        environment=Environment.PRODUCTION,
        online_inference_enabled=False,
        trusted_hosts=("testserver",),
        application_secret=secret,
    )
    app = create_app(
        settings,
        dashboard_service=cast(DashboardService, FakeDashboard()),
        replay_control=cast(ReplayControlService, LoopProbeReplay()),
    )
    token = hmac.new(secret.encode(), b"dashboard-replay-v1", sha256).hexdigest()
    with TestClient(app) as client:
        response = client.post(
            "/dashboard/replay",
            data={"action": "start", "source_asset_id": "9", "csrf_token": token},
            follow_redirects=False,
        )
    assert response.status_code == 303
    assert "probe+completed" in response.headers["location"]
    assert observed == {"on_event_loop": False}


def test_demo_landing_page_and_json_state() -> None:
    with _client() as client:
        page = client.get("/")
        state = client.get("/v1/demo")
    assert page.status_code == 200
    assert "Predicting jet-engine failure" in page.text
    assert "Run live simulation" not in page.text  # button text is set by JS, not the template
    assert 'id="demo-run"' in page.text
    assert 'id="demo-chart"' in page.text
    assert "held out" in page.text or "locked away" in page.text
    assert state.status_code == 200
    body = state.json()
    assert body["demo_source_asset_id"] == 9
    assert body["series"][0]["cycle"] == 42
    assert body["series"][0]["risk_level"] == "critical"
    assert body["max_cycles_per_request"] >= 1


def test_demo_page_renders_degraded_without_data_access() -> None:
    dashboard = FakeDashboard()
    dashboard.fail = True
    with _client(dashboard) as client:
        page = client.get("/")
    assert page.status_code == 200
    assert 'id="demo-run"' in page.text
    assert "postgresql+psycopg" not in page.text

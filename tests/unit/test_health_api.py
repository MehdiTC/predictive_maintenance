"""API tests for the health endpoints."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from turbine_guard.config.settings import Settings
from turbine_guard.services.health import ReadinessResult


def test_liveness_returns_alive(client: TestClient) -> None:
    response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_readiness_returns_ready(client: TestClient) -> None:
    response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready", "checks": {}}


def test_readiness_reports_failed_checks_with_503(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "turbine_guard.api.routes.health.check_readiness",
        lambda _: ReadinessResult(checks={"database": False}),
    )

    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {"status": "not_ready", "checks": {"database": False}}


def test_openapi_schema_documents_health_endpoints(client: TestClient) -> None:
    response = client.get("/openapi.json")

    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/health/live" in paths
    assert "/health/ready" in paths


def test_docs_page_is_served(client: TestClient) -> None:
    response = client.get("/docs")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_app_exposes_injected_settings(app: FastAPI, app_settings: Settings) -> None:
    assert app.state.settings is app_settings

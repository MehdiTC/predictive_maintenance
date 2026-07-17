"""Versioned, bounded JSON data source for the server-rendered dashboard."""

import uuid
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Header, Query, Request

from turbine_guard.api.schemas.dashboard import (
    AlertSummaryResponse,
    AssetDashboardResponse,
    DemoStateResponse,
    DriftDetailResponse,
    FleetOverviewResponse,
    ModelOverviewResponse,
    PerformanceDetailResponse,
    PredictionHistoryResponse,
    ReplayActionRequest,
    ReplayActionResponse,
    ReplayStatusResponse,
)
from turbine_guard.services.dashboard import DashboardService
from turbine_guard.services.errors import RequestParameterError
from turbine_guard.services.replay_control import ReplayControlService

router = APIRouter(prefix="/v1")


def _dashboard(request: Request) -> DashboardService:
    service: DashboardService | None = request.app.state.dashboard_service
    if service is None:
        raise RuntimeError("Dashboard data access is disabled.")
    return service


def _replay(request: Request) -> ReplayControlService:
    service: ReplayControlService | None = request.app.state.replay_control
    if service is None:
        raise RuntimeError("Replay status is disabled.")
    return service


def _bounded_limit(request: Request, value: int | None) -> int:
    settings = request.app.state.settings
    selected = settings.api_default_page_size if value is None else value
    if selected > settings.api_max_page_size:
        raise RequestParameterError("Requested limit exceeds the configured maximum.")
    return int(selected)


@router.get("/demo", response_model=DemoStateResponse, tags=["dashboard"])
def demo_state(request: Request) -> DemoStateResponse:
    """State for the guided landing-page simulation: run, series, and limits."""
    return _dashboard(request).demo()


@router.get("/fleet", response_model=FleetOverviewResponse, tags=["dashboard"])
def fleet_overview(
    request: Request,
    limit: Annotated[int | None, Query(ge=1, le=200)] = None,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> FleetOverviewResponse:
    return _dashboard(request).fleet(limit=_bounded_limit(request, limit), offset=offset)


@router.get(
    "/assets/{asset_id}/dashboard", response_model=AssetDashboardResponse, tags=["dashboard"]
)
def asset_dashboard(
    asset_id: uuid.UUID,
    request: Request,
    sensors: Annotated[list[str] | None, Query()] = None,
    limit: Annotated[int | None, Query(ge=1, le=200)] = None,
) -> AssetDashboardResponse:
    selected = tuple(sensors or request.app.state.settings.dashboard_default_sensor_columns)
    return _dashboard(request).asset(
        asset_id, sensor_columns=selected, limit=_bounded_limit(request, limit)
    )


@router.get("/predictions/history", response_model=PredictionHistoryResponse, tags=["dashboard"])
def prediction_history(
    request: Request,
    limit: Annotated[int | None, Query(ge=1, le=200)] = None,
    offset: Annotated[int, Query(ge=0)] = 0,
    asset_id: uuid.UUID | None = None,
    risk_level: str | None = None,
    model_version: str | None = None,
    since: datetime | None = None,
) -> PredictionHistoryResponse:
    if since is not None:
        if since.tzinfo is None or since.utcoffset() is None:
            raise RequestParameterError("The since filter must include a timezone offset.")
        since = since.astimezone(UTC)
    return _dashboard(request).prediction_history(
        limit=_bounded_limit(request, limit),
        offset=offset,
        asset_id=asset_id,
        risk_level=risk_level,
        model_version=model_version,
        since=since,
    )


@router.get("/alerts", response_model=AlertSummaryResponse, tags=["dashboard"])
def alert_summary(
    request: Request, limit: Annotated[int | None, Query(ge=1, le=200)] = None
) -> AlertSummaryResponse:
    return _dashboard(request).alerts(limit=_bounded_limit(request, limit))


@router.get("/models/overview", response_model=ModelOverviewResponse, tags=["dashboard"])
def model_overview(request: Request) -> ModelOverviewResponse:
    return _dashboard(request).model()


@router.get("/monitoring/drift", response_model=DriftDetailResponse, tags=["dashboard"])
def drift_detail(request: Request) -> DriftDetailResponse:
    return _dashboard(request).drift()


@router.get("/monitoring/performance", response_model=PerformanceDetailResponse, tags=["dashboard"])
def performance_detail(request: Request) -> PerformanceDetailResponse:
    return _dashboard(request).performance()


@router.get("/replay", response_model=ReplayStatusResponse, tags=["replay"])
def replay_status(
    request: Request, limit: Annotated[int, Query(ge=1, le=100)] = 20
) -> ReplayStatusResponse:
    return _replay(request).status(limit=limit)


@router.post("/replay/actions", response_model=ReplayActionResponse, tags=["replay"])
def replay_action(
    payload: ReplayActionRequest,
    request: Request,
    control_token: Annotated[str | None, Header(alias="X-Replay-Control-Token")] = None,
) -> ReplayActionResponse:
    client = "unknown" if request.client is None else request.client.host
    command = payload.model_copy(update={"control_token": control_token})
    return _replay(request).perform(command, client_id=client)

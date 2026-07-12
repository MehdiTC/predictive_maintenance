"""Thin versioned routes over the Loop 7 application service."""

import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Query, Request, Response, status

from turbine_guard.api.schemas.online import (
    AssetDetailResponse,
    AssetHealthResponse,
    AssetListResponse,
    CurrentModelResponse,
    ErrorResponse,
    MonitoringSummaryResponse,
    RecentPredictionsResponse,
    SensorIngestionResponse,
    SensorReadingRequest,
)
from turbine_guard.services.errors import RequestParameterError
from turbine_guard.services.inference import OnlineInferenceService, SensorObservation

router = APIRouter(prefix="/v1")
ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    status.HTTP_422_UNPROCESSABLE_CONTENT: {"model": ErrorResponse},
    status.HTTP_409_CONFLICT: {"model": ErrorResponse},
    status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse},
}


def _service(request: Request) -> OnlineInferenceService:
    service: OnlineInferenceService | None = request.app.state.online_service
    if service is None:
        raise RuntimeError("Online inference is disabled.")
    return service


def _limit(request: Request, value: int | None) -> int:
    selected = request.app.state.settings.api_default_page_size if value is None else value
    if selected > request.app.state.settings.api_max_page_size:
        raise RequestParameterError("Requested limit exceeds the configured maximum.")
    return int(selected)


@router.post(
    "/sensor-readings",
    response_model=SensorIngestionResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["inference"],
    summary="Persist a sensor cycle and produce the champion prediction",
    responses={
        status.HTTP_200_OK: {
            "model": SensorIngestionResponse,
            "description": "Exact reading and current-model prediction retry.",
        },
        **ERROR_RESPONSES,
    },
)
def ingest_sensor_reading(
    payload: SensorReadingRequest, request: Request, response: Response
) -> SensorIngestionResponse:
    """Atomically store one contiguous cycle and its model-version-pinned prediction."""
    observation = SensorObservation(
        external_asset_id=payload.external_asset_id,
        cycle=payload.cycle,
        observed_at=payload.observed_at,
        operating_settings=payload.operating_settings,
        sensor_values=payload.sensor_values,
        source=payload.source,
        ingestion_id=payload.ingestion_id,
        schema_version=payload.schema_version,
    )
    result = _service(request).ingest(observation, request.state.request_id)
    if result.idempotent:
        response.status_code = status.HTTP_200_OK
    return result


@router.get("/assets", response_model=AssetListResponse, tags=["assets"])
def list_assets(
    request: Request,
    limit: Annotated[int | None, Query(ge=1, le=200)] = None,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> AssetListResponse:
    """List bounded asset summaries in deterministic external-ID order."""
    return _service(request).list_assets(limit=_limit(request, limit), offset=offset)


@router.get(
    "/assets/{asset_id}",
    response_model=AssetDetailResponse,
    tags=["assets"],
    responses={status.HTTP_404_NOT_FOUND: {"model": ErrorResponse}},
)
def get_asset(asset_id: uuid.UUID, request: Request) -> AssetDetailResponse:
    """Return asset metadata and its latest operational state."""
    return _service(request).get_asset(asset_id)


@router.get(
    "/assets/{asset_id}/health",
    response_model=AssetHealthResponse,
    tags=["assets"],
    responses={status.HTTP_404_NOT_FOUND: {"model": ErrorResponse}},
)
def get_asset_health(asset_id: uuid.UUID, request: Request) -> AssetHealthResponse:
    """Return current RUL, risk, staleness, and a bounded prediction trend."""
    return _service(request).get_asset_health(asset_id)


@router.get("/predictions/recent", response_model=RecentPredictionsResponse, tags=["predictions"])
def recent_predictions(
    request: Request,
    limit: Annotated[int | None, Query(ge=1, le=200)] = None,
    asset_id: uuid.UUID | None = None,
) -> RecentPredictionsResponse:
    """Return recent-first model predictions, optionally for one asset."""
    return _service(request).recent_predictions(limit=_limit(request, limit), asset_id=asset_id)


@router.get(
    "/models/current",
    response_model=CurrentModelResponse,
    tags=["models"],
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse}},
)
def current_model(request: Request) -> CurrentModelResponse:
    """Return the cached MLflow champion identity and recorded evidence."""
    return _service(request).current_model()


@router.get(
    "/monitoring/summary",
    response_model=MonitoringSummaryResponse,
    tags=["monitoring"],
)
def monitoring_summary(request: Request) -> MonitoringSummaryResponse:
    """Return only currently observable service and persistence counts."""
    return _service(request).monitoring_summary()

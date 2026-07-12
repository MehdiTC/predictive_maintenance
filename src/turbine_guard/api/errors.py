"""Safe structured exception mapping for the online API."""

import logging
from typing import Any

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from turbine_guard.database.errors import PredictionConflictError, SensorReadingConflictError
from turbine_guard.observability.metrics import OnlineMetrics
from turbine_guard.services.errors import ServiceError

logger = logging.getLogger(__name__)


def _request_id(request: Request) -> str:
    return str(getattr(request.state, "request_id", "unavailable"))


def _payload(
    request: Request, code: str, message: str, details: list[dict[str, Any]] | None = None
) -> dict[str, object]:
    error: dict[str, object] = {
        "code": code,
        "message": message,
        "request_id": _request_id(request),
    }
    if details is not None:
        error["details"] = details
    return {"error": error}


def _metrics(request: Request) -> OnlineMetrics:
    metrics: OnlineMetrics = request.app.state.metrics
    return metrics


async def service_error_handler(request: Request, exc: ServiceError) -> JSONResponse:
    if exc.status_code == 409:
        _metrics(request).record_failure("conflict")
    elif exc.status_code == 503:
        _metrics(request).record_failure("model")
    if request.url.path == "/v1/sensor-readings" and exc.status_code >= 500:
        _metrics(request).record_failure("prediction")
    logger.warning(
        "online_request_failed",
        extra={"request_id": _request_id(request), "error_code": exc.code},
    )
    return JSONResponse(_payload(request, exc.code, str(exc)), status_code=exc.status_code)


async def persistence_conflict_handler(
    request: Request, exc: SensorReadingConflictError | PredictionConflictError
) -> JSONResponse:
    _metrics(request).record_failure("conflict")
    code = (
        "sensor_reading_conflict"
        if isinstance(exc, SensorReadingConflictError)
        else "prediction_conflict"
    )
    return JSONResponse(_payload(request, code, str(exc)), status_code=409)


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    details = [
        {"field": ".".join(str(part) for part in error["loc"]), "type": error["type"]}
        for error in exc.errors()
    ]
    return JSONResponse(
        _payload(request, "request_validation_failed", "Request validation failed.", details),
        status_code=422,
    )


async def database_error_handler(request: Request, exc: SQLAlchemyError) -> JSONResponse:
    _metrics(request).record_failure("database")
    logger.exception(
        "database_request_failed",
        exc_info=exc,
        extra={"request_id": _request_id(request), "error_code": "database_unavailable"},
    )
    return JSONResponse(
        _payload(request, "database_unavailable", "The operational database is unavailable."),
        status_code=503,
    )


async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
    if request.url.path == "/v1/sensor-readings":
        _metrics(request).record_failure("prediction")
    logger.exception(
        "unexpected_request_failure",
        exc_info=exc,
        extra={"request_id": _request_id(request), "error_code": "internal_error"},
    )
    return JSONResponse(
        _payload(request, "internal_error", "An unexpected internal error occurred."),
        status_code=500,
    )

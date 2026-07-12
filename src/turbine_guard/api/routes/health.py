"""Liveness and readiness endpoints."""

from fastapi import APIRouter, Request, Response, status

from turbine_guard.api.schemas.health import LivenessResponse, ReadinessResponse
from turbine_guard.services.health import check_readiness

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live", summary="Liveness probe")
def read_liveness() -> LivenessResponse:
    """Report that the application process is running and able to respond."""
    return LivenessResponse()


@router.get(
    "/ready",
    summary="Readiness probe",
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ReadinessResponse}},
)
def read_readiness(request: Request, response: Response) -> ReadinessResponse:
    """Report whether the service's dependencies allow it to handle requests.

    Offline/test applications may inject an empty map. Loop 7 online mode
    requires PostgreSQL, the champion model, and feature compatibility.
    """
    result = check_readiness(request.app.state.readiness_checks)
    request.app.state.metrics.ready.set(1 if result.ready else 0)
    if not result.ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(
        status="ready" if result.ready else "not_ready",
        checks=result.checks,
    )

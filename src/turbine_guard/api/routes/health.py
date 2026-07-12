"""Liveness and readiness endpoints."""

from fastapi import APIRouter, Response, status

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
def read_readiness(response: Response) -> ReadinessResponse:
    """Report whether the service's dependencies allow it to handle requests.

    Loop 0 has no external dependencies, so the check map is empty and the
    service is always ready. Later loops add real dependency checks, which
    turn this into a 503 when something required is unavailable.
    """
    result = check_readiness()
    if not result.ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(
        status="ready" if result.ready else "not_ready",
        checks=result.checks,
    )

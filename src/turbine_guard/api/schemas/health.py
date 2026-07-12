"""Response models for the health endpoints."""

from typing import Literal

from pydantic import BaseModel, Field


class LivenessResponse(BaseModel):
    """Response for ``GET /health/live``."""

    status: Literal["alive"] = "alive"


class ReadinessResponse(BaseModel):
    """Response for ``GET /health/ready``."""

    status: Literal["ready", "not_ready"]
    checks: dict[str, bool] = Field(
        default_factory=dict,
        description="Dependency name mapped to whether it is currently available.",
    )

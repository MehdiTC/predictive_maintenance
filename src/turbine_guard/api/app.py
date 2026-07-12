"""FastAPI application factory."""

import logging
from collections.abc import Callable, Mapping

from fastapi import FastAPI

from turbine_guard import __version__
from turbine_guard.api.routes import health
from turbine_guard.config.settings import Settings, get_settings
from turbine_guard.logging_config import configure_logging

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    *,
    readiness_checks: Mapping[str, Callable[[], bool]] | None = None,
) -> FastAPI:
    """Create the TurbineGuard FastAPI application.

    Args:
        settings: Application settings. When omitted, settings are loaded
            from the environment; tests can inject explicit values.
    """
    app_settings = settings if settings is not None else get_settings()
    configure_logging(app_settings.log_level)

    app = FastAPI(
        title="TurbineGuard",
        description="Predictive-maintenance platform for turbine sensor data.",
        version=__version__,
    )
    app.state.settings = app_settings
    app.state.readiness_checks = dict(readiness_checks or {})
    app.include_router(health.router)

    logger.info(
        "application_created",
        extra={
            "app_name": app_settings.app_name,
            "environment": app_settings.environment.value,
        },
    )
    return app

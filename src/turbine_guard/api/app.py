"""FastAPI application factory and explicit Loop 7 resource lifespan."""

import logging
import re
import time
import uuid
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST
from sqlalchemy import Engine
from sqlalchemy.exc import SQLAlchemyError
from starlette.middleware.base import RequestResponseEndpoint
from starlette.middleware.trustedhost import TrustedHostMiddleware

from turbine_guard import __version__
from turbine_guard.api.errors import (
    database_error_handler,
    persistence_conflict_handler,
    service_error_handler,
    unexpected_error_handler,
    validation_error_handler,
)
from turbine_guard.api.routes import health, online
from turbine_guard.config.settings import Settings, get_settings
from turbine_guard.database.errors import PredictionConflictError, SensorReadingConflictError
from turbine_guard.database.session import (
    DatabaseConfig,
    check_database_connection,
    create_database_engine,
    create_session_factory,
)
from turbine_guard.logging_config import configure_logging
from turbine_guard.observability.metrics import OnlineMetrics
from turbine_guard.services.errors import ServiceError
from turbine_guard.services.inference import OnlineInferenceService
from turbine_guard.serving.model_loader import ChampionModelLoader

logger = logging.getLogger(__name__)
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def create_app(
    settings: Settings | None = None,
    *,
    readiness_checks: Mapping[str, Callable[[], bool]] | None = None,
    online_service: OnlineInferenceService | None = None,
    metrics: OnlineMetrics | None = None,
) -> FastAPI:
    """Create an app with lazy lifespan resources and injectable test boundaries."""
    app_settings = settings if settings is not None else get_settings()
    configure_logging(app_settings.log_level)
    app_metrics = metrics or OnlineMetrics()
    engine: Engine | None = None
    loader: ChampionModelLoader | None = None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        nonlocal engine, loader
        checks = dict(readiness_checks or {})
        if app_settings.online_inference_enabled and online_service is None:
            engine = create_database_engine(DatabaseConfig.from_settings(app_settings))
            loader = ChampionModelLoader(app_settings)
            app.state.online_service = OnlineInferenceService(
                create_session_factory(engine), loader, app_metrics, app_settings
            )

            def database_ready() -> bool:
                available = check_database_connection(engine)
                if not available:
                    app_metrics.record_failure("database")
                return available

            def model_ready() -> bool:
                available = loader.check_model()
                if not available:
                    app_metrics.record_failure("model")
                return available

            checks = {
                "database": database_ready,
                "model": model_ready,
                "feature_contract": loader.check_feature_contract,
                **checks,
            }
            if app_settings.model_preload_enabled:
                try:
                    loaded = loader.get()
                    app_metrics.set_model(
                        loaded.metadata.model_name,
                        loaded.metadata.version,
                        loaded.metadata.alias,
                    )
                except Exception:
                    app_metrics.record_failure("model")
                    logger.exception("champion_preload_failed")
        else:
            app.state.online_service = online_service
        app.state.readiness_checks = checks
        try:
            yield
        finally:
            if engine is not None:
                engine.dispose()

    docs_url = "/docs" if app_settings.api_docs_enabled else None
    app = FastAPI(
        title="TurbineGuard",
        description="Versioned predictive-maintenance inference and asset-health service.",
        version=__version__,
        docs_url=docs_url,
        redoc_url=None,
        openapi_url="/openapi.json" if app_settings.api_docs_enabled else None,
        lifespan=lifespan,
    )
    app.state.settings = app_settings
    app.state.metrics = app_metrics
    app.state.online_service = online_service
    app.state.readiness_checks = dict(readiness_checks or {})

    if app_settings.cors_allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(app_settings.cors_allowed_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "X-Request-ID"],
        )
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(app_settings.trusted_hosts))

    @app.middleware("http")
    async def request_context(request: Request, call_next: RequestResponseEndpoint) -> Response:
        supplied = request.headers.get("X-Request-ID", "")
        request_id = supplied if REQUEST_ID_PATTERN.fullmatch(supplied) else str(uuid.uuid4())
        request.state.request_id = request_id
        content_length = request.headers.get("content-length")
        started = time.perf_counter()
        response: Response
        if content_length is not None and int(content_length) > app_settings.api_max_request_bytes:
            response = JSONResponse(
                {
                    "error": {
                        "code": "request_too_large",
                        "message": "Request body exceeds the configured limit.",
                        "request_id": request_id,
                    }
                },
                status_code=413,
            )
        else:
            response = await call_next(request)
        duration = time.perf_counter() - started
        route = request.scope.get("route")
        route_path = getattr(route, "path", "unmatched")
        app_metrics.record_http(route_path, request.method, response.status_code, duration)
        logger.info(
            "http_request_completed",
            extra={
                "request_id": request_id,
                "route": route_path,
                "method": request.method,
                "status": response.status_code,
                "latency_ms": duration * 1000.0,
            },
        )
        response.headers["X-Request-ID"] = request_id
        return response

    app.add_exception_handler(ServiceError, service_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(SensorReadingConflictError, persistence_conflict_handler)  # type: ignore[arg-type]
    app.add_exception_handler(PredictionConflictError, persistence_conflict_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, validation_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(SQLAlchemyError, database_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, unexpected_error_handler)

    @app.get("/metrics", include_in_schema=False)
    def prometheus_metrics() -> Response:
        return Response(app_metrics.render(), headers={"Content-Type": CONTENT_TYPE_LATEST})

    app.include_router(health.router)
    if app_settings.online_inference_enabled or online_service is not None:
        app.include_router(online.router)

    logger.info(
        "application_created",
        extra={
            "app_name": app_settings.app_name,
            "environment": app_settings.environment.value,
            "online_inference_enabled": app_settings.online_inference_enabled,
        },
    )
    return app

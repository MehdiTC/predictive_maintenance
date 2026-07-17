"""FastAPI application factory and explicit Loop 7 resource lifespan."""

import logging
import re
import time
import uuid
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST
from sqlalchemy import Engine
from sqlalchemy.exc import SQLAlchemyError
from starlette.middleware.base import RequestResponseEndpoint
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.staticfiles import StaticFiles

from turbine_guard import __version__
from turbine_guard.api.errors import (
    database_error_handler,
    persistence_conflict_handler,
    service_error_handler,
    unexpected_error_handler,
    validation_error_handler,
)
from turbine_guard.api.routes import dashboard, dashboard_api, health, online
from turbine_guard.config.settings import Environment, Settings, get_settings
from turbine_guard.database.errors import PredictionConflictError, SensorReadingConflictError
from turbine_guard.database.session import (
    DatabaseConfig,
    check_database_connection,
    create_database_engine,
    create_session_factory,
)
from turbine_guard.logging_config import configure_logging
from turbine_guard.observability.metrics import OnlineMetrics
from turbine_guard.services.dashboard import DashboardService
from turbine_guard.services.errors import ServiceError
from turbine_guard.services.inference import OnlineInferenceService
from turbine_guard.services.replay_control import ReplayControlService
from turbine_guard.serving.champion import ChampionLoader

logger = logging.getLogger(__name__)
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def _create_champion_loader(settings: Settings) -> ChampionLoader:
    """Select the configured champion source.

    The deployment-bundle loader deliberately avoids importing MLflow, so the
    free public demo never pays MLflow's import cost; import lazily here.
    """
    if settings.model_source == "deployment_bundle":
        from turbine_guard.serving.bundle_loader import DeploymentBundleLoader

        return DeploymentBundleLoader(settings)
    from turbine_guard.serving.model_loader import ChampionModelLoader

    return ChampionModelLoader(settings)


def create_app(
    settings: Settings | None = None,
    *,
    readiness_checks: Mapping[str, Callable[[], bool]] | None = None,
    online_service: OnlineInferenceService | None = None,
    dashboard_service: DashboardService | None = None,
    replay_control: ReplayControlService | None = None,
    metrics: OnlineMetrics | None = None,
) -> FastAPI:
    """Create an app with lazy lifespan resources and injectable test boundaries."""
    app_settings = settings if settings is not None else get_settings()
    configure_logging(app_settings.log_level)
    app_metrics = metrics or OnlineMetrics()
    engine: Engine | None = None
    loader: ChampionLoader | None = None
    owned_replay_control: ReplayControlService | None = None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        nonlocal engine, loader, owned_replay_control
        checks = dict(readiness_checks or {})
        if app_settings.online_inference_enabled and online_service is None:
            engine = create_database_engine(DatabaseConfig.from_settings(app_settings))
            loader = _create_champion_loader(app_settings)
            app.state.online_service = OnlineInferenceService(
                create_session_factory(engine), loader, app_metrics, app_settings
            )
            sessions = create_session_factory(engine)
            app.state.dashboard_service = dashboard_service or DashboardService(
                sessions, loader, app_settings
            )
            if replay_control is None:
                owned_replay_control = ReplayControlService.create(sessions, app_settings)
            app.state.replay_control = replay_control or owned_replay_control

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
            app.state.dashboard_service = dashboard_service
            app.state.replay_control = replay_control
        app.state.readiness_checks = checks
        try:
            yield
        finally:
            if owned_replay_control is not None:
                owned_replay_control.close()
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
    app.state.dashboard_service = dashboard_service
    app.state.replay_control = replay_control
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
        oversized = False
        if content_length is not None:
            try:
                oversized = int(content_length) > app_settings.api_max_request_bytes
            except ValueError:
                oversized = True
        if oversized:
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
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self' https://cdn.plot.ly; "
            "style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; "
            "font-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'self'; "
            "form-action 'self'"
        )
        if app_settings.environment is Environment.PRODUCTION:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
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
    if app_settings.dashboard_enabled:
        static_path = Path(__file__).resolve().parents[1] / "dashboard" / "static"
        app.mount("/static", StaticFiles(directory=static_path), name="static")
        app.include_router(dashboard_api.router)
        app.include_router(dashboard.router)

    logger.info(
        "application_created",
        extra={
            "app_name": app_settings.app_name,
            "environment": app_settings.environment.value,
            "online_inference_enabled": app_settings.online_inference_enabled,
        },
    )
    return app

"""Server-rendered Loop 11 pages with progressive chart enhancement."""

import hmac
import uuid
from hashlib import sha256
from pathlib import Path
from typing import Literal, cast
from urllib.parse import parse_qs, urlencode

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

from turbine_guard.api.schemas.dashboard import ReplayActionRequest
from turbine_guard.services.dashboard import DashboardService
from turbine_guard.services.errors import AssetNotFoundError, ServiceError
from turbine_guard.services.replay_control import ReplayControlService

router = APIRouter(include_in_schema=False)
templates = Jinja2Templates(
    directory=Path(__file__).resolve().parents[2] / "dashboard" / "templates"
)


def _dashboard(request: Request) -> DashboardService:
    service: DashboardService | None = request.app.state.dashboard_service
    if service is None:
        raise RuntimeError("Dashboard data access is unavailable.")
    return service


def _replay(request: Request) -> ReplayControlService:
    service: ReplayControlService | None = request.app.state.replay_control
    if service is None:
        raise RuntimeError("Replay status is unavailable.")
    return service


@router.get("/", response_class=HTMLResponse)
def demo_page(request: Request) -> HTMLResponse:
    """Story-driven landing page: one narrative, one chart, one button."""
    try:
        demo = _dashboard(request).demo().model_dump(mode="json")
    except Exception:
        demo = None
    return templates.TemplateResponse(
        request, "demo.html", _context(request, page="demo", demo=demo)
    )


@router.get("/dashboard", response_class=HTMLResponse)
def fleet_page(request: Request, offset: int = 0) -> HTMLResponse:
    try:
        fleet = _dashboard(request).fleet(
            limit=request.app.state.settings.api_default_page_size, offset=max(0, offset)
        )
        alerts = _dashboard(request).alerts(limit=50)
        replay = _replay(request).status(limit=10)
        return templates.TemplateResponse(
            request,
            "fleet.html",
            _context(
                request,
                page="fleet",
                fleet=fleet,
                alerts=alerts,
                replay=replay,
                csrf_token=_csrf_token(request),
                message=request.query_params.get("message"),
            ),
        )
    except Exception:
        return _degraded(request, "Fleet data is temporarily unavailable.")


@router.get("/dashboard/assets/{asset_id}", response_class=HTMLResponse)
def asset_page(request: Request, asset_id: uuid.UUID, sensors: str | None = None) -> HTMLResponse:
    selected = (
        tuple(item.strip() for item in (sensors or "").split(",") if item.strip())
        or request.app.state.settings.dashboard_default_sensor_columns
    )
    try:
        asset = _dashboard(request).asset(
            asset_id,
            sensor_columns=selected,
            limit=request.app.state.settings.dashboard_history_limit,
        )
    except AssetNotFoundError:
        return templates.TemplateResponse(
            request,
            "error.html",
            _context(request, page="asset", error="Asset not found."),
            status_code=404,
        )
    except Exception:
        return _degraded(request, "Asset health is temporarily unavailable.")
    return templates.TemplateResponse(
        request,
        "asset.html",
        _context(request, page="asset", asset=asset),
    )


@router.get("/dashboard/predictions", response_class=HTMLResponse)
def predictions_page(
    request: Request,
    risk_level: str | None = None,
    model_version: str | None = None,
    offset: int = 0,
) -> HTMLResponse:
    try:
        history = _dashboard(request).prediction_history(
            limit=request.app.state.settings.api_default_page_size,
            offset=max(0, offset),
            risk_level=risk_level or None,
            model_version=model_version or None,
        )
        return templates.TemplateResponse(
            request,
            "predictions.html",
            _context(
                request,
                page="predictions",
                history=history,
                risk_level=risk_level or "",
                model_version=model_version or "",
            ),
        )
    except Exception:
        return _degraded(request, "Prediction history is temporarily unavailable.")


@router.get("/dashboard/models", response_class=HTMLResponse)
def models_page(request: Request) -> HTMLResponse:
    try:
        model = _dashboard(request).model()
        return templates.TemplateResponse(
            request, "models.html", _context(request, page="models", model=model)
        )
    except Exception:
        return _degraded(request, "Model registry information is temporarily unavailable.")


@router.get("/dashboard/monitoring", response_class=HTMLResponse)
def monitoring_page(request: Request) -> HTMLResponse:
    try:
        drift = _dashboard(request).drift()
        performance = _dashboard(request).performance()
        return templates.TemplateResponse(
            request,
            "monitoring.html",
            _context(request, page="monitoring", drift=drift, performance=performance),
        )
    except Exception:
        return _degraded(request, "Monitoring reports are temporarily unavailable.")


@router.post("/dashboard/replay", response_class=HTMLResponse)
async def replay_form(request: Request) -> RedirectResponse:
    body = (await request.body()).decode("utf-8", errors="strict")
    values = {key: items[-1] for key, items in parse_qs(body, keep_blank_values=True).items()}
    if not _valid_csrf(request, values.get("csrf_token")):
        return _redirect_message("Replay request rejected: invalid form token.")
    try:
        payload = ReplayActionRequest(
            action=cast(
                Literal["start", "step", "pause", "resume", "accelerate", "reset"],
                values.get("action", "step"),
            ),
            source_asset_id=_optional_int(values.get("source_asset_id")),
            run_id=_optional_uuid(values.get("run_id")),
            max_cycles=_optional_int(values.get("max_cycles")),
            confirm_reset=values.get("confirm_reset") == "yes",
            control_token=values.get("control_token") or None,
        )
        client = "unknown" if request.client is None else request.client.host
        # The control service performs blocking HTTP calls back to this same
        # server; running it on the event loop would deadlock the process.
        result = await run_in_threadpool(_replay(request).perform, payload, client_id=client)
        return _redirect_message(result.message)
    except (ValueError, ServiceError) as exc:
        return _redirect_message(f"Replay request rejected: {exc}")


def _context(request: Request, *, page: str, **values: object) -> dict[str, object]:
    return {
        "request": request,
        "page": page,
        "public_demo_mode": request.app.state.settings.public_demo_mode,
        **values,
    }


def _degraded(request: Request, message: str) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "error.html",
        _context(request, page="error", error=message),
        status_code=503,
    )


def _csrf_token(request: Request) -> str | None:
    secret = request.app.state.settings.application_secret
    if secret is None:
        return None
    return hmac.new(secret.encode(), b"dashboard-replay-v1", sha256).hexdigest()


def _valid_csrf(request: Request, supplied: str | None) -> bool:
    expected = _csrf_token(request)
    return expected is not None and supplied is not None and hmac.compare_digest(expected, supplied)


def _redirect_message(message: str) -> RedirectResponse:
    return RedirectResponse(f"/dashboard?{urlencode({'message': message})}", status_code=303)


def _optional_int(value: str | None) -> int | None:
    return None if value in (None, "") else int(value)


def _optional_uuid(value: str | None) -> uuid.UUID | None:
    return None if value in (None, "") else uuid.UUID(value)

"""Replay-side client for the real Loop 7 sensor-ingestion contract.

The client speaks plain HTTP against ``POST /v1/sensor-readings`` using the
Loop 7 request/response schemas. Payloads are deterministic per run and cycle
(including the simulated observation timestamp and ingestion ID), so a resend
after an uncertain outcome is byte-identical and the API's exact-retry
idempotency doubles as the reconciliation mechanism. Tests may inject
``fastapi.testclient.TestClient``, which is an ``httpx.Client`` subclass.
"""

import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

import httpx
from pydantic import ValidationError

from turbine_guard.api.schemas.online import SensorIngestionResponse, SensorReadingRequest
from turbine_guard.data.schema import (
    CYCLE_COLUMN,
    OPERATING_SETTING_COLUMNS,
    SCHEMA_VERSION,
    SENSOR_COLUMNS,
)
from turbine_guard.replay.errors import ReplayIngestionError, ReplayTransientError
from turbine_guard.replay.source import ReplayTrajectory

logger = logging.getLogger(__name__)

INGESTION_PATH = "/v1/sensor-readings"
REPLAY_SOURCE_NAME = "replay"
_RETRYABLE_STATUS = frozenset({500, 502, 503, 504})


def build_reading_request(
    trajectory: ReplayTrajectory,
    cycle: int,
    *,
    run_id: uuid.UUID,
    external_asset_id: str,
    replay_started_at: datetime,
    simulated_cycle_duration_seconds: float,
) -> SensorReadingRequest:
    """Build the exact Loop 7 payload for one cycle, and only that cycle.

    ``observed_at`` is simulated from the replay epoch and cycle index;
    ``ingestion_id`` is deterministic per run and cycle so retries carry the
    same identity. Nothing beyond the requested cycle's row is read.
    """
    row = trajectory.row(cycle)
    if int(row[CYCLE_COLUMN]) != cycle:
        raise ReplayIngestionError("Trajectory row does not match the requested cycle.")
    observed_at = replay_started_at + timedelta(
        seconds=(cycle - 1) * simulated_cycle_duration_seconds
    )
    return SensorReadingRequest(
        external_asset_id=external_asset_id,
        cycle=cycle,
        observed_at=observed_at,
        **{name: row[name] for name in OPERATING_SETTING_COLUMNS},
        **{name: row[name] for name in SENSOR_COLUMNS},
        source=REPLAY_SOURCE_NAME,
        ingestion_id=f"replay-run:{run_id}:cycle:{cycle}",
        schema_version=SCHEMA_VERSION,
    )


@dataclass(frozen=True)
class ReplayClientConfig:
    """Bounded-retry behavior for one replay client."""

    max_attempts: int = 5
    backoff_seconds: float = 0.5
    backoff_multiplier: float = 2.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1.")
        if self.backoff_seconds < 0 or self.backoff_multiplier < 1:
            raise ValueError("Backoff must be non-negative with multiplier >= 1.")


@dataclass(frozen=True)
class IngestionResult:
    """Confirmed API acceptance of one replayed cycle."""

    asset_id: uuid.UUID
    external_asset_id: str
    cycle: int
    idempotent: bool
    predicted_rul: float
    risk_level: str
    model_version: str
    retries: int


class ReplayIngestionClient:
    """Send one cycle at a time with idempotent retries and typed failures."""

    def __init__(
        self,
        http: httpx.Client,
        config: ReplayClientConfig | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._http = http
        self._config = config or ReplayClientConfig()
        self._sleep = sleeper

    def send_reading(self, request: SensorReadingRequest) -> IngestionResult:
        """POST one cycle; retry transient failures; never mask conflicts.

        Timeouts and 5xx responses are retried with exponential backoff by
        resending the identical payload — safe because ingestion is exactly
        idempotent. Validation failures and conflicts are permanent and raise
        :class:`ReplayIngestionError` immediately.
        """
        payload = request.model_dump(mode="json", exclude_none=True)
        delay = self._config.backoff_seconds
        last_detail = "no attempt made"
        for attempt in range(1, self._config.max_attempts + 1):
            try:
                response = self._http.post(INGESTION_PATH, json=payload)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_detail = f"transport failure: {exc.__class__.__name__}"
                logger.warning(
                    "replay_send_transport_failure",
                    extra={
                        "cycle": request.cycle,
                        "attempt": attempt,
                        "detail": last_detail,
                    },
                )
            else:
                if response.status_code in (200, 201):
                    return self._confirm(request, response, retries=attempt - 1)
                code, message = _error_detail(response)
                if response.status_code in _RETRYABLE_STATUS:
                    last_detail = f"HTTP {response.status_code} {code}: {message}"
                    logger.warning(
                        "replay_send_retryable_failure",
                        extra={
                            "cycle": request.cycle,
                            "attempt": attempt,
                            "status": response.status_code,
                            "error_code": code,
                        },
                    )
                else:
                    raise ReplayIngestionError(
                        f"Ingestion of cycle {request.cycle} was rejected permanently "
                        f"(HTTP {response.status_code} {code}): {message}"
                    )
            if attempt < self._config.max_attempts and delay > 0:
                self._sleep(delay)
                delay *= self._config.backoff_multiplier
        raise ReplayTransientError(
            f"Ingestion of cycle {request.cycle} failed after "
            f"{self._config.max_attempts} attempts; last failure: {last_detail}."
        )

    @staticmethod
    def _confirm(
        request: SensorReadingRequest, response: httpx.Response, *, retries: int
    ) -> IngestionResult:
        """Verify the API confirmed exactly the asset and cycle that was sent."""
        try:
            parsed = SensorIngestionResponse.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            raise ReplayIngestionError(
                f"Ingestion response for cycle {request.cycle} was unparsable."
            ) from exc
        if parsed.external_asset_id != request.external_asset_id or parsed.cycle != request.cycle:
            raise ReplayIngestionError(
                f"Ingestion response identity mismatch: sent {request.external_asset_id!r} "
                f"cycle {request.cycle}, received {parsed.external_asset_id!r} "
                f"cycle {parsed.cycle}."
            )
        return IngestionResult(
            asset_id=parsed.asset_id,
            external_asset_id=parsed.external_asset_id,
            cycle=parsed.cycle,
            idempotent=parsed.idempotent,
            predicted_rul=parsed.prediction.predicted_rul,
            risk_level=parsed.prediction.risk_level,
            model_version=parsed.prediction.model_version,
            retries=retries,
        )


def _error_detail(response: httpx.Response) -> tuple[str, str]:
    """Best-effort structured error extraction without trusting the body."""
    try:
        body = response.json()
        error = body.get("error", {}) if isinstance(body, dict) else {}
        return str(error.get("code", "unknown")), str(error.get("message", "no message"))
    except ValueError:
        return "unknown", "response body was not JSON"

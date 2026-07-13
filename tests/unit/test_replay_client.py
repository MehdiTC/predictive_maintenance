"""Replay ingestion client: retries, conflicts, timeouts, and verification."""

import json
import uuid
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from turbine_guard.api.schemas.online import SensorReadingRequest
from turbine_guard.replay.client import (
    INGESTION_PATH,
    ReplayClientConfig,
    ReplayIngestionClient,
)
from turbine_guard.replay.errors import ReplayIngestionError, ReplayTransientError

ASSET_UUID = str(uuid.uuid4())


def _request(cycle: int = 1) -> SensorReadingRequest:
    return SensorReadingRequest(
        external_asset_id="replay-FD001-001",
        cycle=cycle,
        observed_at="2026-07-13T12:00:00Z",  # type: ignore[arg-type]
        operating_setting_1=0.1,
        operating_setting_2=0.2,
        operating_setting_3=100.0,
        **{f"sensor_{index:02d}": float(index) for index in range(1, 22)},
        source="replay",
        ingestion_id=f"replay-run:{uuid.uuid4()}:cycle:{cycle}",
        schema_version="1",
    )


def _success_body(
    cycle: int,
    *,
    external_asset_id: str = "replay-FD001-001",
    idempotent: bool = False,
) -> dict[str, Any]:
    return {
        "request_id": "req-1",
        "asset_id": ASSET_UUID,
        "external_asset_id": external_asset_id,
        "cycle": cycle,
        "reading_id": str(uuid.uuid4()),
        "prediction": {
            "predicted_rul": 88.0,
            "lower_rul": 80.0,
            "upper_rul": 96.0,
            "risk_level": "healthy",
            "failure_within_30": False,
            "failure_within_50": False,
            "model_name": "fake-rul",
            "model_version": "1",
            "model_alias": "champion",
            "model_run_id": "run-1",
            "feature_version": "1",
            "prediction_timestamp": "2026-07-13T12:00:01Z",
            "latency_ms": 5.0,
        },
        "idempotent": idempotent,
        "reading_idempotent": idempotent,
        "prediction_idempotent": idempotent,
        "data_quality_warnings": [],
    }


def _error_body(code: str) -> dict[str, Any]:
    return {"error": {"code": code, "message": f"{code} happened", "request_id": "req-1"}}


def _client(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    max_attempts: int = 3,
    backoff: float = 0.01,
) -> tuple[ReplayIngestionClient, list[float]]:
    delays: list[float] = []
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://api")
    client = ReplayIngestionClient(
        http,
        ReplayClientConfig(max_attempts=max_attempts, backoff_seconds=backoff),
        sleeper=delays.append,
    )
    return client, delays


class TestSuccessfulIngestion:
    def test_accepted_cycle_returns_confirmed_identity(self) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(201, json=_success_body(1))

        client, _ = _client(handler)
        result = client.send_reading(_request(1))
        assert result.cycle == 1
        assert str(result.asset_id) == ASSET_UUID
        assert result.idempotent is False
        assert result.retries == 0
        assert seen[0].url.path == INGESTION_PATH
        payload = json.loads(seen[0].content)
        assert payload["cycle"] == 1
        assert payload["source"] == "replay"

    def test_exact_retry_returns_idempotent_result(self) -> None:
        client, _ = _client(lambda _: httpx.Response(200, json=_success_body(2, idempotent=True)))
        result = client.send_reading(_request(2))
        assert result.idempotent is True

    def test_identity_mismatch_is_rejected(self) -> None:
        client, _ = _client(lambda _: httpx.Response(201, json=_success_body(9)))
        with pytest.raises(ReplayIngestionError, match="identity mismatch"):
            client.send_reading(_request(1))

    def test_unparsable_success_body_is_rejected(self) -> None:
        client, _ = _client(lambda _: httpx.Response(201, json={"weird": True}))
        with pytest.raises(ReplayIngestionError, match="unparsable"):
            client.send_reading(_request(1))


class TestPermanentFailures:
    @pytest.mark.parametrize(
        ("status", "code"),
        [
            (409, "sensor_reading_conflict"),
            (409, "history_conflict"),
            (422, "request_validation_failed"),
        ],
    )
    def test_conflicts_and_validation_never_retry(self, status: int, code: str) -> None:
        calls: list[int] = []

        def handler(_: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(status, json=_error_body(code))

        client, delays = _client(handler)
        with pytest.raises(ReplayIngestionError, match=code):
            client.send_reading(_request(1))
        assert len(calls) == 1
        assert delays == []


class TestTransientFailures:
    def test_503_then_success_retries_with_backoff(self) -> None:
        responses = [
            httpx.Response(503, json=_error_body("model_unavailable")),
            httpx.Response(503, json=_error_body("database_unavailable")),
            httpx.Response(201, json=_success_body(1)),
        ]

        def handler(_: httpx.Request) -> httpx.Response:
            return responses.pop(0)

        client, delays = _client(handler, max_attempts=5, backoff=0.01)
        result = client.send_reading(_request(1))
        assert result.retries == 2
        assert delays == [0.01, 0.02]

    def test_timeout_then_success_reconciles_idempotently(self) -> None:
        first = True

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal first
            if first:
                first = False
                raise httpx.ConnectTimeout("timed out", request=request)
            return httpx.Response(200, json=_success_body(1, idempotent=True))

        client, _ = _client(handler)
        result = client.send_reading(_request(1))
        assert result.idempotent is True
        assert result.retries == 1

    def test_exhausted_retries_raise_transient_error(self) -> None:
        calls: list[int] = []

        def handler(_: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(503, json=_error_body("database_unavailable"))

        client, _ = _client(handler, max_attempts=3)
        with pytest.raises(ReplayTransientError, match="after 3 attempts"):
            client.send_reading(_request(1))
        assert len(calls) == 3

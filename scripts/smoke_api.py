#!/usr/bin/env python3
"""Exercise the containerized API health and one deterministic ingestion request."""

import argparse
import json
import sys
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Any


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="smoke_api")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args(argv)
    base_url = args.base_url.rstrip("/")
    live = _request(f"{base_url}/health/live")
    if live[0] != 200 or live[1] != {"status": "alive"}:
        raise RuntimeError(f"Liveness failed: {live!r}")
    ready = _request(f"{base_url}/health/ready")
    checks = ready[1].get("checks", {}) if isinstance(ready[1], dict) else {}
    if ready[0] != 200 or ready[1].get("status") != "ready" or not all(checks.values()):
        raise RuntimeError(f"Readiness failed: {ready!r}")
    for path in ("/docs", "/metrics"):
        status, _ = _request(f"{base_url}{path}", expect_json=False)
        if status != 200:
            raise RuntimeError(f"GET {path} returned HTTP {status}.")

    status, response = _request(
        f"{base_url}/v1/sensor-readings",
        method="POST",
        payload=_sensor_payload(),
    )
    if status not in (200, 201):
        raise RuntimeError(f"Ingestion returned HTTP {status}: {response!r}")
    prediction = response.get("prediction") if isinstance(response, dict) else None
    if not isinstance(prediction, dict) or not prediction.get("model_version"):
        raise RuntimeError(f"Ingestion returned no versioned prediction: {response!r}")
    sys.stdout.write(
        json.dumps(
            {
                "status": "passed",
                "readiness": checks,
                "ingestion_status": status,
                "model_version": prediction["model_version"],
            },
            sort_keys=True,
        )
        + "\n"
    )
    return 0


def _request(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    expect_json: bool = True,
) -> tuple[int, Any]:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            content = response.read()
            parsed = json.loads(content) if expect_json else content.decode(errors="replace")
            return response.status, parsed
    except urllib.error.HTTPError as exc:
        content = exc.read()
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = content.decode(errors="replace")
        return exc.code, parsed


def _sensor_payload() -> dict[str, Any]:
    cycle = 1
    asset_id = 1
    sensors: dict[str, float] = {}
    for index in range(1, 22):
        if index in (1, 5):
            value = 518.67
        elif index == 6:
            value = 21.61
        else:
            value = 100 + index + 0.5 * cycle + 0.1 * asset_id
        sensors[f"sensor_{index:02d}"] = value
    return {
        "external_asset_id": "compose-smoke-asset",
        "cycle": cycle,
        "observed_at": "2026-07-13T00:00:00Z",
        "operating_setting_1": 0.002,
        "operating_setting_2": -0.0002,
        "operating_setting_3": 100.0,
        **sensors,
        "source": "compose-smoke",
        "schema_version": "1",
        "ingestion_id": "compose-smoke-asset-cycle-1",
    }


if __name__ == "__main__":
    raise SystemExit(main())

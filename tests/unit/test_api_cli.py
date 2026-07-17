"""Tests for the production Uvicorn entry point."""

from typing import Any

import pytest

from turbine_guard.api import cli
from turbine_guard.config.settings import Environment, Settings


def test_production_api_entrypoint_uses_typed_bind_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []
    settings = Settings(
        environment=Environment.TESTING,
        online_inference_enabled=False,
        api_host="0.0.0.0",
        api_port=8123,
    )
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(
        cli.uvicorn,
        "run",
        lambda app, **kwargs: calls.append((app, kwargs)),
    )

    assert cli.main([]) == 0
    assert calls == [
        (
            "turbine_guard.api.app:create_app",
            {
                "factory": True,
                "host": "0.0.0.0",
                "port": 8123,
                "access_log": False,
                "proxy_headers": False,
                "forwarded_allow_ips": "127.0.0.1",
                "timeout_graceful_shutdown": 30,
            },
        )
    ]


def test_production_api_entrypoint_rejects_positional_arguments() -> None:
    with pytest.raises(ValueError, match="configured through"):
        cli.main(["--reload"])

"""Production Uvicorn entry point driven by typed application settings."""

from collections.abc import Sequence

import uvicorn

from turbine_guard.config.settings import get_settings


def main(argv: Sequence[str] | None = None) -> int:
    """Start one production-style API process and return after graceful shutdown."""
    if argv:
        raise ValueError("The API command is configured through TURBINE_GUARD_* settings.")
    settings = get_settings()
    uvicorn.run(
        "turbine_guard.api.app:create_app",
        factory=True,
        host=settings.api_host,
        port=settings.api_port,
        access_log=False,
        proxy_headers=settings.proxy_headers_enabled,
        forwarded_allow_ips=settings.forwarded_allow_ips,
        timeout_graceful_shutdown=30,
    )
    return 0

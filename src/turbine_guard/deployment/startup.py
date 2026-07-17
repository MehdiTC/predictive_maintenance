"""Free-tier demo startup: restore the pinned bundle, migrate, then serve.

Normal startup never downloads data, trains, or mutates a model registry.
The only network fetch is the checksum-pinned immutable deployment bundle,
and Alembic migrations are idempotent, so repeated cold starts on ephemeral
free compute converge to the same state. Operational history (readings,
predictions, replay runs, reports) lives in the external PostgreSQL
database and survives compute restarts.
"""

import logging
from collections.abc import Sequence
from pathlib import Path

from alembic import command
from alembic.config import Config

from turbine_guard.config.settings import Settings, get_settings
from turbine_guard.deployment.manifest import BundleError
from turbine_guard.deployment.restore import restore_deployment_bundle
from turbine_guard.logging_config import configure_logging

logger = logging.getLogger(__name__)

ALEMBIC_INI = Path("alembic.ini")


def prepare_demo_runtime(settings: Settings) -> None:
    """Restore the pinned bundle when configured, then migrate the database."""
    if settings.deployment_bundle_url is not None:
        result = restore_deployment_bundle(settings)
        logger.info(
            "demo_bundle_ready",
            extra={
                "status": result.status.value,
                "registry_version": result.manifest.registry_version,
            },
        )
    if not ALEMBIC_INI.is_file():
        raise BundleError(
            f"{ALEMBIC_INI} was not found in the working directory; "
            "run the demo entry point from the application root."
        )
    command.upgrade(Config(str(ALEMBIC_INI)), "head")
    logger.info("demo_migrations_applied")


def main(argv: Sequence[str] | None = None) -> int:
    """Prepare the demo runtime and hand over to the normal API entry point."""
    if argv:
        raise ValueError("The demo entry point is configured through TURBINE_GUARD_* settings.")
    settings = get_settings()
    configure_logging(settings.log_level)
    try:
        prepare_demo_runtime(settings)
    except Exception:
        logger.exception("demo_startup_failed")
        return 1
    from turbine_guard.api.cli import main as api_main

    return api_main()

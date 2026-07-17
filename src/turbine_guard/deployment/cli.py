"""Command-line interface for deployment bundle export and restore.

The thin ``scripts/deployment_bundle.py`` wrapper calls :func:`main`; all
logic lives in :mod:`turbine_guard.deployment` so it stays unit-testable.
"""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from turbine_guard.config.settings import get_settings
from turbine_guard.deployment.export import export_deployment_bundle
from turbine_guard.deployment.manifest import BundleError
from turbine_guard.deployment.restore import restore_deployment_bundle
from turbine_guard.logging_config import configure_logging

logger = logging.getLogger(__name__)

DEFAULT_ARCHIVE_NAME = "turbine-guard-deployment-bundle.tar.gz"


def build_parser(default_data_dir: Path) -> argparse.ArgumentParser:
    """Build the argument parser for the deployment bundle CLI."""
    parser = argparse.ArgumentParser(
        prog="deployment_bundle",
        description=(
            "Export the verified live champion as an immutable checksum-pinned "
            "deployment bundle, or restore the pinned bundle into the data directory."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    export_parser = subparsers.add_parser(
        "export",
        help="Package the loaded champion, manifests, and replay source into one archive.",
    )
    export_parser.add_argument(
        "--output",
        type=Path,
        default=default_data_dir / "deployment" / DEFAULT_ARCHIVE_NAME,
        help="Archive path to write (default: %(default)s)",
    )
    subparsers.add_parser(
        "restore",
        help=(
            "Download, verify, and restore the bundle configured through "
            "TURBINE_GUARD_DEPLOYMENT_BUNDLE_URL and _SHA256."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one bundle command; return a process exit code."""
    settings = get_settings()
    configure_logging(settings.log_level)
    args = build_parser(settings.data_dir).parse_args(argv)
    try:
        if args.command == "export":
            result = export_deployment_bundle(settings, args.output)
            logger.info(
                "deployment_bundle_export_result",
                extra={
                    "archive_path": str(result.archive_path),
                    "archive_sha256": result.archive_sha256,
                    "registry_version": result.manifest.registry_version,
                    "pin_hint": (
                        "Publish the archive, then set "
                        "TURBINE_GUARD_DEPLOYMENT_BUNDLE_URL to its revision-pinned URL "
                        f"and TURBINE_GUARD_DEPLOYMENT_BUNDLE_SHA256={result.archive_sha256}."
                    ),
                },
            )
        else:
            restored = restore_deployment_bundle(settings)
            logger.info(
                "deployment_bundle_restore_result",
                extra={
                    "status": restored.status.value,
                    "archive_sha256": restored.archive_sha256,
                    "registry_version": restored.manifest.registry_version,
                },
            )
    except (BundleError, RuntimeError) as exc:
        logger.error("deployment_bundle_failed", extra={"error": str(exc)})
        return 1
    return 0

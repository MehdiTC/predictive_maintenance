"""Command-line interface for dataset acquisition.

The thin ``scripts/download_data.py`` wrapper calls :func:`main`; all logic
lives in :mod:`turbine_guard.data.acquisition` so it stays unit-testable.
"""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from turbine_guard.config.settings import get_settings
from turbine_guard.data.acquisition import AcquisitionConfig, AcquisitionError, acquire
from turbine_guard.logging_config import configure_logging

logger = logging.getLogger(__name__)


def build_parser(default_url: str, default_data_dir: Path) -> argparse.ArgumentParser:
    """Build the argument parser for the acquisition CLI."""
    parser = argparse.ArgumentParser(
        prog="download_data",
        description=(
            "Download the NASA C-MAPSS FD001 subset into the immutable raw "
            "data layer and write a provenance manifest."
        ),
    )
    parser.add_argument(
        "--url",
        default=default_url,
        help=(
            "Source archive URL; https:// or file:// for a pre-downloaded "
            "archive (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir,
        help="Base data directory (default: %(default)s)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download the archive and replace existing raw files and manifest.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run acquisition from command-line arguments; return a process exit code."""
    settings = get_settings()
    configure_logging(settings.log_level)
    args = build_parser(settings.cmapss_source_url, settings.data_dir).parse_args(argv)
    config = AcquisitionConfig(
        data_dir=args.data_dir,
        source_url=args.url,
        force=args.force,
    )
    try:
        result = acquire(config)
    except AcquisitionError as exc:
        logger.error("acquisition_failed", extra={"error": str(exc)})
        return 1
    logger.info(
        "acquisition_result",
        extra={
            "status": result.status.value,
            "manifest_path": str(result.manifest_path),
        },
    )
    return 0

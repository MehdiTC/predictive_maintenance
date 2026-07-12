"""Command-line interface for raw-data processing.

The thin ``scripts/process_data.py`` wrapper calls :func:`main`; all logic
lives in :mod:`turbine_guard.data.processing` so it stays unit-testable.
"""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from turbine_guard.config.settings import get_settings
from turbine_guard.data.processing import ProcessingConfig, ProcessingError, process
from turbine_guard.logging_config import configure_logging

logger = logging.getLogger(__name__)


def build_parser(default_data_dir: Path) -> argparse.ArgumentParser:
    """Build the argument parser for the processing CLI."""
    parser = argparse.ArgumentParser(
        prog="process_data",
        description=(
            "Parse and validate the acquired NASA C-MAPSS FD001 raw files and "
            "write validated Parquet outputs plus a processing report."
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
        help="Rebuild the processed outputs even when they are current.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run processing from command-line arguments; return a process exit code."""
    settings = get_settings()
    configure_logging(settings.log_level)
    args = build_parser(settings.data_dir).parse_args(argv)
    config = ProcessingConfig(data_dir=args.data_dir, force=args.force)
    try:
        result = process(config)
    except ProcessingError as exc:
        logger.error("processing_failed", extra={"error": str(exc)})
        return 1
    report = result.report
    logger.info(
        "processing_result",
        extra={
            "status": result.status.value,
            "report_path": str(result.report_path),
            "output_paths": [str(path) for path in result.output_paths],
            "datasets": {
                validation.dataset: {
                    "rows": (
                        validation.trajectory_stats.row_count
                        if validation.trajectory_stats
                        else (validation.rul_stats.row_count if validation.rul_stats else None)
                    ),
                    "passed": validation.passed,
                }
                for validation in report.datasets
            },
            "warnings": list(report.warnings),
        },
    )
    return 0

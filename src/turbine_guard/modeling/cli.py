"""Command-line interface for the deterministic Loop 4 training pipeline."""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from turbine_guard.config.settings import get_settings
from turbine_guard.logging_config import configure_logging
from turbine_guard.modeling.config import (
    AlertConfig,
    TargetConfig,
    TrainingConfig,
)
from turbine_guard.modeling.pipeline import TrainingError, train_models

logger = logging.getLogger(__name__)


def build_parser(default_data_dir: Path) -> argparse.ArgumentParser:
    """Build the training CLI parser."""
    parser = argparse.ArgumentParser(
        prog="train_models",
        description=(
            "Verify Loop 3 artifacts, train bounded RUL candidates, select on validation, "
            "calibrate conformal intervals, and evaluate replay/official test data."
        ),
    )
    parser.add_argument("--data-dir", type=Path, default=default_data_dir)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rul-cap", type=int, default=125)
    parser.add_argument("--critical-horizon", type=int, default=30)
    parser.add_argument("--warning-horizon", type=int, default=50)
    parser.add_argument("--conformal-coverage", type=float, default=0.90)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Intentionally rebuild even when current artifacts verify.",
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> TrainingConfig:
    try:
        return TrainingConfig(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            random_seed=args.seed,
            targets=(
                TargetConfig("uncapped"),
                TargetConfig(f"capped_{args.rul_cap}", args.rul_cap),
            ),
            alerts=AlertConfig(
                critical_horizon=args.critical_horizon,
                warning_horizon=args.warning_horizon,
            ),
            conformal_coverage=args.conformal_coverage,
            force=args.force,
        )
    except ValueError as exc:
        raise TrainingError(f"Invalid training configuration: {exc}") from exc


def main(argv: Sequence[str] | None = None) -> int:
    """Run Loop 4 training and return a process exit code."""
    settings = get_settings()
    configure_logging(settings.log_level)
    args = build_parser(settings.data_dir).parse_args(argv)
    try:
        result = train_models(_config_from_args(args))
    except TrainingError as exc:
        logger.error("model_training_failed", extra={"error": str(exc)})
        return 1
    logger.info(
        "model_training_result",
        extra={"status": result.status.value, **result.summary},
    )
    return 0

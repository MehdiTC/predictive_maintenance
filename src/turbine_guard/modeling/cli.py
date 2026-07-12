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
    parser.add_argument(
        "--track-with-mlflow",
        action="store_true",
        help="Track the verified Loop 4 execution and optionally register its champion.",
    )
    parser.add_argument(
        "--force-mlflow-run",
        action="store_true",
        help="Create a new parent/child run set even when this execution was already logged.",
    )
    parser.add_argument(
        "--force-new-model-version",
        action="store_true",
        help="Register a new version even when the identical champion checksum already exists.",
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
        if (args.force_mlflow_run or args.force_new_model_version) and not args.track_with_mlflow:
            raise ValueError("MLflow force flags require --track-with-mlflow.")
        training_config = _config_from_args(args)
        result = train_models(training_config)
        tracking_summary: dict[str, object] = {"mlflow_tracking": "disabled"}
        if args.track_with_mlflow:
            from turbine_guard.tracking.config import MlflowConfig
            from turbine_guard.tracking.mlflow_tracker import MlflowTracker

            mlflow_config = MlflowConfig.from_settings(
                settings,
                force_new_run=args.force_mlflow_run,
                force_new_model_version=args.force_new_model_version,
            )
            tracked = MlflowTracker(mlflow_config).track(training_config)
            tracking_summary = {
                "mlflow_tracking": tracked.status.value,
                "mlflow_parent_run_id": tracked.parent_run_id,
                "mlflow_candidate_runs": len(tracked.candidate_run_ids),
                "registered_model": tracked.registered_model_name,
                "registered_version": tracked.registered_version,
                "registry_aliases": tracked.aliases,
                "max_prediction_difference": tracked.max_prediction_difference,
            }
    except (RuntimeError, ValueError) as exc:
        logger.error("model_training_failed", extra={"error": str(exc)})
        return 1
    logger.info(
        "model_training_result",
        extra={"status": result.status.value, **result.summary, **tracking_summary},
    )
    return 0

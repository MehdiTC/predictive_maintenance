"""Command-line interface for building labels, splits, and features.

The thin ``scripts/build_features.py`` wrapper calls :func:`main`; all logic
lives in :mod:`turbine_guard.features` so it stays unit-testable.
"""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from turbine_guard.config.settings import get_settings
from turbine_guard.features.config import BuildConfig, FeatureConfig, RulConfig, SplitConfig
from turbine_guard.features.pipeline import (
    FeatureBuildConfig,
    FeatureBuildError,
    build_features,
)
from turbine_guard.logging_config import configure_logging

logger = logging.getLogger(__name__)


def build_parser(default_data_dir: Path) -> argparse.ArgumentParser:
    """Build the argument parser for the feature-build CLI."""
    parser = argparse.ArgumentParser(
        prog="build_features",
        description=(
            "Generate RUL labels, deterministic asset-level splits, and "
            "leakage-safe features from the validated FD001 Parquet outputs, "
            "plus split and feature manifests."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir,
        help="Base data directory (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SplitConfig().seed,
        help="Random seed for the asset-level split (default: %(default)s)",
    )
    parser.add_argument(
        "--rul-cap",
        type=int,
        default=None,
        help="Optional RUL cap; when set, a rul_capped target is also generated.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild feature outputs even when they are current.",
    )
    return parser


def _build_config(args: argparse.Namespace) -> FeatureBuildConfig:
    """Assemble a typed build configuration from parsed CLI arguments."""
    try:
        build = BuildConfig(
            features=FeatureConfig(),
            split=SplitConfig(seed=args.seed),
            rul=RulConfig(cap=args.rul_cap),
        )
    except ValueError as exc:
        raise FeatureBuildError(f"Invalid configuration: {exc}") from exc
    return FeatureBuildConfig(data_dir=args.data_dir, force=args.force, build=build)


def main(argv: Sequence[str] | None = None) -> int:
    """Run a feature build from CLI arguments; return a process exit code."""
    settings = get_settings()
    configure_logging(settings.log_level)
    args = build_parser(settings.data_dir).parse_args(argv)
    try:
        config = _build_config(args)
        result = build_features(config)
    except FeatureBuildError as exc:
        logger.error("feature_build_failed", extra={"error": str(exc)})
        return 1

    manifest = result.feature_manifest
    logger.info(
        "feature_build_result",
        extra={
            "status": result.status.value,
            "feature_manifest_path": str(result.feature_manifest_path),
            "feature_count": len(manifest.feature_columns),
            "seed": manifest.seed,
            "rul_cap": manifest.rul_cap,
            "asset_counts": result.split_manifest.asset_counts,
            "row_counts": result.split_manifest.row_counts,
            "outputs": [record.filename for record in manifest.outputs],
        },
    )
    return 0

"""Explicit, idempotent container bootstrap for data, models, and MLflow."""

import argparse
import hashlib
import io
import json
import logging
import sys
import zipfile
from collections.abc import Sequence
from pathlib import Path

from turbine_guard.config.settings import Settings, get_settings
from turbine_guard.data.acquisition import AcquisitionConfig, acquire
from turbine_guard.data.processing import ProcessingConfig, process
from turbine_guard.data.schema import SENSOR_COLUMNS
from turbine_guard.features.pipeline import FeatureBuildConfig, build_features
from turbine_guard.logging_config import configure_logging
from turbine_guard.modeling.config import (
    CandidateConfig,
    ModelKind,
    SelectionConfig,
    TargetConfig,
    TrainingConfig,
)
from turbine_guard.modeling.pipeline import train_models
from turbine_guard.tracking.config import MlflowConfig
from turbine_guard.tracking.mlflow_tracker import MlflowTracker

logger = logging.getLogger(__name__)
CI_FIXTURE_NAME = "ci_fd001.zip"


def main(argv: Sequence[str] | None = None) -> int:
    """Run explicit bootstrap; ``--ci-fixture`` selects the offline miniature dataset."""
    parser = argparse.ArgumentParser(
        prog="bootstrap",
        description=(
            "Idempotently acquire, process, feature-build, train, track, register, and "
            "promote the initial champion. This is explicit and never runs at API startup."
        ),
    )
    parser.add_argument(
        "--ci-fixture",
        action="store_true",
        help="Use a deterministic miniature FD001 fixture and bounded smoke candidates.",
    )
    args = parser.parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level)
    try:
        result = bootstrap(settings, ci_fixture=args.ci_fixture)
    except (OSError, RuntimeError, ValueError) as exc:
        logger.error("bootstrap_failed", extra={"error": str(exc)})
        return 1
    sys.stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


def bootstrap(settings: Settings, *, ci_fixture: bool = False) -> dict[str, object]:
    """Build or verify every prerequisite and return the registered champion summary."""
    source_url = settings.cmapss_source_url
    if ci_fixture:
        source_url = _ensure_ci_fixture(settings.data_dir).as_uri()

    acquisition = acquire(AcquisitionConfig(data_dir=settings.data_dir, source_url=source_url))
    processing = process(
        ProcessingConfig(data_dir=settings.data_dir, validate_canonical=not ci_fixture)
    )
    features = build_features(FeatureBuildConfig(data_dir=settings.data_dir))
    training_config = _training_config(settings.data_dir, ci_fixture=ci_fixture)
    training = train_models(training_config)
    tracked = MlflowTracker(MlflowConfig.from_settings(settings)).track(training_config)
    champion = tracked.aliases.get(settings.mlflow_champion_alias)
    if champion is None:
        raise RuntimeError("Bootstrap completed without assigning the configured champion alias.")
    return {
        "fixture": "ci" if ci_fixture else "nasa_fd001",
        "acquisition": acquisition.status.value,
        "processing": processing.status.value,
        "features": features.status.value,
        "training": training.status.value,
        "mlflow": tracked.status.value,
        "registered_model": tracked.registered_model_name,
        "champion_version": champion,
    }


def _training_config(data_dir: Path, *, ci_fixture: bool) -> TrainingConfig:
    if not ci_fixture:
        return TrainingConfig(data_dir=data_dir)
    return TrainingConfig(
        data_dir=data_dir,
        targets=(TargetConfig("uncapped"), TargetConfig("capped_125", 125)),
        candidates=(
            CandidateConfig("constant", ModelKind.CONSTANT),
            CandidateConfig("ridge", ModelKind.RIDGE, (("alpha", 1.0),), 1),
            CandidateConfig(
                "tree",
                ModelKind.HIST_GRADIENT_BOOSTING,
                (("max_iter", 5), ("max_leaf_nodes", 7), ("learning_rate", 0.1)),
                2,
            ),
            CandidateConfig(
                "xgb",
                ModelKind.XGBOOST,
                (("n_estimators", 5), ("max_depth", 2), ("learning_rate", 0.1)),
                3,
            ),
        ),
        selection=SelectionConfig(
            minimum_critical_recall=0.0,
            maximum_false_alarms_per_1000_cycles=1000.0,
            relative_rmse_tolerance=0.0,
        ),
        conformal_coverage=0.8,
        latency_repeats=1,
    )


def _ensure_ci_fixture(data_dir: Path) -> Path:
    path = data_dir / "bootstrap" / CI_FIXTURE_NAME
    expected = _ci_fixture_bytes()
    if path.exists():
        actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        expected_sha = hashlib.sha256(expected).hexdigest()
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"Existing CI bootstrap fixture {path} is not the deterministic fixture."
            )
        return path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(expected)
    temporary.replace(path)
    return path.resolve()


def _ci_fixture_bytes() -> bytes:
    train_lengths = {asset: 20 + asset for asset in range(1, 21)}
    test_lengths = {asset: 10 + asset for asset in range(1, 21)}
    contents = {
        "train_FD001.txt": _trajectory_text(train_lengths),
        "test_FD001.txt": _trajectory_text(test_lengths),
        "RUL_FD001.txt": "".join(f"{40 + asset}\n" for asset in range(1, 21)),
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for filename, content in contents.items():
            info = zipfile.ZipInfo(filename, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, content)
    return buffer.getvalue()


def _trajectory_text(lengths: dict[int, int]) -> str:
    return "".join(
        _trajectory_line(asset_id, cycle)
        for asset_id, length in lengths.items()
        for cycle in range(1, length + 1)
    )


def _trajectory_line(asset_id: int, cycle: int) -> str:
    settings = (f"{0.001 * (asset_id + cycle):.4f}", f"{-0.0002 * cycle:.4f}", "100.0")
    sensors = []
    for index in range(1, len(SENSOR_COLUMNS) + 1):
        if index in (1, 5):
            sensors.append("518.67")
        elif index == 6:
            sensors.append("21.61" if cycle % 2 else "21.6101")
        else:
            sensors.append(f"{100 + index + 0.5 * cycle + 0.1 * asset_id:.2f}")
    return " ".join((str(asset_id), str(cycle), *settings, *sensors)) + "  \n"

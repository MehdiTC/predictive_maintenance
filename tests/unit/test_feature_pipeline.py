"""Tests for the feature-build pipeline: outputs, manifests, idempotency."""

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from turbine_guard.features.builder import FeatureBuilder
from turbine_guard.features.config import BuildConfig, RulConfig, SplitConfig
from turbine_guard.features.pipeline import (
    BuildStatus,
    FeatureBuildConfig,
    FeatureBuildError,
    build_features,
)
from turbine_guard.features.splits import PARTITION_NAMES


def build(data_dir: Path, **overrides: object) -> FeatureBuildConfig:
    return FeatureBuildConfig(data_dir=data_dir, **overrides)


def test_build_writes_all_outputs_and_manifests(processed_data_dir: Path) -> None:
    result = build_features(build(processed_data_dir))

    assert result.status is BuildStatus.BUILT
    features_dir = processed_data_dir / "features" / "cmapss" / "FD001"
    for name in ("train", "validation", "calibration", "replay", "test_features", "test_labels"):
        assert (features_dir / f"{name}.parquet").exists()
    assert (features_dir / "split_manifest.json").exists()
    assert (features_dir / "feature_manifest.json").exists()


def test_split_manifest_counts_are_asset_level(processed_data_dir: Path) -> None:
    result = build_features(build(processed_data_dir))
    manifest = result.split_manifest

    assert manifest.asset_counts == {
        "train": 14,
        "validation": 3,
        "calibration": 1,
        "replay": 2,
    }
    assert sum(manifest.asset_counts.values()) == 20
    all_assets = [a for name in PARTITION_NAMES for a in manifest.partitions[name]]
    assert len(all_assets) == len(set(all_assets)) == 20


def test_feature_manifest_describes_outputs_and_columns(processed_data_dir: Path) -> None:
    result = build_features(build(processed_data_dir))
    manifest = result.feature_manifest

    assert manifest.identifier_columns == ("asset_id", "cycle")
    assert manifest.target_columns == ("rul",)
    assert manifest.metadata_columns == ("split",)
    assert manifest.feature_columns == FeatureBuilder().feature_columns()
    assert manifest.imputation is None
    assert manifest.rul_cap is None
    # Every output checksum matches the file on disk.
    for record in manifest.outputs:
        path = processed_data_dir / "features" / "cmapss" / "FD001" / record.filename
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record.sha256


def test_model_ready_column_order(processed_data_dir: Path) -> None:
    build_features(build(processed_data_dir))
    features_dir = processed_data_dir / "features" / "cmapss" / "FD001"

    train = pd.read_parquet(features_dir / "train.parquet")
    expected = ["asset_id", "cycle", "split", "rul", *FeatureBuilder().feature_columns()]
    assert list(train.columns) == expected

    test = pd.read_parquet(features_dir / "test_features.parquet")
    assert "rul" not in test.columns  # test carries no per-row labels
    assert list(test.columns) == ["asset_id", "cycle", "split", *FeatureBuilder().feature_columns()]


def test_capped_target_adds_column(processed_data_dir: Path) -> None:
    config = build(processed_data_dir, build=BuildConfig(rul=RulConfig(cap=20)))
    result = build_features(config)

    assert result.feature_manifest.target_columns == ("rul", "rul_capped")
    train = pd.read_parquet(processed_data_dir / "features" / "cmapss" / "FD001" / "train.parquet")
    assert "rul_capped" in train.columns
    assert train["rul_capped"].max() <= 20


def test_parquet_round_trip(processed_data_dir: Path) -> None:
    result = build_features(build(processed_data_dir))
    for path in result.output_paths:
        frame = pd.read_parquet(path)
        assert len(frame) > 0


def test_rerun_is_idempotent(processed_data_dir: Path) -> None:
    first = build_features(build(processed_data_dir))
    mtimes = {path: path.stat().st_mtime_ns for path in first.output_paths}

    second = build_features(build(processed_data_dir))

    assert second.status is BuildStatus.ALREADY_BUILT
    assert {path: path.stat().st_mtime_ns for path in second.output_paths} == mtimes


def test_changed_seed_triggers_rebuild(processed_data_dir: Path) -> None:
    build_features(build(processed_data_dir))
    result = build_features(
        build(processed_data_dir, build=BuildConfig(split=SplitConfig(seed=99)))
    )
    assert result.status is BuildStatus.BUILT


def test_tampered_output_detected(processed_data_dir: Path) -> None:
    first = build_features(build(processed_data_dir))
    victim = first.output_paths[0]
    victim.write_bytes(victim.read_bytes() + b"tampered")

    with pytest.raises(FeatureBuildError, match="Checksum mismatch"):
        build_features(build(processed_data_dir))

    recovered = build_features(build(processed_data_dir, force=True))
    assert recovered.status is BuildStatus.BUILT


def test_missing_output_detected(processed_data_dir: Path) -> None:
    first = build_features(build(processed_data_dir))
    first.output_paths[1].unlink()

    with pytest.raises(FeatureBuildError, match="missing"):
        build_features(build(processed_data_dir))


def test_force_rebuilds(processed_data_dir: Path) -> None:
    build_features(build(processed_data_dir))
    result = build_features(build(processed_data_dir, force=True))
    assert result.status is BuildStatus.BUILT


def test_missing_processing_report_rejected(tmp_path: Path) -> None:
    with pytest.raises(FeatureBuildError, match="processing report"):
        build_features(build(tmp_path / "data"))


def test_tampered_loop2_output_detected(processed_data_dir: Path) -> None:
    train_parquet = processed_data_dir / "processed" / "cmapss" / "FD001" / "train_FD001.parquet"
    train_parquet.write_bytes(train_parquet.read_bytes() + b"corrupt")

    with pytest.raises(FeatureBuildError, match="does not match the processing report"):
        build_features(build(processed_data_dir))


def test_inputs_unchanged_by_build(processed_data_dir: Path) -> None:
    processed_dir = processed_data_dir / "processed" / "cmapss" / "FD001"
    raw_dir = processed_data_dir / "raw" / "cmapss" / "FD001"
    before = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [*processed_dir.glob("*.parquet"), *raw_dir.iterdir()]
    }

    build_features(build(processed_data_dir))

    after = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [*processed_dir.glob("*.parquet"), *raw_dir.iterdir()]
    }
    assert after == before


def test_null_counts_recorded_for_early_cycles(processed_data_dir: Path) -> None:
    result = build_features(build(processed_data_dir))
    train_record = next(r for r in result.feature_manifest.outputs if r.filename == "train.parquet")
    # 14 train assets each contribute nulls at cycle 1 (delta + 3 rolling-std windows).
    assert train_record.null_count > 0
    labels_record = next(
        r for r in result.feature_manifest.outputs if r.filename == "test_labels.parquet"
    )
    assert labels_record.null_count == 0

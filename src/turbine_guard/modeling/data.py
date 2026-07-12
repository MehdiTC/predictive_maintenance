"""Verification and loading of the Loop 3 model-ready feature contract."""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
import pandas as pd

from turbine_guard.data.schema import ASSET_ID_COLUMN, CYCLE_COLUMN
from turbine_guard.features.config import RUL_COLUMN, SPLIT_COLUMN
from turbine_guard.features.labels import FINAL_CYCLE_COLUMN
from turbine_guard.features.manifest import (
    FeatureManifest,
    SplitManifest,
    load_feature_manifest,
    load_split_manifest,
    sha256_of,
)
from turbine_guard.features.splits import PARTITION_NAMES
from turbine_guard.modeling.config import TrainingConfig


class DatasetRole(StrEnum):
    """Explicit purpose of each dataset; roles are never inferred from use."""

    TRAIN = "train"
    VALIDATION = "validation"
    CALIBRATION = "calibration"
    REPLAY = "replay"
    OFFICIAL_TEST = "official_test"


class ModelDataError(RuntimeError):
    """Raised when Loop 3 artifacts cannot safely be used for modeling."""


@dataclass(frozen=True)
class VerifiedModelData:
    """Verified role frames and the exact feature/provenance contracts."""

    frames: dict[DatasetRole, pd.DataFrame]
    official_labels: pd.DataFrame
    feature_columns: tuple[str, ...]
    feature_manifest: FeatureManifest
    split_manifest: SplitManifest
    feature_manifest_sha256: str
    split_manifest_sha256: str
    input_checksums: dict[str, str]

    def frame(self, role: DatasetRole) -> pd.DataFrame:
        """Return the frame assigned to an explicit role."""
        return self.frames[role]


def load_verified_model_data(config: TrainingConfig) -> VerifiedModelData:
    """Verify checksums, schemas, feature order, and asset isolation, then load data."""
    base = config.features_dir
    feature_manifest_path = base / "feature_manifest.json"
    split_manifest_path = base / "split_manifest.json"
    if not feature_manifest_path.exists() or not split_manifest_path.exists():
        raise ModelDataError(
            f"Loop 3 manifests are missing under {base}. Run feature generation first "
            "(uv run python scripts/build_features.py)."
        )
    try:
        feature_manifest = load_feature_manifest(feature_manifest_path)
        split_manifest = load_split_manifest(split_manifest_path)
    except (OSError, ValueError) as exc:
        raise ModelDataError(f"Loop 3 manifests could not be read: {exc}") from exc

    if feature_manifest.dataset_subset != config.subset:
        raise ModelDataError(
            f"Feature manifest subset is {feature_manifest.dataset_subset}, "
            f"expected {config.subset}."
        )
    split_sha = sha256_of(split_manifest_path)
    if feature_manifest.split_manifest_sha256 != split_sha:
        raise ModelDataError("Split manifest checksum does not match the feature manifest.")
    if feature_manifest.imputation is not None:
        raise ModelDataError(
            "Loop 3 feature manifest unexpectedly records fitted imputation; Loop 4 expects "
            "structural nulls and fits preprocessing on training rows itself."
        )

    _verify_output_checksums(base, feature_manifest)
    frames = {
        DatasetRole(role): pd.read_parquet(base / f"{role}.parquet") for role in PARTITION_NAMES
    }
    official_features = pd.read_parquet(base / "test_features.parquet")
    official_labels = pd.read_parquet(base / "test_labels.parquet")
    frames[DatasetRole.OFFICIAL_TEST] = official_features

    feature_columns = tuple(feature_manifest.feature_columns)
    if not feature_columns or len(feature_columns) != len(set(feature_columns)):
        raise ModelDataError("Feature manifest has an empty or duplicate ordered feature list.")
    _verify_role_frames(frames, feature_manifest, split_manifest, feature_columns)
    _verify_official_labels(official_features, official_labels)
    _verify_asset_isolation(frames, split_manifest)

    checksums = {record.filename: record.sha256 for record in feature_manifest.outputs} | {
        "feature_manifest.json": sha256_of(feature_manifest_path),
        "split_manifest.json": split_sha,
    }
    return VerifiedModelData(
        frames=frames,
        official_labels=official_labels,
        feature_columns=feature_columns,
        feature_manifest=feature_manifest,
        split_manifest=split_manifest,
        feature_manifest_sha256=sha256_of(feature_manifest_path),
        split_manifest_sha256=split_sha,
        input_checksums=checksums,
    )


def official_final_rows(data: VerifiedModelData) -> pd.DataFrame:
    """Return exactly one final observed feature row per official test asset."""
    frame = data.frame(DatasetRole.OFFICIAL_TEST)
    final_rows = (
        frame.sort_values([ASSET_ID_COLUMN, CYCLE_COLUMN], kind="stable")
        .groupby(ASSET_ID_COLUMN, as_index=False)
        .tail(1)
    )
    merged = final_rows.merge(
        data.official_labels,
        left_on=[ASSET_ID_COLUMN, CYCLE_COLUMN],
        right_on=[ASSET_ID_COLUMN, FINAL_CYCLE_COLUMN],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(data.official_labels):
        raise ModelDataError("Official final-row features do not match every benchmark label.")
    return merged.sort_values(ASSET_ID_COLUMN, kind="stable").reset_index(drop=True)


def model_matrix(frame: pd.DataFrame, feature_columns: tuple[str, ...]) -> pd.DataFrame:
    """Extract the exact ordered feature matrix and reject unsupported values."""
    if tuple(column for column in frame.columns if column in feature_columns) != feature_columns:
        raise ModelDataError("Frame feature order does not match the manifest.")
    matrix = frame.loc[:, list(feature_columns)]
    values = matrix.to_numpy(dtype="float64")
    if bool(np.isinf(values).any()):
        raise ModelDataError("Model features contain infinite values.")
    return matrix


def target_values(frame: pd.DataFrame, cap: int | None) -> pd.Series:
    """Return the explicit uncapped or derived capped target for one experiment."""
    if RUL_COLUMN not in frame.columns:
        raise ModelDataError(f"Frame does not contain the required '{RUL_COLUMN}' target.")
    values = frame[RUL_COLUMN].astype("float64")
    if bool(values.isna().any()) or bool((values < 0).any()):
        raise ModelDataError("RUL target must be finite and non-negative.")
    return values if cap is None else values.clip(upper=cap)


def _verify_output_checksums(base: Path, manifest: FeatureManifest) -> None:
    for record in manifest.outputs:
        path = base / record.filename
        if not path.exists():
            raise ModelDataError(f"Loop 3 output {path} is missing.")
        actual = sha256_of(path)
        if actual != record.sha256:
            raise ModelDataError(
                f"Loop 3 output checksum mismatch for {path}: expected {record.sha256}, "
                f"found {actual}."
            )


def _verify_role_frames(
    frames: dict[DatasetRole, pd.DataFrame],
    manifest: FeatureManifest,
    split_manifest: SplitManifest,
    feature_columns: tuple[str, ...],
) -> None:
    ids = tuple(manifest.identifier_columns)
    metadata = tuple(manifest.metadata_columns)
    targets = tuple(manifest.target_columns)
    forbidden = set(ids) | set(metadata) | set(targets)
    overlap = forbidden & set(feature_columns)
    if overlap:
        raise ModelDataError(f"Targets, identifiers, or metadata entered feature list: {overlap}.")

    expected_train = (*ids, *metadata, *targets, *feature_columns)
    for role in (
        DatasetRole.TRAIN,
        DatasetRole.VALIDATION,
        DatasetRole.CALIBRATION,
        DatasetRole.REPLAY,
    ):
        frame = frames[role]
        if tuple(frame.columns) != expected_train:
            raise ModelDataError(f"{role.value} schema or feature order does not match manifest.")
        if set(frame[SPLIT_COLUMN].unique()) != {role.value}:
            raise ModelDataError(f"{role.value} frame carries an incorrect split label.")
        if len(frame) != split_manifest.row_counts[role.value]:
            raise ModelDataError(f"{role.value} row count does not match split manifest.")
        model_matrix(frame, feature_columns)

    official = frames[DatasetRole.OFFICIAL_TEST]
    expected_test = (*ids, *metadata, *feature_columns)
    if tuple(official.columns) != expected_test:
        raise ModelDataError("Official-test feature schema or order does not match manifest.")
    if set(official[SPLIT_COLUMN].unique()) != {"test"}:
        raise ModelDataError("Official-test frame carries an incorrect split label.")
    model_matrix(official, feature_columns)


def _verify_asset_isolation(
    frames: dict[DatasetRole, pd.DataFrame], split_manifest: SplitManifest
) -> None:
    seen: set[int] = set()
    for role in (
        DatasetRole.TRAIN,
        DatasetRole.VALIDATION,
        DatasetRole.CALIBRATION,
        DatasetRole.REPLAY,
    ):
        assets = {int(asset) for asset in frames[role][ASSET_ID_COLUMN].unique()}
        expected = set(split_manifest.partitions[role.value])
        if assets != expected:
            raise ModelDataError(f"{role.value} asset IDs do not match the split manifest.")
        overlap = seen & assets
        if overlap:
            raise ModelDataError(f"Assets appear in multiple internal roles: {sorted(overlap)}.")
        seen |= assets


def _verify_official_labels(features: pd.DataFrame, labels: pd.DataFrame) -> None:
    expected = (ASSET_ID_COLUMN, FINAL_CYCLE_COLUMN, RUL_COLUMN)
    if tuple(labels.columns) != expected:
        raise ModelDataError("Official-test label schema is incompatible.")
    if labels[ASSET_ID_COLUMN].duplicated().any():
        raise ModelDataError("Official-test labels contain duplicate assets.")
    if set(labels[ASSET_ID_COLUMN]) != set(features[ASSET_ID_COLUMN]):
        raise ModelDataError("Official-test feature and label asset IDs differ.")
    if bool(labels[RUL_COLUMN].isna().any()) or bool((labels[RUL_COLUMN] < 0).any()):
        raise ModelDataError("Official-test labels must be finite and non-negative.")

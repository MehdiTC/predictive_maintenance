"""Leakage-safe asset assignment and Loop 4 candidate retraining."""

import hashlib
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any

import pandas as pd

from turbine_guard.features.manifest import load_feature_manifest, sha256_of
from turbine_guard.monitoring.data import LabeledAssetData


@dataclass(frozen=True)
class RetrainingSplit:
    additions: tuple[LabeledAssetData, ...]
    holdout: tuple[LabeledAssetData, ...]

    @property
    def addition_rows(self) -> int:
        return sum(asset.row_count for asset in self.additions)

    @property
    def holdout_rows(self) -> int:
        return sum(asset.row_count for asset in self.holdout)

    def record(self) -> dict[str, Any]:
        return {
            "strategy": "sha256_asset_level",
            "retraining_addition_asset_ids": [str(asset.asset_id) for asset in self.additions],
            "promotion_holdout_asset_ids": [str(asset.asset_id) for asset in self.holdout],
            "retraining_addition_rows": self.addition_rows,
            "promotion_holdout_rows": self.holdout_rows,
        }


def split_labeled_assets(
    assets: list[LabeledAssetData],
    *,
    holdout_fraction: float,
    minimum_holdout_assets: int,
    seed: int,
) -> RetrainingSplit:
    """Assign every new asset to exactly one fit or promotion role deterministically."""
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("Holdout fraction must be strictly between zero and one.")
    if minimum_holdout_assets < 1:
        raise ValueError("At least one promotion holdout asset is required.")
    holdout_count = max(minimum_holdout_assets, ceil(len(assets) * holdout_fraction))
    if len(assets) - holdout_count < 1:
        raise ValueError("Too few labeled assets for disjoint fitting and promotion holdout.")
    ordered = sorted(
        assets,
        key=lambda asset: hashlib.sha256(f"{seed}:{asset.asset_id}".encode()).hexdigest(),
    )
    additions = tuple(ordered[:-holdout_count])
    holdout = tuple(ordered[-holdout_count:])
    if {asset.asset_id for asset in additions} & {asset.asset_id for asset in holdout}:
        raise AssertionError("Retraining and promotion asset roles overlap.")
    return RetrainingSplit(additions, holdout)


def assemble_training_frame(
    original_training: pd.DataFrame,
    additions: tuple[LabeledAssetData, ...],
    *,
    feature_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Append labeled operational fit assets to the immutable original training base."""
    required = {"asset_id", "cycle", "rul", *feature_columns}
    if not required <= set(original_training.columns):
        raise ValueError("Original training frame is incompatible with the feature contract.")
    frames = [original_training.loc[:, ["asset_id", "cycle", "rul", *feature_columns]].copy()]
    next_asset_id = int(original_training["asset_id"].max()) + 1
    for offset, asset in enumerate(additions):
        frame = asset.frame.loc[:, ["asset_id", "cycle", "rul", *feature_columns]].copy()
        frame["asset_id"] = next_asset_id + offset
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    if len(combined) != len(original_training) + sum(asset.row_count for asset in additions):
        raise AssertionError("Retraining row assembly lost or duplicated records.")
    return combined


def assemble_holdout_frame(
    holdout: tuple[LabeledAssetData, ...], *, feature_columns: tuple[str, ...]
) -> pd.DataFrame:
    """Assemble the promotion-only frame with stable integer evaluation asset IDs."""
    frames: list[pd.DataFrame] = []
    for evaluation_id, asset in enumerate(holdout, start=1):
        frame = asset.frame.loc[:, ["asset_id", "cycle", "rul", *feature_columns]].copy()
        frame["asset_id"] = evaluation_id
        frames.append(frame)
    if not frames:
        raise ValueError("Promotion evaluation requires at least one holdout asset.")
    return pd.concat(frames, ignore_index=True)


def load_original_training_frame(
    data_dir: Path, *, expected_feature_columns: tuple[str, ...]
) -> pd.DataFrame:
    """Load only the verified Loop 3 training role; protected roles are never opened."""
    base = data_dir / "features" / "cmapss" / "FD001"
    manifest = load_feature_manifest(base / "feature_manifest.json")
    if tuple(manifest.feature_columns) != expected_feature_columns:
        raise ValueError("Original training feature contract differs from the champion.")
    record = next((item for item in manifest.outputs if item.filename == "train.parquet"), None)
    if record is None:
        raise ValueError("Feature manifest has no original training output.")
    path = base / record.filename
    if sha256_of(path) != record.sha256:
        raise ValueError("Original training output failed checksum verification.")
    return pd.read_parquet(path)

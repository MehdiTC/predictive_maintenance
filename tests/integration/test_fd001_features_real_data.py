"""Local integration test building features from the real FD001 dataset.

Skipped automatically when the dataset has not been acquired and processed (for
example in CI), so the normal suite never needs internet access or the real
download. The processed layer is copied into a temporary directory first, so
the repository's own ``data/`` directory is never modified by tests.
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from turbine_guard.features.builder import FeatureBuilder, IncrementalFeatureState
from turbine_guard.features.config import FeatureConfig
from turbine_guard.features.pipeline import (
    BuildStatus,
    FeatureBuildConfig,
    build_features,
)

REPO_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
REPORT_PATH = REPO_DATA_DIR / "processed" / "cmapss" / "FD001" / "processing_report.json"

pytestmark = [
    pytest.mark.real_data,
    pytest.mark.skipif(
        not REPORT_PATH.exists(),
        reason="FD001 not processed locally (run: make acquire && make process)",
    ),
]


@pytest.fixture(scope="module")
def real_data_copy(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Copy of the processed layer, isolated from the repository."""
    data_dir = tmp_path_factory.mktemp("fd001_features") / "data"
    shutil.copytree(
        REPO_DATA_DIR / "processed" / "cmapss" / "FD001",
        data_dir / "processed" / "cmapss" / "FD001",
    )
    return data_dir


def test_real_fd001_feature_build(real_data_copy: Path) -> None:
    result = build_features(FeatureBuildConfig(data_dir=real_data_copy))

    assert result.status is BuildStatus.BUILT
    assert result.split_manifest.asset_counts == {
        "train": 70,
        "validation": 15,
        "calibration": 5,
        "replay": 10,
    }
    assert sum(result.split_manifest.row_counts.values()) == 20_631  # all train rows partitioned

    manifest = result.feature_manifest
    assert len(manifest.feature_columns) == 552
    by_name = {record.filename: record for record in manifest.outputs}
    assert by_name["test_features.parquet"].record_count == 13_096
    assert by_name["test_labels.parquet"].record_count == 100
    assert not by_name["test_features.parquet"].has_targets

    # Idempotent rerun.
    assert build_features(FeatureBuildConfig(data_dir=real_data_copy)).status is (
        BuildStatus.ALREADY_BUILT
    )


def test_real_asset_offline_matches_incremental(real_data_copy: Path) -> None:
    train = pd.read_parquet(
        real_data_copy / "processed" / "cmapss" / "FD001" / "train_FD001.parquet"
    )
    builder = FeatureBuilder(FeatureConfig())
    asset_id = int(train["asset_id"].min())
    asset = train[train["asset_id"] == asset_id].sort_values("cycle")

    offline = builder.transform(asset).reset_index(drop=True)
    state = IncrementalFeatureState(builder, asset_id=asset_id)
    rows = [
        {key: value for key, value in record.items() if key != "asset_id"}
        for record in asset.to_dict(orient="records")
    ]
    incremental = pd.DataFrame([state.update(observation) for observation in rows])

    cols = list(builder.feature_columns())
    np.testing.assert_allclose(
        incremental[cols].to_numpy(dtype="float64"),
        offline[cols].to_numpy(dtype="float64"),
        equal_nan=True,
    )


def test_real_future_row_mutation_leakage(real_data_copy: Path) -> None:
    train = pd.read_parquet(
        real_data_copy / "processed" / "cmapss" / "FD001" / "train_FD001.parquet"
    )
    builder = FeatureBuilder(FeatureConfig())
    asset_id = int(train["asset_id"].min())
    asset = train[train["asset_id"] == asset_id].sort_values("cycle").reset_index(drop=True)
    cut = 50

    original = builder.transform(asset)
    mutated = asset.copy()
    mutated.loc[mutated["cycle"] > cut, "sensor_04"] += 500.0
    after = builder.transform(mutated)

    cols = list(builder.feature_columns())
    before_rows = original[original["cycle"] <= cut][cols].to_numpy()
    after_rows = after[after["cycle"] <= cut][cols].to_numpy()
    assert np.array_equal(before_rows, after_rows, equal_nan=True)

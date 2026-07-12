"""Local integration test against the actually acquired FD001 dataset.

Skipped automatically when the dataset has not been acquired (for example in
CI), so the normal suite never needs internet access or the real download.
The acquired raw layer is copied into a temporary directory first, so the
repository's own ``data/`` directory is never modified by tests.
"""

import shutil
from pathlib import Path

import pandas as pd
import pytest

from turbine_guard.data.processing import ProcessingConfig, ProcessingStatus, process
from turbine_guard.data.schema import TRAJECTORY_COLUMNS

REPO_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
MANIFEST_PATH = REPO_DATA_DIR / "manifests" / "cmapss_fd001.json"

pytestmark = [
    pytest.mark.real_data,
    pytest.mark.skipif(
        not MANIFEST_PATH.exists(),
        reason="FD001 dataset not acquired locally (run: make acquire)",
    ),
]


@pytest.fixture(scope="module")
def real_data_copy(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Copy of the acquired raw layer, isolated from the repository."""
    data_dir = tmp_path_factory.mktemp("fd001") / "data"
    shutil.copytree(REPO_DATA_DIR / "manifests", data_dir / "manifests")
    shutil.copytree(
        REPO_DATA_DIR / "raw" / "cmapss" / "FD001",
        data_dir / "raw" / "cmapss" / "FD001",
    )
    return data_dir


def test_real_fd001_processes_with_canonical_profile(real_data_copy: Path) -> None:
    result = process(ProcessingConfig(data_dir=real_data_copy))

    assert result.status is ProcessingStatus.PROCESSED
    assert result.report.passed

    by_name = {validation.dataset: validation for validation in result.report.datasets}
    train_stats = by_name["train"].trajectory_stats
    test_stats = by_name["test"].trajectory_stats
    rul_stats = by_name["rul"].rul_stats
    assert train_stats is not None
    assert test_stats is not None
    assert rul_stats is not None
    assert train_stats.row_count == 20_631
    assert train_stats.asset_count == 100
    assert test_stats.row_count == 13_096
    assert test_stats.asset_count == 100
    assert rul_stats.row_count == 100

    train = pd.read_parquet(result.output_paths[0])
    assert list(train.columns) == list(TRAJECTORY_COLUMNS)
    assert len(train) == 20_631

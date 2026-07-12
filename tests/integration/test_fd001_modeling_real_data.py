"""Optional local FD001 Loop 4 input-contract integration test.

The expensive real candidate training remains an explicit CLI validation step;
the default suite only verifies the real role/schema contract when data exists.
"""

from pathlib import Path

import pytest

from turbine_guard.modeling.config import TrainingConfig
from turbine_guard.modeling.data import DatasetRole, load_verified_model_data

REPO_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
FEATURE_MANIFEST = REPO_DATA_DIR / "features" / "cmapss" / "FD001" / "feature_manifest.json"

pytestmark = [
    pytest.mark.real_data,
    pytest.mark.skipif(not FEATURE_MANIFEST.exists(), reason="FD001 Loop 3 features not available"),
]


def test_real_fd001_modeling_contract() -> None:
    data = load_verified_model_data(TrainingConfig(data_dir=REPO_DATA_DIR))
    assert len(data.feature_columns) == 552
    assert len(data.frame(DatasetRole.TRAIN)) == 14_407
    assert len(data.frame(DatasetRole.VALIDATION)) == 3_160
    assert len(data.frame(DatasetRole.CALIBRATION)) == 909
    assert len(data.frame(DatasetRole.REPLAY)) == 2_155

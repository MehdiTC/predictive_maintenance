"""Optional real-FD001 Loop 5 tracking/registry integration test."""

from pathlib import Path

import pytest

from turbine_guard.modeling.config import TrainingConfig
from turbine_guard.tracking.config import MlflowConfig
from turbine_guard.tracking.mlflow_tracker import MlflowTracker, TrackingStatus

REPO_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
TRAINING_MANIFEST = REPO_DATA_DIR / "models" / "cmapss" / "FD001" / "training_manifest.json"

pytestmark = [
    pytest.mark.real_data,
    pytest.mark.skipif(
        not TRAINING_MANIFEST.exists(), reason="FD001 Loop 4 artifacts not available"
    ),
]


def test_real_fd001_artifacts_track_and_register_in_temporary_store(tmp_path: Path) -> None:
    tracking = MlflowConfig(
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        experiment_name="FD001-real-artifact-test",
        registered_model_name="FD001-real-test-model",
        artifact_location=str(tmp_path / "artifacts"),
        registration_enabled=True,
        promote_champion=True,
        candidate_alias="candidate",
        challenger_alias="challenger",
        champion_alias="champion",
        archived_alias="archived",
        run_name_prefix="real-fd001",
        project_tag="turbine-guard-test",
        environment="testing",
    )
    result = MlflowTracker(tracking).track(TrainingConfig(data_dir=REPO_DATA_DIR))

    assert result.status is TrackingStatus.LOGGED
    assert len(result.candidate_run_ids) == 14
    assert result.selected_candidate_id == "capped_125--ridge_alpha_1"
    assert result.registered_version == "1"
    assert result.aliases["champion"] == "1"
    assert result.max_prediction_difference == 0.0

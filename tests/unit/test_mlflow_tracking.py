"""Temporary-SQLite MLflow tracking, pyfunc, registry, alias, and idempotency tests."""

from pathlib import Path

import mlflow
import numpy as np
import pytest
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from turbine_guard.modeling.artifacts import load_joblib
from turbine_guard.modeling.config import (
    CandidateConfig,
    ModelKind,
    SelectionConfig,
    TargetConfig,
    TrainingConfig,
)
from turbine_guard.modeling.data import DatasetRole, load_verified_model_data, model_matrix
from turbine_guard.modeling.pipeline import ModelBundle, train_models
from turbine_guard.tracking.artifacts import load_tracking_artifacts
from turbine_guard.tracking.config import MlflowConfig
from turbine_guard.tracking.mlflow_tracker import (
    MlflowTracker,
    TrackingStatus,
)
from turbine_guard.tracking.model import POINT_COLUMN


def tiny_candidates() -> tuple[CandidateConfig, ...]:
    return (
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
    )


def training_config(data_dir: Path, output_dir: Path) -> TrainingConfig:
    return TrainingConfig(
        data_dir=data_dir,
        output_dir=output_dir,
        targets=(TargetConfig("uncapped"), TargetConfig("capped_125", 125)),
        candidates=tiny_candidates(),
        selection=SelectionConfig(
            minimum_critical_recall=0.0,
            maximum_false_alarms_per_1000_cycles=1000.0,
            relative_rmse_tolerance=0.0,
        ),
        conformal_coverage=0.8,
        latency_repeats=1,
    )


def mlflow_config(
    tmp_path: Path,
    *,
    force_new_run: bool = False,
    force_new_model_version: bool = False,
) -> MlflowConfig:
    return MlflowConfig(
        tracking_uri=f"sqlite:///{tmp_path / 'tracking.db'}",
        experiment_name="FD001-test",
        registered_model_name="FD001-test-model",
        artifact_location=str(tmp_path / "mlflow-artifacts"),
        registration_enabled=True,
        promote_champion=True,
        candidate_alias="candidate",
        challenger_alias="challenger",
        champion_alias="champion",
        archived_alias="archived",
        run_name_prefix="test",
        project_tag="turbine-guard-test",
        environment="testing",
        force_new_run=force_new_run,
        force_new_model_version=force_new_model_version,
    )


def test_tracking_packaging_registry_and_prediction_contract(
    feature_data_dir: Path, tmp_path: Path
) -> None:
    training = training_config(feature_data_dir, tmp_path / "models")
    train_models(training)
    tracking = mlflow_config(tmp_path)
    result = MlflowTracker(tracking).track(training)

    assert result.status is TrackingStatus.LOGGED
    assert len(result.candidate_run_ids) == 8
    assert result.registered_version == "1"
    assert result.max_prediction_difference == 0.0
    assert result.aliases == {"candidate": "1", "challenger": "1", "champion": "1"}

    client = MlflowClient(tracking_uri=tracking.tracking_uri, registry_uri=tracking.tracking_uri)
    parent = client.get_run(result.parent_run_id)
    assert parent.data.tags["raw_acquisition_manifest_sha256"]
    assert parent.data.tags["validation_report_sha256"]
    assert parent.data.tags["feature_manifest_sha256"]
    assert parent.data.tags["git_commit_sha"]
    child = client.get_run(result.selected_run_id)
    assert child.data.metrics["selection/candidate_eligible"] == 1.0
    assert child.data.metrics["selection/candidate_rank"] >= 1.0
    assert "validation/rmse" in child.data.metrics
    assert "replay/rmse" in child.data.metrics
    assert "official_test/rmse" in child.data.metrics
    assert "calibration/interval_coverage" in child.data.metrics
    assert "policy/base/predictive_cost" in child.data.metrics
    version = client.get_model_version(tracking.registered_model_name, "1")
    assert version.run_id == result.selected_run_id
    assert version.tags["feature_manifest_sha256"]
    assert any(
        item.path == "champion/model_card.md"
        for item in client.list_artifacts(result.selected_run_id, "champion")
    )
    assert client.list_artifacts(result.parent_run_id, "lineage")
    assert client.list_artifacts(result.selected_run_id, "candidate/pipeline")

    mlflow.set_tracking_uri(tracking.tracking_uri)
    mlflow.set_registry_uri(tracking.tracking_uri)
    loaded = mlflow.pyfunc.load_model(f"models:/{tracking.registered_model_name}@champion")
    data = load_verified_model_data(training)
    features = (
        model_matrix(data.frame(DatasetRole.VALIDATION), data.feature_columns).dropna().head(3)
    )
    local = load_joblib(load_tracking_artifacts(training).champion_path)
    assert isinstance(local, ModelBundle)
    prediction = loaded.predict(features)
    np.testing.assert_allclose(prediction[POINT_COLUMN], local.predict(features), rtol=0, atol=0)
    assert list(prediction.columns) == [
        "predicted_rul",
        "lower_rul",
        "upper_rul",
        "risk_level",
    ]
    assert set(prediction["risk_level"]) <= {"healthy", "warning", "critical"}
    assert loaded.metadata.signature is not None
    input_names = {item.name for item in loaded.metadata.signature.inputs.inputs}
    assert input_names == set(data.feature_columns)
    assert not {"asset_id", "cycle", "split", "rul"} & input_names

    reordered = features.loc[:, list(reversed(features.columns))]
    np.testing.assert_allclose(
        loaded.predict(reordered)[POINT_COLUMN], local.predict(features), rtol=0, atol=0
    )
    with_extra = features.assign(unexpected=123.0)
    np.testing.assert_allclose(
        loaded.predict(with_extra)[POINT_COLUMN], local.predict(features), rtol=0, atol=0
    )
    with pytest.raises(MlflowException, match="missing"):
        loaded.predict(features.drop(columns=[features.columns[0]]))


def test_tracking_and_registry_idempotency_and_explicit_new_version(
    feature_data_dir: Path, tmp_path: Path
) -> None:
    training = training_config(feature_data_dir, tmp_path / "models")
    train_models(training)
    first = MlflowTracker(mlflow_config(tmp_path)).track(training)
    second = MlflowTracker(mlflow_config(tmp_path)).track(training)
    assert second.status is TrackingStatus.ALREADY_LOGGED
    assert second.parent_run_id == first.parent_run_id
    assert second.registered_version == "1"

    reused_version = MlflowTracker(mlflow_config(tmp_path, force_new_run=True)).track(training)
    assert reused_version.parent_run_id != first.parent_run_id
    assert reused_version.registered_version == "1"

    forced = MlflowTracker(
        mlflow_config(tmp_path, force_new_run=True, force_new_model_version=True)
    ).track(training)
    assert forced.status is TrackingStatus.LOGGED
    assert forced.parent_run_id != first.parent_run_id
    assert forced.registered_version == "2"
    assert forced.aliases["champion"] == "2"
    assert forced.aliases["archived"] == "1"
    client = MlflowClient(
        tracking_uri=mlflow_config(tmp_path).tracking_uri,
        registry_uri=mlflow_config(tmp_path).tracking_uri,
    )
    assert len(client.search_model_versions("name = 'FD001-test-model'")) == 2
    assert client.get_model_version("FD001-test-model", "1") is not None


def test_tracking_rejects_tampered_local_champion(feature_data_dir: Path, tmp_path: Path) -> None:
    training = training_config(feature_data_dir, tmp_path / "models")
    train_models(training)
    artifacts = load_tracking_artifacts(training)
    artifacts.champion_path.write_bytes(artifacts.champion_path.read_bytes() + b"tamper")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        MlflowTracker(mlflow_config(tmp_path)).track(training)

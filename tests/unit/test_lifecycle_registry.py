"""Local MLflow lifecycle registration, alias transition, rejection, and rollback."""

from pathlib import Path

import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline

from turbine_guard.modeling.artifacts import serialize_joblib, sha256_bytes
from turbine_guard.modeling.conformal import SplitConformalCalibrator
from turbine_guard.modeling.estimators import MedianRulRegressor
from turbine_guard.modeling.pipeline import ModelBundle
from turbine_guard.tracking.config import MlflowConfig
from turbine_guard.tracking.lifecycle import (
    configured_mlflow,
    promote_candidate,
    register_candidate,
    rollback_champion,
)


def _config(tmp_path: Path) -> MlflowConfig:
    return MlflowConfig(
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        experiment_name="lifecycle-test",
        registered_model_name="LifecycleModel",
        artifact_location=str(tmp_path / "artifacts"),
        registration_enabled=True,
        promote_champion=False,
        candidate_alias="candidate",
        challenger_alias="challenger",
        champion_alias="champion",
        archived_alias="archived",
        run_name_prefix="test",
        project_tag="test",
        environment="testing",
    )


def _bundle(constant: float) -> ModelBundle:
    pipeline = Pipeline((("model", MedianRulRegressor()),))
    pipeline.fit(pd.DataFrame({"feature": [0.0, 1.0]}), [constant, constant])
    conformal = SplitConformalCalibrator(0.9).fit([constant], [constant])
    return ModelBundle(
        pipeline=pipeline,
        feature_columns=("feature",),
        target_name="capped_125",
        target_cap=125,
        critical_horizon=30,
        warning_horizon=50,
        conformal=conformal,
        metadata={
            "model_kind": "constant_median",
            "model_configuration": {},
        },
    )


def _comparison(value: float) -> dict[str, object]:
    regression = {"mae": value, "rmse": value, "r2": 0.0, "nasa_score": value}
    return {
        "candidate": {
            "regression": regression,
            "critical": {
                "recall": 1.0,
                "precision": 1.0,
                "false_alarms_per_1000_cycles": 0.0,
            },
            "interval": {"empirical_coverage": 1.0, "average_width": 0.0},
            "inference_latency_ms": 0.1,
            "artifact_size_bytes": 100,
        },
        "champion": {"regression": regression},
        "naive": {"regression": regression},
    }


def _register(
    tmp_path: Path, config: MlflowConfig, lifecycle_id: str, constant: float
) -> tuple[str, ModelBundle]:
    bundle = _bundle(constant)
    content = serialize_joblib(bundle)
    path = tmp_path / f"{lifecycle_id}.joblib"
    path.write_bytes(content)
    registration = register_candidate(
        config=config,
        lifecycle_id=lifecycle_id,
        bundle_path=path,
        bundle_sha256=sha256_bytes(content),
        bundle=bundle,
        input_example=pd.DataFrame({"feature": [0.0, 1.0]}),
        comparison=_comparison(constant),
        lineage={
            "feature_version": "1",
            "feature_manifest_sha256": "a" * 64,
            "target_configuration": '{"cap": 125, "name": "capped_125"}',
            "git_sha": "b" * 40,
        },
    )
    assert registration.max_prediction_difference == 0.0
    assert registration.aliases["candidate"] == registration.version
    assert registration.aliases["challenger"] == registration.version
    return registration.version, bundle


def test_candidate_registration_promotion_rejection_rollback_and_idempotency(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    first, _ = _register(tmp_path, config, "lifecycle-1", 5.0)
    with configured_mlflow(config) as client:
        client.set_registered_model_alias(config.registered_model_name, "champion", first)

    second, _ = _register(tmp_path, config, "lifecycle-2", 4.0)
    repeated, _ = _register(tmp_path, config, "lifecycle-2", 4.0)
    assert repeated == second
    aliases = promote_candidate(config=config, candidate_version=second, expected_champion=first)
    assert aliases == {
        "candidate": second,
        "challenger": second,
        "archived": first,
        "champion": second,
    }

    rejected, _ = _register(tmp_path, config, "lifecycle-rejected", 100.0)
    with configured_mlflow(config) as client:
        current = client.get_model_version_by_alias(config.registered_model_name, "champion")
        assert str(current.version) == second
        assert rejected != second

    displaced, rolled_back = rollback_champion(config=config, target_version=first)
    assert displaced == second
    assert rolled_back["champion"] == first
    assert rolled_back["archived"] == second
    with configured_mlflow(config) as client:
        versions = client.search_model_versions(f"name = '{config.registered_model_name}'")
        assert len(versions) == 3
        runs = MlflowClient(tracking_uri=config.tracking_uri).search_runs(
            [client.get_experiment_by_name(config.experiment_name).experiment_id]
        )
        assert len(runs) == 3

"""Loop 9 candidate registry, safe alias transition, and rollback operations."""

import json
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from turbine_guard.modeling.pipeline import ModelBundle
from turbine_guard.tracking.config import MlflowConfig
from turbine_guard.tracking.mlflow_tracker import BUNDLE_TAG
from turbine_guard.tracking.model import log_bundle_model

LIFECYCLE_TAG = "turbine_guard.lifecycle_id"


@dataclass(frozen=True)
class ChampionRegistryState:
    version: str
    run_id: str
    bundle: ModelBundle
    run_params: dict[str, str]
    run_tags: dict[str, str]
    run_metrics: dict[str, float]
    version_tags: dict[str, str]


@dataclass(frozen=True)
class CandidateRegistration:
    version: str
    run_id: str
    max_prediction_difference: float
    aliases: dict[str, str]


@contextmanager
def configured_mlflow(config: MlflowConfig) -> Iterator[MlflowClient]:
    """Temporarily configure both tracking and registry without leaking global state."""
    previous_tracking = mlflow.get_tracking_uri()
    previous_registry = mlflow.get_registry_uri()
    config.prepare_local_store()
    try:
        mlflow.set_tracking_uri(config.tracking_uri)
        mlflow.set_registry_uri(config.tracking_uri)
        yield MlflowClient(tracking_uri=config.tracking_uri, registry_uri=config.tracking_uri)
    finally:
        mlflow.set_tracking_uri(previous_tracking)
        mlflow.set_registry_uri(previous_registry)


def load_champion(config: MlflowConfig) -> ChampionRegistryState:
    """Load the exact current champion and recover its Loop 4 fit configuration."""
    with configured_mlflow(config) as client:
        version = client.get_model_version_by_alias(
            config.registered_model_name, config.champion_alias
        )
        if version.run_id is None:
            raise ValueError("Champion registry version has no source run.")
        model = mlflow.pyfunc.load_model(
            f"models:/{config.registered_model_name}@{config.champion_alias}"
        )
        python_model = model.unwrap_python_model()
        bundle = getattr(python_model, "_bundle", None)
        if not isinstance(bundle, ModelBundle):
            raise TypeError("Champion pyfunc does not contain a verified Loop 4 bundle.")
        run = client.get_run(version.run_id)
        return ChampionRegistryState(
            version=str(version.version),
            run_id=str(version.run_id),
            bundle=bundle,
            run_params={str(key): str(value) for key, value in run.data.params.items()},
            run_tags={str(key): str(value) for key, value in run.data.tags.items()},
            run_metrics={str(key): float(value) for key, value in run.data.metrics.items()},
            version_tags={str(key): str(value) for key, value in version.tags.items()},
        )


def champion_baseline_metrics(state: ChampionRegistryState) -> dict[str, Any]:
    """Map the Loop 5 source-run evidence into the delayed-monitoring metric shape."""
    metrics = state.run_metrics
    return {
        "regression": {
            name: metrics.get(f"replay/{name}") for name in ("mae", "rmse", "nasa_score")
        },
        "critical": {
            "recall": metrics.get("replay/critical/recall"),
            "precision": metrics.get("replay/critical/precision"),
            "false_alarms_per_1000_cycles": metrics.get(
                "replay/critical/false_alarms_per_1000_cycles"
            ),
        },
        "interval": {
            "empirical_coverage": metrics.get("replay/interval_coverage"),
            "average_width": metrics.get("replay/interval_average_width"),
        },
    }


def attach_training_reference(
    *,
    config: MlflowConfig,
    champion: ChampionRegistryState,
    reference_path: Path,
    reference_sha256: str,
) -> None:
    """Idempotently associate the training-only reference with the exact model version."""
    with configured_mlflow(config) as client:
        current = client.get_model_version(config.registered_model_name, champion.version)
        existing = current.tags.get("turbine_guard.training_reference_sha256")
        if existing is not None and existing != reference_sha256:
            raise ValueError("Champion version already has a different training reference.")
        if existing is None:
            client.log_artifact(
                champion.run_id, str(reference_path), artifact_path="monitoring/reference"
            )
            client.set_model_version_tag(
                config.registered_model_name,
                champion.version,
                "turbine_guard.training_reference_sha256",
                reference_sha256,
            )


def register_candidate(
    *,
    config: MlflowConfig,
    lifecycle_id: str,
    bundle_path: Path,
    bundle_sha256: str,
    bundle: ModelBundle,
    input_example: pd.DataFrame,
    comparison: dict[str, Any],
    lineage: dict[str, str],
) -> CandidateRegistration:
    """Log/register exactly once, then perform candidate → verify → challenger transitions."""
    with configured_mlflow(config) as client:
        experiment_id = _experiment_id(client, config)
        run_id, model_uri = _candidate_run(
            client=client,
            experiment_id=experiment_id,
            lifecycle_id=lifecycle_id,
            bundle_path=bundle_path,
            bundle_sha256=bundle_sha256,
            bundle=bundle,
            input_example=input_example,
            comparison=comparison,
            lineage=lineage,
        )
        _ensure_registered_model(client, config)
        version = _candidate_version(
            client,
            config,
            lifecycle_id=lifecycle_id,
            run_id=run_id,
            model_uri=model_uri,
            bundle_sha256=bundle_sha256,
            lineage=lineage,
            comparison=comparison,
        )
        client.set_registered_model_alias(
            config.registered_model_name, config.candidate_alias, version
        )
        maximum = _verify_equivalence(config, version, bundle, input_example)
        client.set_model_version_tag(
            config.registered_model_name,
            version,
            "turbine_guard.max_prediction_difference",
            str(maximum),
        )
        client.set_registered_model_alias(
            config.registered_model_name, config.challenger_alias, version
        )
        return CandidateRegistration(version, run_id, maximum, aliases(client, config))


def promote_candidate(
    *, config: MlflowConfig, candidate_version: str, expected_champion: str
) -> dict[str, str]:
    """Archive the expected champion and atomically ordered-reassign the champion alias."""
    with configured_mlflow(config) as client:
        current = str(
            client.get_model_version_by_alias(
                config.registered_model_name, config.champion_alias
            ).version
        )
        if current == candidate_version:
            return aliases(client, config)
        if current != expected_champion:
            raise ValueError(
                f"Champion changed from expected version {expected_champion} to {current}."
            )
        _validate_version(client, config, candidate_version)
        client.set_registered_model_alias(
            config.registered_model_name, config.archived_alias, current
        )
        client.set_model_version_tag(
            config.registered_model_name,
            current,
            "turbine_guard.champion_displaced_at",
            datetime.now(UTC).isoformat(),
        )
        client.set_registered_model_alias(
            config.registered_model_name, config.champion_alias, candidate_version
        )
        return aliases(client, config)


def rollback_champion(*, config: MlflowConfig, target_version: str) -> tuple[str, dict[str, str]]:
    """Validate and restore a numbered version while retaining the displaced champion."""
    with configured_mlflow(config) as client:
        current = str(
            client.get_model_version_by_alias(
                config.registered_model_name, config.champion_alias
            ).version
        )
        if current == target_version:
            return current, aliases(client, config)
        _validate_version(client, config, target_version)
        client.set_registered_model_alias(
            config.registered_model_name, config.archived_alias, current
        )
        client.set_registered_model_alias(
            config.registered_model_name, config.champion_alias, target_version
        )
        client.set_model_version_tag(
            config.registered_model_name,
            target_version,
            "turbine_guard.rollback_at",
            datetime.now(UTC).isoformat(),
        )
        return current, aliases(client, config)


def aliases(client: MlflowClient, config: MlflowConfig) -> dict[str, str]:
    model = client.get_registered_model(config.registered_model_name)
    values = model.aliases
    if isinstance(values, dict):
        return {str(alias): str(version) for alias, version in values.items()}
    return {str(value.alias): str(value.version) for value in values}


def _experiment_id(client: MlflowClient, config: MlflowConfig) -> str:
    experiment = client.get_experiment_by_name(config.experiment_name)
    if experiment is not None:
        return str(experiment.experiment_id)
    return client.create_experiment(
        config.experiment_name,
        artifact_location=config.resolved_artifact_location(),
        tags={"project": config.project_tag, "purpose": "model_lifecycle"},
    )


def _candidate_run(
    *,
    client: MlflowClient,
    experiment_id: str,
    lifecycle_id: str,
    bundle_path: Path,
    bundle_sha256: str,
    bundle: ModelBundle,
    input_example: pd.DataFrame,
    comparison: dict[str, Any],
    lineage: dict[str, str],
) -> tuple[str, str]:
    existing = client.search_runs(
        [experiment_id],
        filter_string=(
            f"tags.`{LIFECYCLE_TAG}` = '{lifecycle_id}' and "
            "tags.`run_purpose` = 'retraining_candidate' and attributes.status = 'FINISHED'"
        ),
        max_results=1,
    )
    if existing:
        model_uri = existing[0].data.tags.get("turbine_guard.lifecycle_model_uri")
        if not model_uri:
            raise ValueError("Existing lifecycle run is missing its logged model URI.")
        return existing[0].info.run_id, model_uri

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=f"lifecycle-{lifecycle_id[:12]}",
        tags={
            LIFECYCLE_TAG: lifecycle_id,
            "run_purpose": "retraining_candidate",
            "feature_version": lineage["feature_version"],
            "target_type": bundle.target_name,
            **lineage,
        },
    ) as run:
        mlflow.log_params(
            {
                "rul_cap": bundle.target_cap if bundle.target_cap is not None else "none",
                "feature_count": len(bundle.feature_columns),
                "conformal_target_coverage": bundle.conformal.coverage,
                "model_family": bundle.metadata["model_kind"],
                **bundle.metadata.get("model_configuration", {}),
            }
        )
        _log_comparison_metrics(comparison)
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "promotion_comparison.json"
            report.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
            mlflow.log_artifact(str(report), artifact_path="promotion")
        info = log_bundle_model(
            name="lifecycle_candidate",
            bundle_path=bundle_path,
            bundle_sha256=bundle_sha256,
            feature_columns=bundle.feature_columns,
            input_example=input_example,
            metadata={
                "lifecycle_id": lifecycle_id,
                "target_definition": {
                    "name": bundle.target_name,
                    "cap": bundle.target_cap,
                },
                "bundle_sha256": bundle_sha256,
            },
        )
        client.set_tag(run.info.run_id, "turbine_guard.lifecycle_model_uri", info.model_uri)
        return run.info.run_id, info.model_uri


def _candidate_version(
    client: MlflowClient,
    config: MlflowConfig,
    *,
    lifecycle_id: str,
    run_id: str,
    model_uri: str,
    bundle_sha256: str,
    lineage: dict[str, str],
    comparison: dict[str, Any],
) -> str:
    for version in client.search_model_versions(f"name = '{config.registered_model_name}'"):
        if version.tags.get(LIFECYCLE_TAG) == lifecycle_id:
            return str(version.version)
    candidate_metrics = comparison["candidate"]
    tags = {
        LIFECYCLE_TAG: lifecycle_id,
        BUNDLE_TAG: bundle_sha256,
        "turbine_guard.source_run_id": run_id,
        "feature_manifest_sha256": lineage["feature_manifest_sha256"],
        "feature_version": lineage["feature_version"],
        "target_configuration": lineage["target_configuration"],
        "validation_rmse": str(candidate_metrics["regression"]["rmse"]),
        "replay_rmse": str(candidate_metrics["regression"]["rmse"]),
        "registration_timestamp": datetime.now(UTC).isoformat(),
    }
    created = client.create_model_version(
        name=config.registered_model_name,
        source=model_uri,
        run_id=run_id,
        tags=tags,
        description="Loop 9 retraining candidate evaluated on a disjoint promotion holdout.",
    )
    return str(created.version)


def _ensure_registered_model(client: MlflowClient, config: MlflowConfig) -> None:
    try:
        client.get_registered_model(config.registered_model_name)
    except MlflowException:
        client.create_registered_model(
            config.registered_model_name,
            tags={"project": config.project_tag, "dataset_subset": "FD001"},
            description="TurbineGuard Loop 9 lifecycle registry.",
        )


def _verify_equivalence(
    config: MlflowConfig,
    version: str,
    bundle: ModelBundle,
    input_example: pd.DataFrame,
) -> float:
    loaded = mlflow.pyfunc.load_model(f"models:/{config.registered_model_name}/{version}")
    expected_point = bundle.predict(input_example)
    expected_lower, expected_upper = bundle.predict_interval(input_example)
    result = loaded.predict(input_example)
    if not isinstance(result, pd.DataFrame):
        raise ValueError("Reloaded candidate returned an invalid output type.")
    actual = result[["predicted_rul", "lower_rul", "upper_rul"]].to_numpy(dtype="float64")
    expected = np.column_stack((expected_point, expected_lower, expected_upper))
    maximum = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
    expected_risk = np.where(
        expected_point <= bundle.critical_horizon,
        "critical",
        np.where(expected_point <= bundle.warning_horizon, "warning", "healthy"),
    )
    if list(result["risk_level"].astype(str)) != list(expected_risk):
        raise ValueError("Reloaded candidate risk predictions differ from the local bundle.")
    return maximum


def _validate_version(client: MlflowClient, config: MlflowConfig, version: str) -> None:
    item = client.get_model_version(config.registered_model_name, version)
    if item.run_id is None:
        raise ValueError(f"Registry version {version} has no source run.")
    model = mlflow.pyfunc.load_model(f"models:/{config.registered_model_name}/{version}")
    schema = model.metadata.get_input_schema()
    if schema is None or not schema.input_names():
        raise ValueError(f"Registry version {version} has no valid feature signature.")


def _log_comparison_metrics(comparison: dict[str, Any]) -> None:
    metrics: dict[str, float] = {}
    for role in ("candidate", "champion", "naive"):
        for name, value in comparison[role]["regression"].items():
            if value is not None:
                metrics[f"promotion/{role}/{name}"] = float(value)
    candidate = comparison["candidate"]
    metrics.update(
        {
            "promotion/candidate/critical_recall": float(candidate["critical"]["recall"]),
            "promotion/candidate/critical_precision": float(candidate["critical"]["precision"]),
            "promotion/candidate/false_alarms_per_1000_cycles": float(
                candidate["critical"]["false_alarms_per_1000_cycles"]
            ),
            "promotion/candidate/interval_coverage": float(
                candidate["interval"]["empirical_coverage"]
            ),
            "promotion/candidate/interval_width": float(candidate["interval"]["average_width"]),
            "promotion/candidate/inference_latency_ms": float(candidate["inference_latency_ms"]),
            "promotion/candidate/artifact_size_bytes": float(candidate["artifact_size_bytes"]),
        }
    )
    mlflow.log_metrics(metrics)

"""Concise inspection and prediction-equivalence CLI for local/remote MLflow."""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from turbine_guard.config.settings import get_settings
from turbine_guard.modeling.artifacts import load_joblib
from turbine_guard.modeling.config import TrainingConfig
from turbine_guard.modeling.data import DatasetRole, load_verified_model_data, model_matrix
from turbine_guard.modeling.pipeline import ModelBundle
from turbine_guard.tracking.artifacts import load_tracking_artifacts
from turbine_guard.tracking.config import MlflowConfig
from turbine_guard.tracking.model import POINT_COLUMN


def build_parser() -> argparse.ArgumentParser:
    """Build the MLflow utility parser."""
    parser = argparse.ArgumentParser(prog="mlflow_models")
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser(
        "inspect", help="Show parent/candidate runs, metrics, versions, and aliases."
    )
    inspect_parser.add_argument("--limit", type=int, default=5)

    verify = subparsers.add_parser(
        "verify", help="Compare a registered model with the local champion bundle."
    )
    selector = verify.add_mutually_exclusive_group()
    selector.add_argument("--alias", default=None)
    selector.add_argument("--version", default=None)
    verify.add_argument("--rows", type=int, default=10)
    verify.add_argument("--data-dir", type=Path, default=None)
    verify.add_argument("--output-dir", type=Path, default=None)

    subparsers.add_parser("state", help="List configured local tracking and artifact paths.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run an MLflow utility command with nonzero status on failure."""
    args = build_parser().parse_args(argv)
    settings = get_settings()
    config = MlflowConfig.from_settings(settings)
    try:
        config.prepare_local_store()
        mlflow.set_tracking_uri(config.tracking_uri)
        mlflow.set_registry_uri(config.tracking_uri)
        if args.command == "inspect":
            payload = _inspect(config, args.limit)
        elif args.command == "verify":
            payload = _verify(
                config,
                data_dir=args.data_dir or settings.data_dir,
                output_dir=args.output_dir,
                alias=args.alias or (None if args.version else config.champion_alias),
                version_value=args.version,
                rows=args.rows,
            )
        else:
            payload = _state(config)
    except (OSError, RuntimeError, ValueError) as exc:
        sys.stderr.write(json.dumps({"status": "error", "error": str(exc)}) + "\n")
        return 1
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


def _inspect(config: MlflowConfig, limit: int) -> dict[str, Any]:
    if limit < 1:
        raise ValueError("--limit must be positive.")
    client = MlflowClient(tracking_uri=config.tracking_uri, registry_uri=config.tracking_uri)
    experiment = client.get_experiment_by_name(config.experiment_name)
    parents: list[dict[str, Any]] = []
    if experiment is not None:
        runs = client.search_runs(
            [experiment.experiment_id],
            filter_string="tags.`run_purpose` = 'complete_training_execution'",
            order_by=["attributes.start_time DESC"],
            max_results=limit,
        )
        for run in runs:
            children = client.search_runs(
                [experiment.experiment_id],
                filter_string=f"tags.`mlflow.parentRunId` = '{run.info.run_id}'",
                max_results=1000,
            )
            parents.append(
                {
                    "run_id": run.info.run_id,
                    "status": run.info.status,
                    "execution_id": run.data.tags.get("turbine_guard.execution_id"),
                    "selected_candidate_id": run.data.tags.get("selected_candidate_id"),
                    "registry_version": run.data.tags.get("turbine_guard.registry_version"),
                    "candidates": [
                        {
                            "run_id": child.info.run_id,
                            "candidate_id": child.data.tags.get("candidate_id"),
                            "eligible": child.data.tags.get("eligible"),
                            "rank": child.data.metrics.get("selection/candidate_rank"),
                            "validation_rmse": child.data.metrics.get("validation/rmse"),
                            "common_domain_rmse": child.data.metrics.get(
                                "validation/common_domain/rmse"
                            ),
                            "critical_recall": child.data.metrics.get("validation/critical/recall"),
                        }
                        for child in sorted(
                            children,
                            key=lambda item: item.data.metrics.get(
                                "selection/candidate_rank", float("inf")
                            ),
                        )
                    ],
                }
            )
    versions: list[dict[str, Any]] = []
    aliases: dict[str, str] = {}
    try:
        registered = client.get_registered_model(config.registered_model_name)
        if isinstance(registered.aliases, dict):
            aliases = {
                str(alias): str(version_value)
                for alias, version_value in registered.aliases.items()
            }
        else:
            aliases = {item.alias: item.version for item in registered.aliases}
        versions = [
            {
                "version": item.version,
                "run_id": item.run_id,
                "status": item.status,
                "bundle_sha256": item.tags.get("turbine_guard.champion_bundle_sha256"),
                "validation_rmse": item.tags.get("validation_rmse"),
                "replay_rmse": item.tags.get("replay_rmse"),
                "official_test_rmse": item.tags.get("official_test_rmse"),
            }
            for item in client.search_model_versions(f"name = '{config.registered_model_name}'")
        ]
    except MlflowException:
        pass
    return {
        "tracking_uri": config.tracking_uri,
        "experiment_name": config.experiment_name,
        "registered_model_name": config.registered_model_name,
        "parent_runs": parents,
        "model_versions": versions,
        "aliases": aliases,
    }


def _verify(
    config: MlflowConfig,
    *,
    data_dir: Path,
    output_dir: Path | None,
    alias: str | None,
    version_value: str | None,
    rows: int,
) -> dict[str, Any]:
    if rows < 1:
        raise ValueError("--rows must be positive.")
    training = TrainingConfig(data_dir=data_dir, output_dir=output_dir)
    artifacts = load_tracking_artifacts(training)
    data = load_verified_model_data(training)
    features = model_matrix(data.frame(DatasetRole.VALIDATION), data.feature_columns)
    complete = features.dropna()
    sample = (complete if not complete.empty else features).head(rows)
    local = load_joblib(artifacts.champion_path)
    if not isinstance(local, ModelBundle):
        raise RuntimeError("Local champion is not a ModelBundle.")
    selector = f"@{alias}" if alias is not None else f"/{version_value}"
    uri = f"models:/{config.registered_model_name}{selector}"
    registered = mlflow.pyfunc.load_model(uri)
    prediction = registered.predict(sample)
    difference = np.abs(prediction[POINT_COLUMN].to_numpy(dtype="float64") - local.predict(sample))
    maximum = float(np.max(difference)) if difference.size else 0.0
    if maximum > 1e-12:
        raise RuntimeError(f"Prediction equivalence failed with max difference {maximum}.")
    return {
        "status": "equivalent",
        "model_uri": uri,
        "rows": len(sample),
        "feature_count": len(data.feature_columns),
        "max_absolute_difference": maximum,
        "output_columns": list(prediction.columns),
    }


def _state(config: MlflowConfig) -> dict[str, Any]:
    artifact_location = config.resolved_artifact_location()
    return {
        "tracking_uri": config.tracking_uri,
        "artifact_location": artifact_location,
        "tracking_database_exists": _sqlite_database_exists(config.tracking_uri),
        "artifact_directory_exists": (
            artifact_location is not None
            and artifact_location.startswith("file:")
            and Path(artifact_location.removeprefix("file://")).exists()
        ),
    }


def _sqlite_database_exists(uri: str) -> bool | None:
    if not uri.startswith("sqlite:///"):
        return None
    return Path(uri.removeprefix("sqlite:///")).exists()

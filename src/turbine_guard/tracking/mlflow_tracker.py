"""MLflow parent/child tracking, champion packaging, registry, and aliases."""

import json
import logging
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow.entities import Run
from mlflow.exceptions import MlflowException
from mlflow.models.model import ModelInfo
from mlflow.tracking import MlflowClient

from turbine_guard import __version__
from turbine_guard.modeling.artifacts import ArtifactError, load_joblib
from turbine_guard.modeling.config import TrainingConfig
from turbine_guard.modeling.data import (
    DatasetRole,
    load_verified_model_data,
    model_matrix,
)
from turbine_guard.modeling.pipeline import ModelBundle
from turbine_guard.tracking.artifacts import TrackingArtifacts, load_tracking_artifacts
from turbine_guard.tracking.config import MlflowConfig
from turbine_guard.tracking.model import POINT_COLUMN, log_bundle_model

logger = logging.getLogger(__name__)

EXECUTION_TAG = "turbine_guard.execution_id"
BEHAVIOR_TAG = "turbine_guard.tracking_behavior_id"
BUNDLE_TAG = "turbine_guard.champion_bundle_sha256"
SOURCE_CHILD_TAG = "turbine_guard.source_child_run_id"


class TrackingStatus(StrEnum):
    """Outcome of optional MLflow integration."""

    LOGGED = "logged"
    ALREADY_LOGGED = "already_logged"


@dataclass(frozen=True)
class TrackingResult:
    """Concise auditable result of one MLflow tracking operation."""

    status: TrackingStatus
    execution_id: str
    experiment_id: str
    parent_run_id: str
    candidate_run_ids: dict[str, str]
    selected_candidate_id: str
    selected_run_id: str
    registered_model_name: str | None
    registered_version: str | None
    aliases: dict[str, str]
    max_prediction_difference: float | None


class MlflowTrackingError(RuntimeError):
    """Raised when a complete local run cannot be tracked safely."""


class MlflowTracker:
    """Isolated MLflow adapter over completed Loop 4 local artifacts."""

    def __init__(self, config: MlflowConfig) -> None:
        self.config = config

    def track(self, training_config: TrainingConfig) -> TrackingResult:
        """Log or reuse a complete execution, then optionally register its champion."""
        try:
            artifacts = load_tracking_artifacts(training_config)
            self.config.prepare_local_store()
            return self._with_configured_store(training_config, artifacts)
        except (ArtifactError, MlflowException, OSError, ValueError) as exc:
            raise MlflowTrackingError(str(exc)) from exc

    def _with_configured_store(
        self, training_config: TrainingConfig, artifacts: TrackingArtifacts
    ) -> TrackingResult:
        previous_tracking = mlflow.get_tracking_uri()
        previous_registry = mlflow.get_registry_uri()
        try:
            mlflow.set_tracking_uri(self.config.tracking_uri)
            mlflow.set_registry_uri(self.config.tracking_uri)
            client = MlflowClient(
                tracking_uri=self.config.tracking_uri,
                registry_uri=self.config.tracking_uri,
            )
            experiment_id = self._get_or_create_experiment(client)
            existing = self._find_existing_run(client, experiment_id, artifacts.execution_id)
            if existing is not None and not self.config.force_new_run:
                return self._existing_result(client, existing, artifacts)
            return self._log_new_execution(client, experiment_id, training_config, artifacts)
        finally:
            mlflow.set_tracking_uri(previous_tracking)
            mlflow.set_registry_uri(previous_registry)

    def _get_or_create_experiment(self, client: MlflowClient) -> str:
        experiment = client.get_experiment_by_name(self.config.experiment_name)
        if experiment is not None:
            return str(experiment.experiment_id)
        return client.create_experiment(
            self.config.experiment_name,
            artifact_location=self.config.resolved_artifact_location(),
            tags={
                "project": self.config.project_tag,
                "dataset_subset": "FD001",
                "purpose": "offline_modeling",
            },
        )

    def _find_existing_run(
        self, client: MlflowClient, experiment_id: str, execution_id: str
    ) -> Run | None:
        runs = client.search_runs(
            [experiment_id],
            filter_string=(
                f"tags.`{EXECUTION_TAG}` = '{execution_id}' and "
                f"tags.`{BEHAVIOR_TAG}` = '{self.config.behavior_id()}' and "
                "tags.`run_purpose` = 'complete_training_execution' and "
                "attributes.status = 'FINISHED'"
            ),
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        return runs[0] if runs else None

    def _existing_result(
        self, client: MlflowClient, run: Run, artifacts: TrackingArtifacts
    ) -> TrackingResult:
        tags = run.data.tags
        selected_run_id = tags.get(SOURCE_CHILD_TAG, "")
        if not selected_run_id:
            raise MlflowTrackingError(
                "Existing completed parent run lacks source child traceability."
            )
        version_value = tags.get("turbine_guard.registry_version")
        aliases = self._aliases(client) if version_value is not None else {}
        return TrackingResult(
            status=TrackingStatus.ALREADY_LOGGED,
            execution_id=artifacts.execution_id,
            experiment_id=run.info.experiment_id,
            parent_run_id=run.info.run_id,
            candidate_run_ids=self._child_run_ids(client, run.info.experiment_id, run.info.run_id),
            selected_candidate_id=artifacts.training_manifest.selected_candidate_id,
            selected_run_id=selected_run_id,
            registered_model_name=(
                self.config.registered_model_name if version_value is not None else None
            ),
            registered_version=version_value,
            aliases=aliases,
            max_prediction_difference=(
                _optional_float(tags.get("turbine_guard.max_prediction_difference"))
            ),
        )

    def _log_new_execution(
        self,
        client: MlflowClient,
        experiment_id: str,
        training_config: TrainingConfig,
        artifacts: TrackingArtifacts,
    ) -> TrackingResult:
        common_tags = self._lineage_tags(artifacts)
        parent_name = f"{self.config.run_name_prefix}-{artifacts.execution_id[:12]}"
        candidate_run_ids: dict[str, str] = {}
        selected_run_id = ""
        model_info: ModelInfo | None = None
        input_example, local_point = self._model_input(training_config, artifacts)
        ranks = _candidate_ranks(artifacts.candidate_reports)
        selection_by_id = {
            str(item["candidate_id"]): item for item in artifacts.selection_report["candidates"]
        }

        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=parent_name,
            tags={
                **common_tags,
                "run_purpose": "complete_training_execution",
                "selected_candidate_id": artifacts.training_manifest.selected_candidate_id,
            },
        ) as parent:
            parent_run_id = parent.info.run_id
            self._log_parent(artifacts, training_config)
            for candidate in artifacts.candidate_reports:
                candidate_id = str(candidate["candidate_id"])
                selection = selection_by_id[candidate_id]
                with mlflow.start_run(
                    experiment_id=experiment_id,
                    run_name=f"{self.config.run_name_prefix}-{candidate_id}",
                    nested=True,
                    tags={
                        **common_tags,
                        "run_purpose": "candidate_evaluation",
                        "candidate_id": candidate_id,
                        "model_family": str(candidate["model_kind"]),
                        "target_type": str(candidate["target_definition"]["name"]),
                        "eligible": str(bool(selection["eligible"])).lower(),
                        "is_selected_champion": str(
                            candidate_id == artifacts.training_manifest.selected_candidate_id
                        ).lower(),
                    },
                ) as child:
                    child_id = child.info.run_id
                    candidate_run_ids[candidate_id] = child_id
                    self._log_candidate(
                        candidate,
                        selection,
                        ranks[candidate_id],
                        artifacts,
                        training_config,
                    )
                    if candidate_id == artifacts.training_manifest.selected_candidate_id:
                        selected_run_id = child_id
                        self._log_champion_metrics(artifacts)
                        self._log_champion_artifacts(artifacts)
                        model_info = self._log_champion_model(artifacts, input_example)

            if not selected_run_id or model_info is None:
                raise MlflowTrackingError("Selected candidate child run was not logged.")
            client.set_tag(parent_run_id, SOURCE_CHILD_TAG, selected_run_id)

        version_value: str | None = None
        aliases: dict[str, str] = {}
        max_difference: float | None = None
        if self.config.registration_enabled:
            version_value = self._register_or_reuse(client, model_info, selected_run_id, artifacts)
            max_difference = self._verify_registered_model(
                version_value, input_example, local_point
            )
            aliases = self._assign_aliases(client, version_value, artifacts)
            self._attach_registry_metadata(
                client,
                parent_run_id,
                selected_run_id,
                version_value,
                aliases,
                max_difference,
                artifacts,
            )

        return TrackingResult(
            status=TrackingStatus.LOGGED,
            execution_id=artifacts.execution_id,
            experiment_id=experiment_id,
            parent_run_id=parent_run_id,
            candidate_run_ids=candidate_run_ids,
            selected_candidate_id=artifacts.training_manifest.selected_candidate_id,
            selected_run_id=selected_run_id,
            registered_model_name=(
                self.config.registered_model_name if version_value is not None else None
            ),
            registered_version=version_value,
            aliases=aliases,
            max_prediction_difference=max_difference,
        )

    def _lineage_tags(self, artifacts: TrackingArtifacts) -> dict[str, str]:
        metadata = artifacts.champion_metadata
        manifest = artifacts.training_manifest
        return {
            EXECUTION_TAG: artifacts.execution_id,
            BEHAVIOR_TAG: self.config.behavior_id(),
            "project": self.config.project_tag,
            "dataset_name": "NASA C-MAPSS Turbofan Engine Degradation Simulation",
            "dataset_subset": "FD001",
            "raw_acquisition_manifest_sha256": artifacts.acquisition_manifest_sha256,
            "validation_report_sha256": artifacts.validation_report_sha256,
            "split_manifest_sha256": manifest.split_manifest_sha256,
            "feature_manifest_sha256": manifest.feature_manifest_sha256,
            "training_configuration_sha256": manifest.configuration_sha256,
            "training_manifest_sha256": artifacts.training_manifest_sha256,
            "feature_version": str(
                _read_json(artifacts.feature_manifest_path)["feature_config"]["feature_version"]
            ),
            "split_version": str(_read_json(artifacts.split_manifest_path)["split_version"]),
            "evaluation_version": manifest.evaluation_version,
            "git_commit_sha": manifest.git_commit or "unknown",
            "code_version": __version__,
            "random_seed": str(metadata["random_seed"]),
            "environment": self.config.environment,
            "recorded_at": datetime.now(UTC).isoformat(),
        }

    def _log_parent(self, artifacts: TrackingArtifacts, training_config: TrainingConfig) -> None:
        metadata = artifacts.champion_metadata
        mlflow.log_params(
            {
                "dataset_subset": training_config.subset,
                "evaluation_version": artifacts.training_manifest.evaluation_version,
                "random_seed": metadata["random_seed"],
                "candidate_count": len(artifacts.candidate_reports),
                "feature_count": metadata["feature_count"],
                "selected_candidate_id": artifacts.training_manifest.selected_candidate_id,
            }
        )
        mlflow.log_metrics(
            {
                "selection/candidate_count": float(len(artifacts.candidate_reports)),
                "selection/eligible_count": float(
                    sum(bool(item["eligible"]) for item in artifacts.selection_report["candidates"])
                ),
            }
        )
        for path in (
            artifacts.acquisition_manifest_path,
            artifacts.validation_report_path,
            artifacts.feature_manifest_path,
            artifacts.split_manifest_path,
            artifacts.training_manifest_path,
        ):
            mlflow.log_artifact(str(path), artifact_path="lineage")
        mlflow.log_artifact(
            str(artifacts.root / "reports" / "champion_selection.json"),
            artifact_path="selection",
        )
        mlflow.log_artifact(
            str(artifacts.root / "reports" / "candidate_comparison.csv"),
            artifact_path="selection",
        )
        self._log_generated_json(
            "training_configuration.json", metadata["training_configuration"], "configuration"
        )
        for dependency in (Path("pyproject.toml"), Path("uv.lock")):
            if dependency.exists():
                mlflow.log_artifact(str(dependency), artifact_path="environment")

    def _log_candidate(
        self,
        candidate: dict[str, Any],
        selection: dict[str, Any],
        rank: int,
        artifacts: TrackingArtifacts,
        training_config: TrainingConfig,
    ) -> None:
        target = candidate["target_definition"]
        feature_manifest = _read_json(artifacts.feature_manifest_path)
        params: dict[str, Any] = {
            "model_family": candidate["model_kind"],
            "target_type": target["name"],
            "rul_cap": target["cap"] if target["cap"] is not None else "none",
            "imputation_strategy": training_config.imputation_strategy,
            "missing_indicators": candidate["model_kind"] == "ridge",
            "scaling": candidate["model_kind"] == "ridge" and training_config.scale_ridge,
            "feature_count": len(feature_manifest["feature_columns"]),
            "rolling_windows": ",".join(map(str, feature_manifest["feature_config"]["windows"])),
            "alert_horizons": (
                f"critical={training_config.alerts.critical_horizon},"
                f"warning={training_config.alerts.warning_horizon}"
            ),
            "random_seed": training_config.random_seed,
            "threshold_selection_rule": "fixed_predicted_rul_horizons",
            "conformal_target_coverage": training_config.conformal_coverage,
            **candidate["configuration"],
        }
        mlflow.log_params(params)
        metrics = _candidate_metrics(candidate, selection, rank)
        mlflow.log_metrics(metrics)
        candidate_id = str(candidate["candidate_id"])
        self._log_generated_json(
            "candidate_configuration_and_validation.json", candidate, "candidate"
        )
        model_path = artifacts.root / "models" / f"{candidate_id}.joblib"
        mlflow.log_artifact(str(model_path), artifact_path="candidate/pipeline")

    def _log_champion_metrics(self, artifacts: TrackingArtifacts) -> None:
        replay = artifacts.replay_report
        official = artifacts.official_report
        conformal = artifacts.conformal_report
        policy = artifacts.policy_report
        metrics: dict[str, float] = {}
        _add_regression_metrics(metrics, "replay", replay["regression_metrics"]["row_weighted"])
        _add_regression_metrics(
            metrics, "official_test", official["regression_metrics"]["row_weighted"]
        )
        _add_alert_metrics(metrics, "replay/critical", replay["critical_alert_metrics"])
        _add_alert_metrics(metrics, "replay/warning", replay["warning_alert_metrics"])
        calibration = conformal["calibration"]["empirical_metrics"]
        for prefix, record in (
            ("calibration", calibration),
            ("replay", conformal["replay"]),
            ("official_test", conformal["official_test"]),
        ):
            metrics[f"{prefix}/interval_coverage"] = float(record["empirical_coverage"])
            metrics[f"{prefix}/interval_average_width"] = float(record["average_width"])
            metrics[f"{prefix}/interval_median_width"] = float(record["median_width"])
        for scenario in policy["scenarios"]:
            name = str(scenario["name"])
            metrics[f"policy/{name}/reactive_cost"] = float(
                scenario["reactive"]["total_normalized_cost"]
            )
            metrics[f"policy/{name}/predictive_cost"] = float(
                scenario["predictive"]["total_normalized_cost"]
            )
            change = scenario["predictive"]["relative_cost_change_vs_reactive"]
            if change is not None:
                metrics[f"policy/{name}/relative_cost_change"] = float(change)
        mlflow.log_metrics(metrics)

    def _log_champion_artifacts(self, artifacts: TrackingArtifacts) -> None:
        mlflow.log_artifact(str(artifacts.champion_path), artifact_path="champion/local_bundle")
        mlflow.log_artifact(
            str(artifacts.root / "models" / "champion_metadata.json"),
            artifact_path="champion",
        )
        mlflow.log_artifacts(str(artifacts.root / "reports"), artifact_path="champion/reports")
        feature_manifest = _read_json(artifacts.feature_manifest_path)
        self._log_generated_json(
            "ordered_features.json",
            {"feature_columns": feature_manifest["feature_columns"]},
            "champion/contract",
        )
        self._log_generated_json(
            "alert_policy.json",
            artifacts.selection_report["selected_alert_thresholds"],
            "champion/contract",
        )
        self._log_generated_json(
            "checksums.json",
            {
                "raw_acquisition_manifest_sha256": artifacts.acquisition_manifest_sha256,
                "validation_report_sha256": artifacts.validation_report_sha256,
                "split_manifest_sha256": artifacts.training_manifest.split_manifest_sha256,
                "feature_manifest_sha256": artifacts.training_manifest.feature_manifest_sha256,
                "training_configuration_sha256": artifacts.training_manifest.configuration_sha256,
                "training_manifest_sha256": artifacts.training_manifest_sha256,
                "champion_bundle_sha256": artifacts.champion_sha256,
            },
            "champion/lineage",
        )

    def _log_champion_model(
        self, artifacts: TrackingArtifacts, input_example: pd.DataFrame
    ) -> ModelInfo:
        return log_bundle_model(
            name="champion_model",
            bundle_path=artifacts.champion_path,
            bundle_sha256=artifacts.champion_sha256,
            feature_columns=tuple(artifacts.champion_metadata["ordered_feature_list"]),
            input_example=input_example,
            metadata={
                "candidate_id": artifacts.training_manifest.selected_candidate_id,
                "target_definition": artifacts.champion_metadata["target_definition"],
                "feature_manifest_sha256": artifacts.training_manifest.feature_manifest_sha256,
                "bundle_sha256": artifacts.champion_sha256,
                "extra_column_policy": "ignored_by_mlflow_signature_enforcement",
            },
        )

    def _model_input(
        self, training_config: TrainingConfig, artifacts: TrackingArtifacts
    ) -> tuple[pd.DataFrame, np.ndarray]:
        data = load_verified_model_data(training_config)
        validation = model_matrix(data.frame(DatasetRole.VALIDATION), data.feature_columns)
        complete = validation.dropna()
        example = (complete if not complete.empty else validation).head(2).copy()
        bundle = load_joblib(artifacts.champion_path)
        if not isinstance(bundle, ModelBundle):
            raise MlflowTrackingError("Local champion artifact is not a ModelBundle.")
        return example, bundle.predict(example)

    def _register_or_reuse(
        self,
        client: MlflowClient,
        model_info: ModelInfo,
        selected_run_id: str,
        artifacts: TrackingArtifacts,
    ) -> str:
        try:
            client.get_registered_model(self.config.registered_model_name)
        except MlflowException:
            client.create_registered_model(
                self.config.registered_model_name,
                tags={"project": self.config.project_tag, "dataset_subset": "FD001"},
                description=(
                    "TurbineGuard FD001 Remaining Useful Life model. Public NASA simulated "
                    "turbofan data; not an industrial deployment."
                ),
            )
        if not self.config.force_new_model_version:
            for item in client.search_model_versions(
                f"name = '{self.config.registered_model_name}'"
            ):
                if item.tags.get(BUNDLE_TAG) == artifacts.champion_sha256:
                    return str(item.version)

        tags = self._version_tags(artifacts, selected_run_id)
        created = client.create_model_version(
            name=self.config.registered_model_name,
            source=model_info.model_uri,
            run_id=selected_run_id,
            tags=tags,
            description=self._version_description(artifacts),
        )
        return str(created.version)

    def _verify_registered_model(
        self, version_value: str, input_example: pd.DataFrame, local_point: np.ndarray
    ) -> float:
        loaded = mlflow.pyfunc.load_model(
            f"models:/{self.config.registered_model_name}/{version_value}"
        )
        result = loaded.predict(input_example)
        if not isinstance(result, pd.DataFrame) or POINT_COLUMN not in result:
            raise MlflowTrackingError("Registered pyfunc returned an invalid output schema.")
        difference = np.abs(result[POINT_COLUMN].to_numpy(dtype="float64") - local_point)
        maximum = float(np.max(difference)) if difference.size else 0.0
        if maximum > 1e-12:
            raise MlflowTrackingError(
                f"Registered prediction differs from local champion (max diff {maximum})."
            )
        return maximum

    def _assign_aliases(
        self, client: MlflowClient, version_value: str, artifacts: TrackingArtifacts
    ) -> dict[str, str]:
        name = self.config.registered_model_name
        previous_champion: str | None = None
        with suppress(MlflowException):
            previous_champion = str(
                client.get_model_version_by_alias(name, self.config.champion_alias).version
            )
        client.set_registered_model_alias(name, self.config.candidate_alias, version_value)
        client.set_registered_model_alias(name, self.config.challenger_alias, version_value)
        if self.config.promote_champion:
            selected = artifacts.selection_report["selected_model"]
            selected_record = next(
                item
                for item in artifacts.selection_report["candidates"]
                if item["candidate_id"] == selected
            )
            if not bool(selected_record["eligible"]):
                raise MlflowTrackingError("Loop 4 selected model is not promotion-eligible.")
            if previous_champion is not None and previous_champion != version_value:
                client.set_registered_model_alias(
                    name, self.config.archived_alias, previous_champion
                )
                client.set_model_version_tag(
                    name,
                    previous_champion,
                    "turbine_guard.champion_displaced_at",
                    datetime.now(UTC).isoformat(),
                )
            client.set_registered_model_alias(name, self.config.champion_alias, version_value)
        return self._aliases(client)

    def _attach_registry_metadata(
        self,
        client: MlflowClient,
        parent_run_id: str,
        selected_run_id: str,
        version_value: str,
        aliases: dict[str, str],
        maximum: float,
        artifacts: TrackingArtifacts,
    ) -> None:
        client.set_tag(parent_run_id, "turbine_guard.registry_version", version_value)
        client.set_tag(
            parent_run_id, "turbine_guard.registry_aliases", json.dumps(aliases, sort_keys=True)
        )
        client.set_tag(parent_run_id, "turbine_guard.max_prediction_difference", str(maximum))
        card = _model_card(
            self.config.registered_model_name,
            version_value,
            parent_run_id,
            selected_run_id,
            artifacts,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model_card.md"
            path.write_text(card, encoding="utf-8")
            client.log_artifact(selected_run_id, str(path), artifact_path="champion")
        client.set_model_version_tag(
            self.config.registered_model_name,
            version_value,
            "turbine_guard.aliases_after_registration",
            json.dumps(aliases, sort_keys=True),
        )

    def _version_tags(self, artifacts: TrackingArtifacts, selected_run_id: str) -> dict[str, str]:
        validation = artifacts.champion_metadata["evaluation_summaries"]["validation"]
        replay = artifacts.champion_metadata["evaluation_summaries"]["replay"]
        official = artifacts.champion_metadata["evaluation_summaries"]["official_test"]
        return {
            BUNDLE_TAG: artifacts.champion_sha256,
            EXECUTION_TAG: artifacts.execution_id,
            "turbine_guard.source_run_id": selected_run_id,
            "git_sha": artifacts.training_manifest.git_commit or "unknown",
            "feature_manifest_sha256": artifacts.training_manifest.feature_manifest_sha256,
            "split_manifest_sha256": artifacts.training_manifest.split_manifest_sha256,
            "evaluation_version": artifacts.training_manifest.evaluation_version,
            "target_configuration": json.dumps(
                artifacts.champion_metadata["target_definition"], sort_keys=True
            ),
            "validation_rmse": str(validation["rmse"]),
            "replay_rmse": str(replay["rmse"]),
            "official_test_rmse": str(official["rmse"]),
            "registration_timestamp": datetime.now(UTC).isoformat(),
            "champion_selection_reason": artifacts.selection_report["rationale"],
        }

    def _version_description(self, artifacts: TrackingArtifacts) -> str:
        return (
            f"Selected by Loop 4 validation-only gates as "
            f"{artifacts.training_manifest.selected_candidate_id}. "
            f"{artifacts.selection_report['rationale']}"
        )

    def _aliases(self, client: MlflowClient) -> dict[str, str]:
        try:
            model = client.get_registered_model(self.config.registered_model_name)
        except MlflowException:
            return {}
        aliases = model.aliases
        if isinstance(aliases, dict):
            return {str(alias): str(version_value) for alias, version_value in aliases.items()}
        return {alias.alias: alias.version for alias in aliases}

    def _child_run_ids(
        self, client: MlflowClient, experiment_id: str, parent_run_id: str
    ) -> dict[str, str]:
        children = client.search_runs(
            [experiment_id],
            filter_string=f"tags.`mlflow.parentRunId` = '{parent_run_id}'",
            max_results=1000,
        )
        return {
            run.data.tags["candidate_id"]: run.info.run_id
            for run in children
            if "candidate_id" in run.data.tags
        }

    @staticmethod
    def _log_generated_json(filename: str, value: Any, artifact_path: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / filename
            path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            mlflow.log_artifact(str(path), artifact_path=artifact_path)


def _candidate_ranks(candidates: tuple[dict[str, Any], ...]) -> dict[str, int]:
    ordered = sorted(
        candidates,
        key=lambda item: (
            float(item["common_domain_metrics"]["rmse"]),
            float(item["common_domain_metrics"]["nasa_score"]),
            str(item["candidate_id"]),
        ),
    )
    return {str(item["candidate_id"]): index for index, item in enumerate(ordered, start=1)}


def _candidate_metrics(
    candidate: dict[str, Any], selection: dict[str, Any], rank: int
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    _add_regression_metrics(metrics, "validation", candidate["regression_metrics"]["row_weighted"])
    _add_regression_metrics(metrics, "validation/common_domain", candidate["common_domain_metrics"])
    _add_alert_metrics(metrics, "validation/critical", candidate["critical_alert_metrics"])
    _add_alert_metrics(metrics, "validation/warning", candidate["warning_alert_metrics"])
    metrics.update(
        {
            "performance/training_seconds": float(candidate["training_seconds"]),
            "performance/inference_latency_ms": float(candidate["prediction_latency_ms"]),
            "performance/model_artifact_size_bytes": float(candidate["model_size_bytes"]),
            "selection/candidate_eligible": float(bool(selection["eligible"])),
            "selection/candidate_rank": float(rank),
        }
    )
    return metrics


def _add_regression_metrics(output: dict[str, float], prefix: str, record: dict[str, Any]) -> None:
    for name in ("mae", "rmse", "r2", "nasa_score"):
        value = record.get(name)
        if value is not None:
            output[f"{prefix}/{name}"] = float(value)


def _add_alert_metrics(output: dict[str, float], prefix: str, record: dict[str, Any]) -> None:
    mapping = {
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "pr_auc": "pr_auc",
        "false_alarms_per_1000_cycles": "false_alarms_per_1000_cycles",
        "mean_first_alert_lead_time": "mean_alert_lead_time",
        "timely_warning_asset_percentage": "timely_warning_percentage",
        "missed_failures": "missed_failures",
    }
    for source, target in mapping.items():
        value = record.get(source)
        if value is not None:
            output[f"{prefix}/{target}"] = float(value)


def _model_card(
    model_name: str,
    version_value: str,
    parent_run_id: str,
    selected_run_id: str,
    artifacts: TrackingArtifacts,
) -> str:
    metadata = artifacts.champion_metadata
    validation = metadata["evaluation_summaries"]["validation"]
    replay = metadata["evaluation_summaries"]["replay"]
    official = metadata["evaluation_summaries"]["official_test"]
    conformal = artifacts.conformal_report
    target = metadata["target_definition"]
    rationale = artifacts.selection_report["rationale"]
    candidate_id = artifacts.training_manifest.selected_candidate_id
    official_line = (
        f"| Official final row | {official['mae']:.4f} | {official['rmse']:.4f} | "
        f"{official['nasa_score']:.4f} |"
    )
    return f"""# TurbineGuard FD001 RUL Model Card

## Purpose and provenance

This model estimates Remaining Useful Life for the public NASA C-MAPSS FD001 simulated turbofan
dataset. Sensors remain anonymous. It is an independent educational system, is not trained on
proprietary industrial data, and has not been deployed in any proprietary industrial system.

## Model and target

- Registered model: `{model_name}` version `{version_value}`
- Candidate: `{candidate_id}`
- Target: `{target["name"]}` with cap `{target["cap"]}`
- Features: {metadata["feature_count"]} ordered current/delta/trailing-window/EWM features
- Preprocessing: {metadata["preprocessing_policy"]}
- Train/validation/calibration/replay split: 70/15/5/10 assets, split by asset

## Validation and calibration

Selection used validation only. The champion rationale was: {rationale}
Split conformal calibration used calibration rows only at nominal coverage
{conformal["calibration"]["coverage"]:.0%}. Rows within an asset are dependent, so formal
trajectory-level exchangeability guarantees do not strictly apply.

## Key metrics

| Role | MAE | RMSE | NASA score |
| --- | ---: | ---: | ---: |
| Validation | {validation["mae"]:.4f} | {validation["rmse"]:.4f} | {validation["nasa_score"]:.4f} |
| Replay | {replay["mae"]:.4f} | {replay["rmse"]:.4f} | {replay["nasa_score"]:.4f} |
{official_line}

Replay empirical interval coverage was {conformal["replay"]["empirical_coverage"]:.3f} with mean
width {conformal["replay"]["average_width"]:.3f}. Maintenance-policy results are simulations in
normalized hypothetical units, not currency or claimed savings.

## Intended and prohibited uses

Intended for reproducible FD001 research, education, and offline pipeline demonstrations. Do not
use it for safety-critical maintenance decisions, real equipment control, claims about physical
sensor meanings, or claims of industrial deployment without domain validation and governance.

## Registry and lineage

- Parent MLflow run: `{parent_run_id}`
- Source candidate run: `{selected_run_id}`
- Git SHA: `{artifacts.training_manifest.git_commit}`
- Raw acquisition manifest SHA-256: `{artifacts.acquisition_manifest_sha256}`
- Validation report SHA-256: `{artifacts.validation_report_sha256}`
- Split manifest SHA-256: `{artifacts.training_manifest.split_manifest_sha256}`
- Feature manifest SHA-256: `{artifacts.training_manifest.feature_manifest_sha256}`
- Training manifest SHA-256: `{artifacts.training_manifest_sha256}`
"""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return value


def _optional_float(value: str | None) -> float | None:
    return None if value is None else float(value)

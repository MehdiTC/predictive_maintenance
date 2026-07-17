"""Thread-safe lazy MLflow champion loading and feature-contract verification."""

import threading
from datetime import UTC, datetime

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from turbine_guard.config.settings import Settings
from turbine_guard.features.manifest import feature_config_from_manifest, load_feature_manifest
from turbine_guard.serving.champion import (
    LoadedChampion,
    ModelMetadata,
    PyFuncModel,
    validate_prediction_output,
)
from turbine_guard.tracking.config import MlflowConfig
from turbine_guard.tracking.lifecycle import aliases, configured_mlflow

__all__ = [
    "ChampionModelLoader",
    "LoadedChampion",
    "ModelMetadata",
    "PyFuncModel",
    "validate_prediction_output",
]


class ChampionModelLoader:
    """Load one champion per process and support explicit future refresh."""

    def __init__(self, settings: Settings) -> None:
        self._config = MlflowConfig.from_settings(settings)
        self._manifest_path = (
            settings.data_dir / "features" / "cmapss" / "FD001" / "feature_manifest.json"
        )
        self._lock = threading.RLock()
        self._cached: LoadedChampion | None = None

    def get(self) -> LoadedChampion:
        """Return the cached champion, loading and validating it once when needed."""
        with self._lock:
            if self._cached is None:
                self._cached = self._load()
            return self._cached

    def refresh(self) -> LoadedChampion:
        """Load and validate before swapping, preserving the working cache on failure."""
        with self._lock:
            replacement = self._load()
            self._cached = replacement
            return replacement

    def check_model(self) -> bool:
        try:
            self.get()
        except Exception:
            return False
        return True

    def check_feature_contract(self) -> bool:
        try:
            loaded = self.get()
        except Exception:
            return False
        return loaded.feature_columns == tuple(
            loaded.model.metadata.get_input_schema().input_names()
        )

    def registry_aliases(self) -> dict[str, str]:
        """Return the live registry alias-to-version assignments."""
        with configured_mlflow(self._config) as client:
            return aliases(client, self._config)

    def _load(self) -> LoadedChampion:
        try:
            manifest = load_feature_manifest(self._manifest_path)
            feature_config = feature_config_from_manifest(manifest)
            expected = tuple(manifest.feature_columns)
            if feature_config.feature_version != manifest.feature_config.feature_version:
                raise ValueError("Feature version is internally inconsistent.")

            previous_tracking = mlflow.get_tracking_uri()
            previous_registry = mlflow.get_registry_uri()
            try:
                mlflow.set_tracking_uri(self._config.tracking_uri)
                mlflow.set_registry_uri(self._config.tracking_uri)
                client = MlflowClient(
                    tracking_uri=self._config.tracking_uri,
                    registry_uri=self._config.tracking_uri,
                )
                version = client.get_model_version_by_alias(
                    self._config.registered_model_name, self._config.champion_alias
                )
                if version.run_id is None:
                    raise ValueError("Champion registry version has no source run ID.")
                model_uri = (
                    f"models:/{self._config.registered_model_name}@{self._config.champion_alias}"
                )
                model = mlflow.pyfunc.load_model(model_uri)
                run = client.get_run(version.run_id)
            finally:
                mlflow.set_tracking_uri(previous_tracking)
                mlflow.set_registry_uri(previous_registry)

            schema = model.metadata.get_input_schema()
            if schema is None:
                raise ValueError("Champion has no declared input schema.")
            model_columns = tuple(schema.input_names())
            if model_columns != expected:
                raise ValueError("Champion input schema does not match the feature manifest.")
            run_feature_version = run.data.tags.get("feature_version")
            if run_feature_version != feature_config.feature_version:
                raise ValueError("Champion feature version does not match the feature manifest.")
            params = run.data.params
            tags = version.tags
            metadata = ModelMetadata(
                model_name=self._config.registered_model_name,
                version=str(version.version),
                alias=self._config.champion_alias,
                source_run_id=str(version.run_id),
                target_definition=run.data.tags.get("target_type", "unknown"),
                rul_cap=_optional_int(params.get("rul_cap")),
                feature_count=len(expected),
                feature_version=feature_config.feature_version,
                validation_rmse=_optional_float(tags.get("validation_rmse")),
                replay_rmse=_optional_float(tags.get("replay_rmse")),
                official_test_rmse=_optional_float(tags.get("official_test_rmse")),
                conformal_coverage_target=_optional_float(params.get("conformal_target_coverage")),
                loaded_at=datetime.now(UTC),
                checksum=tags.get("turbine_guard.champion_bundle_sha256"),
                lineage_id=tags.get("turbine_guard.execution_id"),
                model_family=run.data.tags.get("model_family"),
                git_sha=(tags.get("git_sha") or run.data.tags.get("git_commit_sha")),
                dataset_checksum=(
                    run.data.tags.get("raw_acquisition_manifest_sha256")
                    or run.data.tags.get("validation_report_sha256")
                ),
                feature_manifest_checksum=(
                    tags.get("feature_manifest_sha256")
                    or run.data.tags.get("feature_manifest_sha256")
                ),
            )
            return LoadedChampion(model, metadata, feature_config, expected)
        except (MlflowException, OSError, ValueError, KeyError) as exc:
            raise RuntimeError("MLflow champion could not be loaded and verified.") from exc


def _optional_float(value: str | None) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: str | None) -> int | None:
    return None if value in (None, "None", "null") else int(value)

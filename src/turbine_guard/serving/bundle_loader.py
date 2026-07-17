"""Champion serving from a restored deployment bundle, without MLflow.

The free public demo loads the checksum-verified exported champion snapshot
directly from the restored data directory. Registry identity and aliases
come from the deployment manifest captured at export time; nothing here can
mutate a model registry.
"""

import threading
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from turbine_guard.config.settings import Settings
from turbine_guard.deployment.manifest import (
    CHAMPION_RELATIVE_PATH,
    FEATURE_MANIFEST_RELATIVE_PATH,
    MANIFEST_FILENAME,
    BundleError,
    DeploymentBundleManifest,
    load_bundle_manifest,
)
from turbine_guard.features.manifest import feature_config_from_manifest, load_feature_manifest
from turbine_guard.modeling.artifacts import load_joblib, sha256_path
from turbine_guard.modeling.pipeline import ModelBundle
from turbine_guard.serving.champion import (
    EXPORTED_SNAPSHOT_SOURCE,
    LoadedChampion,
    ModelMetadata,
)


class _BundleInputSchema:
    """Ordered feature names shaped like an MLflow input schema."""

    def __init__(self, columns: tuple[str, ...]) -> None:
        self._columns = columns

    def input_names(self) -> list[str]:
        return list(self._columns)


class _BundleModelInfo:
    """Duck-typed stand-in for MLflow pyfunc metadata."""

    def __init__(self, columns: tuple[str, ...]) -> None:
        self._schema = _BundleInputSchema(columns)

    def get_input_schema(self) -> _BundleInputSchema:
        return self._schema


class BundleServingModel:
    """Adapt a verified ``ModelBundle`` to the shared ``PyFuncModel`` behavior."""

    def __init__(self, bundle: ModelBundle) -> None:
        self._bundle = bundle
        self.metadata = _BundleModelInfo(bundle.feature_columns)

    def predict(self, model_input: pd.DataFrame) -> pd.DataFrame:
        if tuple(model_input.columns) != self._bundle.feature_columns:
            raise ValueError("Prediction columns do not match the ordered Loop 3 feature manifest.")
        return self._bundle.predict_rich(model_input)


class DeploymentBundleLoader:
    """Load one exported champion snapshot per process with explicit refresh."""

    def __init__(self, settings: Settings) -> None:
        self._data_dir: Path = settings.data_dir
        self._lock = threading.RLock()
        self._cached: LoadedChampion | None = None
        self._manifest: DeploymentBundleManifest | None = None

    def get(self) -> LoadedChampion:
        """Return the cached champion, loading and verifying it once when needed."""
        with self._lock:
            if self._cached is None:
                self._cached = self._load()
            return self._cached

    def refresh(self) -> LoadedChampion:
        """Load and verify before swapping, preserving the working cache on failure."""
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
        """Return the alias assignments captured from MLflow at export time."""
        self.get()
        if self._manifest is None:
            return {}
        return dict(self._manifest.aliases)

    def _load(self) -> LoadedChampion:
        try:
            manifest = load_bundle_manifest(self._data_dir / MANIFEST_FILENAME)

            feature_manifest_path = self._data_dir / FEATURE_MANIFEST_RELATIVE_PATH
            actual_feature_sha = sha256_path(feature_manifest_path)
            if actual_feature_sha != manifest.feature_manifest_sha256:
                raise BundleError(
                    "Restored feature manifest does not match the deployment manifest checksum."
                )
            feature_manifest = load_feature_manifest(feature_manifest_path)
            feature_config = feature_config_from_manifest(feature_manifest)
            expected = tuple(feature_manifest.feature_columns)
            if feature_config.feature_version != manifest.feature_version:
                raise BundleError(
                    "Deployment manifest feature version does not match the feature manifest."
                )
            if len(expected) != manifest.feature_count:
                raise BundleError(
                    "Deployment manifest feature count does not match the feature manifest."
                )

            champion_path = self._data_dir / CHAMPION_RELATIVE_PATH
            actual_champion_sha = sha256_path(champion_path)
            if actual_champion_sha != manifest.champion_bundle_sha256:
                raise BundleError(
                    "Restored champion artifact does not match the deployment manifest "
                    "checksum; refusing to deserialize an unverified artifact."
                )
            bundle = load_joblib(champion_path)
            if not isinstance(bundle, ModelBundle):
                raise BundleError("Restored champion is not a TurbineGuard ModelBundle.")
            if bundle.feature_columns != expected:
                raise BundleError(
                    "Restored champion feature contract does not match the feature manifest."
                )

            metadata = ModelMetadata(
                model_name=manifest.registered_model_name,
                version=manifest.registry_version,
                alias=manifest.champion_alias,
                source_run_id=manifest.source_run_id,
                target_definition=manifest.target_definition,
                rul_cap=manifest.rul_cap,
                feature_count=manifest.feature_count,
                feature_version=manifest.feature_version,
                validation_rmse=manifest.validation_rmse,
                replay_rmse=manifest.replay_rmse,
                official_test_rmse=manifest.official_test_rmse,
                conformal_coverage_target=manifest.conformal_coverage_target,
                loaded_at=datetime.now(UTC),
                checksum=manifest.champion_bundle_sha256,
                lineage_id=manifest.lineage_id,
                model_family=manifest.model_family,
                git_sha=manifest.git_commit,
                dataset_checksum=manifest.dataset_checksum,
                feature_manifest_checksum=manifest.feature_manifest_sha256,
                registry_source=EXPORTED_SNAPSHOT_SOURCE,
            )
            loaded = LoadedChampion(BundleServingModel(bundle), metadata, feature_config, expected)
            self._manifest = manifest
            return loaded
        except (BundleError, OSError, ValueError, KeyError) as exc:
            raise RuntimeError(
                "Deployment bundle champion could not be loaded and verified."
            ) from exc

"""Typed configuration for Loop 5 MLflow tracking and registration."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from turbine_guard.config.settings import Settings


@dataclass(frozen=True)
class MlflowConfig:
    """Configuration for one optional MLflow tracking operation."""

    tracking_uri: str
    experiment_name: str
    registered_model_name: str
    artifact_location: str | None
    registration_enabled: bool
    promote_champion: bool
    candidate_alias: str
    challenger_alias: str
    champion_alias: str
    archived_alias: str
    run_name_prefix: str
    project_tag: str
    environment: str
    force_new_run: bool = False
    force_new_model_version: bool = False

    def __post_init__(self) -> None:
        required = {
            "tracking_uri": self.tracking_uri,
            "experiment_name": self.experiment_name,
            "registered_model_name": self.registered_model_name,
            "candidate_alias": self.candidate_alias,
            "challenger_alias": self.challenger_alias,
            "champion_alias": self.champion_alias,
            "archived_alias": self.archived_alias,
            "run_name_prefix": self.run_name_prefix,
            "project_tag": self.project_tag,
            "environment": self.environment,
        }
        empty = sorted(name for name, value in required.items() if not value.strip())
        if empty:
            raise ValueError(f"MLflow configuration values must not be empty: {empty}.")
        aliases = (
            self.candidate_alias,
            self.challenger_alias,
            self.champion_alias,
            self.archived_alias,
        )
        if len(set(aliases)) != len(aliases):
            raise ValueError("MLflow registry aliases must be distinct.")
        if self.promote_champion and not self.registration_enabled:
            raise ValueError("Champion promotion requires model registration to be enabled.")
        if self.force_new_model_version and not self.force_new_run:
            raise ValueError("A forced new model version requires a forced new MLflow run.")

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        force_new_run: bool = False,
        force_new_model_version: bool = False,
    ) -> "MlflowConfig":
        """Build tracking configuration from the repository's typed settings."""
        artifact_location = settings.mlflow_artifact_location
        if artifact_location is None:
            artifact_location = str(settings.data_dir / "mlflow" / "artifacts")
        return cls(
            tracking_uri=settings.mlflow_tracking_uri,
            experiment_name=settings.mlflow_experiment_name,
            registered_model_name=settings.mlflow_registered_model_name,
            artifact_location=artifact_location,
            registration_enabled=settings.mlflow_registration_enabled,
            promote_champion=settings.mlflow_promote_champion,
            candidate_alias=settings.mlflow_candidate_alias,
            challenger_alias=settings.mlflow_challenger_alias,
            champion_alias=settings.mlflow_champion_alias,
            archived_alias=settings.mlflow_archived_alias,
            run_name_prefix=settings.mlflow_run_name_prefix,
            project_tag=settings.mlflow_project_tag,
            environment=settings.environment.value,
            force_new_run=force_new_run,
            force_new_model_version=force_new_model_version,
        )

    def prepare_local_store(self) -> None:
        """Create parent directories required by relative local SQLite/file stores."""
        sqlite_prefix = "sqlite:///"
        if self.tracking_uri.startswith(sqlite_prefix):
            database = self.tracking_uri.removeprefix(sqlite_prefix)
            if database and database != ":memory:":
                Path(database).expanduser().parent.mkdir(parents=True, exist_ok=True)
        if self.artifact_location is not None and _is_local_path(self.artifact_location):
            _local_path(self.artifact_location).expanduser().mkdir(parents=True, exist_ok=True)

    def resolved_artifact_location(self) -> str | None:
        """Return a local file URI or preserve an explicitly remote artifact URI."""
        if self.artifact_location is None:
            return None
        if _is_local_path(self.artifact_location):
            return _local_path(self.artifact_location).expanduser().resolve().as_uri()
        return self.artifact_location

    def behavior_id(self) -> str:
        """Stable identity for registry behavior, excluding explicit force controls."""
        record = {
            "registered_model_name": self.registered_model_name,
            "registration_enabled": self.registration_enabled,
            "promote_champion": self.promote_champion,
            "aliases": {
                "candidate": self.candidate_alias,
                "challenger": self.challenger_alias,
                "champion": self.champion_alias,
                "archived": self.archived_alias,
            },
        }
        encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def _is_local_path(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme == "" or parsed.scheme == "file"


def _local_path(value: str) -> Path:
    parsed = urlparse(value)
    return Path(parsed.path if parsed.scheme == "file" else value)

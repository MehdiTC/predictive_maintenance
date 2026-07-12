"""Typed, environment-based application settings."""

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from turbine_guard.data.acquisition import DEFAULT_SOURCE_URL

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Environment(StrEnum):
    """Deployment environment the application runs in."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Every field can be overridden with an environment variable using the
    ``TURBINE_GUARD_`` prefix, for example ``TURBINE_GUARD_LOG_LEVEL=DEBUG``.
    Values from a local ``.env`` file are read when present; real ``.env``
    files are gitignored and must never be committed.
    """

    model_config = SettingsConfigDict(
        env_prefix="TURBINE_GUARD_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "turbine-guard"
    environment: Environment = Environment.DEVELOPMENT
    log_level: LogLevel = "INFO"

    data_dir: Path = Path("data")
    """Base directory for the data layers (raw, manifests, ...)."""

    cmapss_source_url: str = DEFAULT_SOURCE_URL
    """Archive URL for the NASA C-MAPSS dataset; https:// or file://."""

    mlflow_tracking_uri: str = "sqlite:///data/mlflow/mlflow.db"
    """MLflow tracking/registry backend; SQLite is the safe local default."""

    mlflow_experiment_name: str = "TurbineGuard-FD001-Offline-Modeling"
    mlflow_registered_model_name: str = "TurbineGuard-FD001-RUL"
    mlflow_artifact_location: str | None = None
    mlflow_registration_enabled: bool = True
    mlflow_promote_champion: bool = True
    mlflow_candidate_alias: str = "candidate"
    mlflow_challenger_alias: str = "challenger"
    mlflow_champion_alias: str = "champion"
    mlflow_archived_alias: str = "archived"
    mlflow_run_name_prefix: str = "fd001-offline"
    mlflow_project_tag: str = "turbine-guard"

    @field_validator("environment", mode="before")
    @classmethod
    def _normalize_environment(cls, value: object) -> object:
        """Accept any casing for the environment name."""
        if isinstance(value, str):
            return value.lower()
        return value

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalize_log_level(cls, value: object) -> object:
        """Accept any casing for the log level name."""
        if isinstance(value, str):
            return value.upper()
        return value

    @field_validator(
        "mlflow_tracking_uri",
        "mlflow_experiment_name",
        "mlflow_registered_model_name",
        "mlflow_candidate_alias",
        "mlflow_challenger_alias",
        "mlflow_champion_alias",
        "mlflow_archived_alias",
        "mlflow_run_name_prefix",
        "mlflow_project_tag",
    )
    @classmethod
    def _non_empty_mlflow_value(cls, value: str) -> str:
        """Reject ambiguous empty MLflow identifiers and URIs."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("MLflow configuration values must not be empty.")
        return normalized


@lru_cache
def get_settings() -> Settings:
    """Return the application settings, loaded once per process."""
    return Settings()

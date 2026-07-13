"""Typed, environment-based application settings."""

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError

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

    database_url: str = "postgresql+psycopg://localhost:5432/turbine_guard"
    """Operational PostgreSQL URL; deliberately separate from MLflow's backend."""

    database_test_url: str | None = None
    database_pool_size: int = 5
    database_max_overflow: int = 10
    database_pool_timeout_seconds: float = 30.0
    database_pool_recycle_seconds: int = 1800
    database_connect_timeout_seconds: int = 5
    database_statement_timeout_ms: int = 30_000
    database_echo: bool = False

    online_inference_enabled: bool = True
    model_preload_enabled: bool = True
    asset_stale_after_seconds: int = 300
    api_default_page_size: int = 50
    api_max_page_size: int = 200
    api_prediction_trend_size: int = 20
    api_max_request_bytes: int = 1_048_576
    api_docs_enabled: bool = True
    cors_allowed_origins: tuple[str, ...] = ()
    trusted_hosts: tuple[str, ...] = ("localhost", "127.0.0.1", "testserver")

    replay_api_base_url: str = "http://127.0.0.1:8000"
    """Base URL of the running Loop 7 inference API the replay client targets."""

    replay_cycle_delay_seconds: float = 1.0
    """Default wait between cycles in continuous replay mode."""

    replay_simulated_cycle_duration_seconds: float = 1.0
    """Simulated wall-clock length of one C-MAPSS cycle (a documented simulation
    assumption, not a claim that one cycle equals any real duration)."""

    replay_lease_seconds: int = 120
    """How long one worker's advance claim on a replay run stays exclusive."""

    replay_http_timeout_seconds: float = 30.0
    replay_max_send_attempts: int = 5
    replay_retry_backoff_seconds: float = 0.5

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

    @field_validator("database_url", "database_test_url")
    @classmethod
    def _postgresql_url(cls, value: str | None) -> str | None:
        """Accept only explicit psycopg PostgreSQL URLs for operational storage."""
        if value is None:
            return None
        normalized = value.strip()
        if not normalized.startswith("postgresql+psycopg://"):
            raise ValueError("Operational database URLs must use postgresql+psycopg://.")
        try:
            parsed = make_url(normalized)
        except ArgumentError as exc:
            raise ValueError("Operational database URL is malformed.") from exc
        if not parsed.database:
            raise ValueError("Operational database URL must include a database name.")
        return normalized

    @field_validator(
        "database_pool_size",
        "database_pool_recycle_seconds",
        "database_connect_timeout_seconds",
        "database_statement_timeout_ms",
    )
    @classmethod
    def _positive_database_integer(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("Database pool and timeout settings must be positive.")
        return value

    @field_validator("database_max_overflow")
    @classmethod
    def _non_negative_database_integer(cls, value: int) -> int:
        if value < 0:
            raise ValueError("Database max overflow must be non-negative.")
        return value

    @field_validator("database_pool_timeout_seconds")
    @classmethod
    def _positive_database_float(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("Database pool timeout must be positive.")
        return value

    @field_validator(
        "asset_stale_after_seconds",
        "api_default_page_size",
        "api_max_page_size",
        "api_prediction_trend_size",
        "api_max_request_bytes",
    )
    @classmethod
    def _positive_online_integer(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("Online API limits and time thresholds must be positive.")
        return value

    @field_validator("cors_allowed_origins", "trusted_hosts")
    @classmethod
    def _non_empty_host_values(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item.strip() for item in value):
            raise ValueError("CORS origins and trusted hosts must not contain empty values.")
        return value

    @field_validator("replay_api_base_url")
    @classmethod
    def _replay_base_url(cls, value: str) -> str:
        normalized = value.strip().rstrip("/")
        if not normalized.startswith(("http://", "https://")):
            raise ValueError("Replay API base URL must be an http:// or https:// URL.")
        return normalized

    @field_validator("replay_cycle_delay_seconds", "replay_retry_backoff_seconds")
    @classmethod
    def _non_negative_replay_seconds(cls, value: float) -> float:
        if value < 0:
            raise ValueError("Replay delay and backoff values must be non-negative.")
        return value

    @field_validator(
        "replay_simulated_cycle_duration_seconds",
        "replay_lease_seconds",
        "replay_http_timeout_seconds",
        "replay_max_send_attempts",
    )
    @classmethod
    def _positive_replay_value(cls, value: float) -> float:
        if value <= 0:
            raise ValueError(
                "Replay duration, lease, timeout, and attempt values must be positive."
            )
        return value

    @model_validator(mode="after")
    def _valid_page_limits(self) -> Self:
        if self.api_default_page_size > self.api_max_page_size:
            raise ValueError("Default API page size must not exceed the maximum.")
        if self.api_max_page_size > 200:
            raise ValueError("Loop 7 API page size cannot exceed 200.")
        return self


@lru_cache
def get_settings() -> Settings:
    """Return the application settings, loaded once per process."""
    return Settings()

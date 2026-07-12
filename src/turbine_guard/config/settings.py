"""Typed, environment-based application settings."""

from enum import StrEnum
from functools import lru_cache
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

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


@lru_cache
def get_settings() -> Settings:
    """Return the application settings, loaded once per process."""
    return Settings()

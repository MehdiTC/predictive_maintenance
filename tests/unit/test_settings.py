"""Tests for typed, environment-based settings."""

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from turbine_guard.config.settings import Environment, Settings, get_settings
from turbine_guard.data.acquisition import DEFAULT_SOURCE_URL

ENV_PREFIX = "TURBINE_GUARD_"


@pytest.fixture(autouse=True)
def _isolated_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Strip ambient TurbineGuard variables and hide any local .env file."""
    for key in list(os.environ):
        if key.startswith(ENV_PREFIX):
            monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)


def test_defaults() -> None:
    settings = Settings()
    assert settings.app_name == "turbine-guard"
    assert settings.environment is Environment.DEVELOPMENT
    assert settings.log_level == "INFO"
    assert settings.data_dir == Path("data")
    assert settings.cmapss_source_url == DEFAULT_SOURCE_URL
    assert settings.mlflow_tracking_uri == "sqlite:///data/mlflow/mlflow.db"
    assert settings.mlflow_experiment_name == "TurbineGuard-FD001-Offline-Modeling"
    assert settings.mlflow_registered_model_name == "TurbineGuard-FD001-RUL"
    assert settings.mlflow_candidate_alias == "candidate"
    assert settings.mlflow_champion_alias == "champion"


def test_data_settings_read_environment_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TURBINE_GUARD_DATA_DIR", "/somewhere/else")
    monkeypatch.setenv("TURBINE_GUARD_CMAPSS_SOURCE_URL", "file:///tmp/archive.zip")

    settings = Settings()

    assert settings.data_dir == Path("/somewhere/else")
    assert settings.cmapss_source_url == "file:///tmp/archive.zip"


def test_mlflow_settings_read_environment_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TURBINE_GUARD_MLFLOW_TRACKING_URI", "sqlite:///custom.db")
    monkeypatch.setenv("TURBINE_GUARD_MLFLOW_EXPERIMENT_NAME", "custom-experiment")
    monkeypatch.setenv("TURBINE_GUARD_MLFLOW_REGISTERED_MODEL_NAME", "custom-model")
    monkeypatch.setenv("TURBINE_GUARD_MLFLOW_REGISTRATION_ENABLED", "false")
    monkeypatch.setenv("TURBINE_GUARD_MLFLOW_PROMOTE_CHAMPION", "false")

    settings = Settings()

    assert settings.mlflow_tracking_uri == "sqlite:///custom.db"
    assert settings.mlflow_experiment_name == "custom-experiment"
    assert settings.mlflow_registered_model_name == "custom-model"
    assert settings.mlflow_registration_enabled is False
    assert settings.mlflow_promote_champion is False


def test_operational_database_settings_are_separate_from_mlflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_url = "postgresql+psycopg://user:secret@localhost:5432/turbine_guard_test"
    monkeypatch.setenv("TURBINE_GUARD_DATABASE_URL", database_url)
    monkeypatch.setenv("TURBINE_GUARD_DATABASE_POOL_SIZE", "3")

    settings = Settings()

    assert settings.database_url == database_url
    assert settings.database_pool_size == 3
    assert settings.mlflow_tracking_uri == "sqlite:///data/mlflow/mlflow.db"


@pytest.mark.parametrize(
    "url",
    [
        "sqlite:///operational.db",
        "postgresql://localhost/database",
        "postgresql+psycopg://localhost",
        "postgresql+psycopg://[broken",
        "",
        "https://db",
    ],
)
def test_invalid_operational_database_url_is_rejected(
    monkeypatch: pytest.MonkeyPatch, url: str
) -> None:
    monkeypatch.setenv("TURBINE_GUARD_DATABASE_URL", url)
    with pytest.raises(ValidationError):
        Settings()


def test_empty_mlflow_setting_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TURBINE_GUARD_MLFLOW_EXPERIMENT_NAME", "  ")
    with pytest.raises(ValidationError, match="must not be empty"):
        Settings()


def test_reads_prefixed_environment_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TURBINE_GUARD_APP_NAME", "turbine-guard-test")
    monkeypatch.setenv("TURBINE_GUARD_ENVIRONMENT", "production")
    monkeypatch.setenv("TURBINE_GUARD_LOG_LEVEL", "ERROR")

    settings = Settings()

    assert settings.app_name == "turbine-guard-test"
    assert settings.environment is Environment.PRODUCTION
    assert settings.log_level == "ERROR"


def test_log_level_accepts_any_casing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TURBINE_GUARD_LOG_LEVEL", "debug")
    assert Settings().log_level == "DEBUG"


def test_environment_accepts_any_casing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TURBINE_GUARD_ENVIRONMENT", "TESTING")
    assert Settings().environment is Environment.TESTING


def test_invalid_log_level_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TURBINE_GUARD_LOG_LEVEL", "verbose")
    with pytest.raises(ValidationError):
        Settings()


def test_invalid_environment_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TURBINE_GUARD_ENVIRONMENT", "staging")
    with pytest.raises(ValidationError):
        Settings()


def test_settings_load_from_dotenv_file(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("TURBINE_GUARD_LOG_LEVEL=CRITICAL\n", encoding="utf-8")
    assert Settings().log_level == "CRITICAL"


def test_get_settings_returns_cached_instance() -> None:
    get_settings.cache_clear()
    try:
        assert get_settings() is get_settings()
    finally:
        get_settings.cache_clear()

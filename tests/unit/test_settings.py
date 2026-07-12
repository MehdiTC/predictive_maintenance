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


def test_data_settings_read_environment_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TURBINE_GUARD_DATA_DIR", "/somewhere/else")
    monkeypatch.setenv("TURBINE_GUARD_CMAPSS_SOURCE_URL", "file:///tmp/archive.zip")

    settings = Settings()

    assert settings.data_dir == Path("/somewhere/else")
    assert settings.cmapss_source_url == "file:///tmp/archive.zip"


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

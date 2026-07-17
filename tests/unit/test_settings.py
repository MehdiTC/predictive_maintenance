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
    assert settings.online_inference_enabled is True
    assert settings.model_preload_enabled is True
    assert settings.api_default_page_size == 50
    assert settings.api_max_page_size == 200
    assert settings.api_host == "127.0.0.1"
    assert settings.api_port == 8000
    assert settings.cors_allowed_origins == ()
    assert settings.monitoring_window_days == 30
    assert settings.retraining_min_new_assets == 5
    assert settings.retraining_min_holdout_assets == 2
    assert settings.promotion_approval_required is True


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


def test_render_database_url_is_normalized_to_psycopg() -> None:
    settings = Settings(database_url="postgresql://user:secret@render-internal/db")
    assert settings.database_url == "postgresql+psycopg://user:secret@render-internal/db"


def test_replay_control_security_configuration_is_enforced() -> None:
    with pytest.raises(ValidationError, match="APPLICATION_SECRET"):
        Settings(replay_controls_enabled=True, public_demo_mode=True)
    with pytest.raises(ValidationError, match="REPLAY_ADMIN_TOKEN"):
        Settings(
            replay_controls_enabled=True,
            public_demo_mode=False,
            application_secret="configured",
        )
    configured = Settings(
        replay_controls_enabled=True,
        public_demo_mode=False,
        application_secret="configured",
        replay_admin_token="operator-token",
    )
    assert configured.replay_controls_enabled is True


def test_dashboard_sensor_selection_rejects_invented_channels() -> None:
    with pytest.raises(ValidationError, match="anonymous sensor"):
        Settings(dashboard_default_sensor_columns=("temperature",))


def test_online_settings_read_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TURBINE_GUARD_ONLINE_INFERENCE_ENABLED", "false")
    monkeypatch.setenv("TURBINE_GUARD_MODEL_PRELOAD_ENABLED", "false")
    monkeypatch.setenv("TURBINE_GUARD_API_HOST", "0.0.0.0")
    monkeypatch.setenv("TURBINE_GUARD_API_PORT", "8080")
    monkeypatch.setenv("TURBINE_GUARD_CORS_ALLOWED_ORIGINS", '["https://example.test"]')
    monkeypatch.setenv("TURBINE_GUARD_TRUSTED_HOSTS", '["api.example.test"]')
    settings = Settings()
    assert settings.online_inference_enabled is False
    assert settings.model_preload_enabled is False
    assert settings.api_host == "0.0.0.0"
    assert settings.api_port == 8080
    assert settings.cors_allowed_origins == ("https://example.test",)
    assert settings.trusted_hosts == ("api.example.test",)


def test_online_page_limits_are_validated() -> None:
    with pytest.raises(ValidationError, match="must not exceed"):
        Settings(api_default_page_size=100, api_max_page_size=50)
    with pytest.raises(ValidationError, match="API host"):
        Settings(api_host=" ")
    with pytest.raises(ValidationError, match="65535"):
        Settings(api_port=65_536)


def test_lifecycle_thresholds_are_validated() -> None:
    with pytest.raises(ValidationError, match="PSI warning"):
        Settings(monitoring_psi_warning=0.3, monitoring_psi_detected=0.2)
    with pytest.raises(ValidationError, match=r"in \[0, 1\]"):
        Settings(retraining_holdout_fraction=1.1)
    with pytest.raises(ValidationError, match="must be positive"):
        Settings(retraining_min_new_assets=0)


@pytest.mark.parametrize(
    "url",
    [
        "sqlite:///operational.db",
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


def test_model_source_defaults_to_mlflow_and_reads_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert Settings().model_source == "mlflow"
    monkeypatch.setenv(f"{ENV_PREFIX}MODEL_SOURCE", "deployment_bundle")
    assert Settings().model_source == "deployment_bundle"


def test_deployment_bundle_url_scheme_is_validated() -> None:
    accepted = Settings(
        deployment_bundle_url="https://example.org/bundle.tar.gz",
        deployment_bundle_sha256="a" * 64,
    )
    assert accepted.deployment_bundle_url == "https://example.org/bundle.tar.gz"
    with pytest.raises(ValidationError, match="https:// or file://"):
        Settings(
            deployment_bundle_url="http://example.org/bundle.tar.gz",
            deployment_bundle_sha256="a" * 64,
        )


def test_deployment_bundle_sha256_must_be_hexadecimal() -> None:
    with pytest.raises(ValidationError, match="64 hexadecimal"):
        Settings(
            deployment_bundle_url="https://example.org/bundle.tar.gz",
            deployment_bundle_sha256="not-a-checksum",
        )


def test_deployment_bundle_url_and_pin_are_required_together() -> None:
    with pytest.raises(ValidationError, match="configured together"):
        Settings(deployment_bundle_url="https://example.org/bundle.tar.gz")
    with pytest.raises(ValidationError, match="configured together"):
        Settings(deployment_bundle_sha256="a" * 64)

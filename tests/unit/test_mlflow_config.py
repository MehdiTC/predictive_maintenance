"""Typed MLflow configuration and local-store tests."""

from dataclasses import replace
from pathlib import Path

import pytest

from turbine_guard.config.settings import Environment, Settings
from turbine_guard.tracking.config import MlflowConfig


def config(
    tmp_path: Path,
    *,
    registration_enabled: bool = True,
    promote_champion: bool = True,
    challenger_alias: str = "challenger",
) -> MlflowConfig:
    return MlflowConfig(
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        experiment_name="experiment",
        registered_model_name="model",
        artifact_location=str(tmp_path / "artifacts"),
        registration_enabled=registration_enabled,
        promote_champion=promote_champion,
        candidate_alias="candidate",
        challenger_alias=challenger_alias,
        champion_alias="champion",
        archived_alias="archived",
        run_name_prefix="fd001",
        project_tag="turbine-guard",
        environment="testing",
    )


def test_from_settings_and_local_defaults(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data", environment=Environment.TESTING)
    result = MlflowConfig.from_settings(settings)
    assert result.artifact_location == str(tmp_path / "data" / "mlflow" / "artifacts")
    assert result.registered_model_name == "TurbineGuard-FD001-RUL"
    assert result.champion_alias == "champion"


def test_prepare_local_sqlite_and_artifact_store(tmp_path: Path) -> None:
    result = config(tmp_path)
    result.prepare_local_store()
    assert (tmp_path / "artifacts").is_dir()
    assert (tmp_path / "mlflow.db").parent.is_dir()
    assert result.resolved_artifact_location() == (tmp_path / "artifacts").as_uri()


def test_aliases_must_be_distinct(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="distinct"):
        config(tmp_path, challenger_alias="candidate")


def test_promotion_requires_registration(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires"):
        config(tmp_path, registration_enabled=False, promote_champion=True)


def test_forced_version_requires_forced_run(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="forced new MLflow run"):
        replace(config(tmp_path), force_new_model_version=True)

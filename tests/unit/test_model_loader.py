"""MLflow champion cache and contract-verification tests without a real registry."""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from turbine_guard.config.settings import Settings
from turbine_guard.serving.model_loader import ChampionModelLoader, validate_prediction_output


class FakeSchema:
    def __init__(self, names: tuple[str, ...]) -> None:
        self._names = names

    def input_names(self) -> list[str]:
        return list(self._names)


class FakeModel:
    def __init__(self, columns: tuple[str, ...]) -> None:
        self.metadata = SimpleNamespace(get_input_schema=lambda: FakeSchema(columns))

    def predict(self, model_input: pd.DataFrame) -> pd.DataFrame:
        del model_input
        return pd.DataFrame(
            {
                "predicted_rul": [40.0],
                "lower_rul": [30.0],
                "upper_rul": [50.0],
                "risk_level": ["warning"],
            }
        )


def _manifest() -> SimpleNamespace:
    record = SimpleNamespace(
        feature_version="1",
        source_columns=("sensor_01",),
        families=("current",),
        windows=(5,),
        ewm_spans=(5,),
        min_periods=1,
    )
    return SimpleNamespace(feature_config=record, feature_columns=("sensor_01_current",))


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch, *, model_columns: tuple[str, ...] = ("sensor_01_current",)
) -> list[str]:
    calls: list[str] = []
    version = SimpleNamespace(
        version="7",
        run_id="run-7",
        tags={
            "validation_rmse": "1.0",
            "replay_rmse": "2.0",
            "official_test_rmse": "3.0",
            "turbine_guard.champion_bundle_sha256": "abc",
            "turbine_guard.execution_id": "lineage",
        },
    )
    run = SimpleNamespace(
        data=SimpleNamespace(
            tags={"feature_version": "1", "target_type": "capped_125"},
            params={"rul_cap": "125", "conformal_target_coverage": "0.9"},
        )
    )

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def get_model_version_by_alias(self, name: str, alias: str) -> SimpleNamespace:
            calls.append(f"{name}@{alias}")
            return version

        def get_run(self, run_id: str) -> SimpleNamespace:
            assert run_id == "run-7"
            return run

    monkeypatch.setattr(
        "turbine_guard.serving.model_loader.load_feature_manifest", lambda _: _manifest()
    )
    monkeypatch.setattr("turbine_guard.serving.model_loader.MlflowClient", FakeClient)
    monkeypatch.setattr(
        "turbine_guard.serving.model_loader.mlflow.pyfunc.load_model",
        lambda uri: calls.append(uri) or FakeModel(model_columns),
    )
    return calls


def test_loader_uses_champion_alias_caches_and_refreshes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _install_fakes(monkeypatch)
    loader = ChampionModelLoader(Settings(data_dir=tmp_path, online_inference_enabled=False))
    first = loader.get()
    assert loader.get() is first
    assert first.metadata.version == "7"
    assert first.metadata.rul_cap == 125
    assert calls.count("TurbineGuard-FD001-RUL@champion") == 1
    assert calls.count("models:/TurbineGuard-FD001-RUL@champion") == 1
    assert loader.refresh() is not first
    assert calls.count("models:/TurbineGuard-FD001-RUL@champion") == 2


def test_loader_rejects_feature_schema_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fakes(monkeypatch, model_columns=("wrong",))
    loader = ChampionModelLoader(Settings(data_dir=tmp_path, online_inference_enabled=False))
    with pytest.raises(RuntimeError, match="could not be loaded"):
        loader.get()
    assert loader.check_model() is False


def test_prediction_output_validation() -> None:
    assert validate_prediction_output(FakeModel(("x",)).predict(pd.DataFrame())) == (
        40.0,
        30.0,
        50.0,
        "warning",
    )
    invalid = FakeModel(("x",)).predict(pd.DataFrame())
    invalid.loc[0, "lower_rul"] = 45.0
    with pytest.raises(ValueError, match="interval"):
        validate_prediction_output(invalid)

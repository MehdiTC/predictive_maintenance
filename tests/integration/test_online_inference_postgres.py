"""HTTP-to-PostgreSQL online inference integration with injected and real champions."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session, sessionmaker

from turbine_guard.api.app import create_app
from turbine_guard.config.settings import Environment, Settings
from turbine_guard.data.schema import OPERATING_SETTING_COLUMNS, SENSOR_COLUMNS
from turbine_guard.database.models import Asset, Prediction, SensorReading
from turbine_guard.features.config import FeatureConfig
from turbine_guard.observability.metrics import OnlineMetrics
from turbine_guard.services.inference import OnlineInferenceService
from turbine_guard.serving.model_loader import (
    ChampionModelLoader,
    LoadedChampion,
    ModelMetadata,
)

pytestmark = pytest.mark.postgres
REPO_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_TRAIN = REPO_DATA_DIR / "processed" / "cmapss" / "FD001" / "train_FD001.parquet"
SPLIT_MANIFEST = REPO_DATA_DIR / "features" / "cmapss" / "FD001" / "split_manifest.json"


class FakeModel:
    def __init__(self, columns: tuple[str, ...], *, fail: bool = False) -> None:
        self.columns = columns
        self.fail = fail
        self.calls = 0
        self.metadata = SimpleNamespace(
            get_input_schema=lambda: SimpleNamespace(input_names=lambda: list(columns))
        )

    def predict(self, model_input: pd.DataFrame) -> pd.DataFrame:
        self.calls += 1
        if self.fail:
            raise RuntimeError("forced model failure")
        assert tuple(model_input.columns) == self.columns
        point = max(0.0, 100.0 - float(model_input.iloc[0]["sensor_01_current"]))
        return pd.DataFrame(
            {
                "predicted_rul": [point],
                "lower_rul": [max(0.0, point - 5)],
                "upper_rul": [point + 5],
                "risk_level": [
                    "critical" if point <= 30 else "warning" if point <= 50 else "healthy"
                ],
            }
        )


class FakeLoader:
    def __init__(self, *, fail: bool = False) -> None:
        config = FeatureConfig()
        from turbine_guard.features.builder import FeatureBuilder

        columns = FeatureBuilder(config).feature_columns()
        self.model = FakeModel(columns, fail=fail)
        self.loaded = LoadedChampion(
            model=self.model,
            metadata=_metadata("1"),
            feature_config=config,
            feature_columns=columns,
        )

    def get(self) -> LoadedChampion:
        return self.loaded

    def set_version(self, version: str) -> None:
        self.loaded = replace(self.loaded, metadata=_metadata(version))


def _metadata(version: str) -> ModelMetadata:
    from datetime import UTC, datetime

    return ModelMetadata(
        model_name="fake-rul",
        version=version,
        alias="champion",
        source_run_id=f"run-{version}",
        target_definition="capped_125",
        rul_cap=125,
        feature_count=552,
        feature_version="1",
        validation_rmse=1,
        replay_rmse=2,
        official_test_rmse=3,
        conformal_coverage_target=0.9,
        loaded_at=datetime.now(UTC),
        checksum=f"checksum-{version}",
        lineage_id=f"lineage-{version}",
    )


def _payload(external_id: str, cycle: int, *, sensor_01: float | None = None) -> dict[str, object]:
    return {
        "external_asset_id": external_id,
        "cycle": cycle,
        "observed_at": f"2026-07-12T12:{cycle:02d}:00Z",
        "operating_setting_1": 1.0,
        "operating_setting_2": 2.0,
        "operating_setting_3": 3.0,
        **{
            f"sensor_{index:02d}": (
                sensor_01 if index == 1 and sensor_01 is not None else float(index + cycle)
            )
            for index in range(1, 22)
        },
        "source": "integration-test",
        "schema_version": "1",
    }


def _client(
    db_session: Session, settings: Settings, loader: FakeLoader | ChampionModelLoader
) -> TestClient:
    factory = sessionmaker(
        bind=db_session.connection(),
        class_=Session,
        expire_on_commit=False,
        autoflush=False,
        join_transaction_mode="create_savepoint",
    )
    metrics = OnlineMetrics()
    service = OnlineInferenceService(factory, cast(ChampionModelLoader, loader), metrics, settings)
    return TestClient(
        create_app(
            settings,
            online_service=service,
            metrics=metrics,
            readiness_checks={"database": lambda: True, "model": lambda: True},
        )
    )


def _settings(postgres_engine: Engine) -> Settings:
    return Settings(
        environment=Environment.TESTING,
        online_inference_enabled=False,
        database_url=postgres_engine.url.render_as_string(hide_password=False),
        model_preload_enabled=False,
    )


def test_atomic_ingestion_idempotency_conflicts_queries_and_new_model_version(
    db_session: Session, postgres_engine: Engine
) -> None:
    loader = FakeLoader()
    settings = _settings(postgres_engine)
    with _client(db_session, settings, loader) as client:
        first = client.post("/v1/sensor-readings", json=_payload("online-asset", 1, sensor_01=10))
        assert first.status_code == 201
        assert first.json()["prediction"]["predicted_rul"] == 90
        assert first.json()["idempotent"] is False
        asset_id = first.json()["asset_id"]

        retry = client.post("/v1/sensor-readings", json=_payload("online-asset", 1, sensor_01=10))
        assert retry.status_code == 200
        assert retry.json()["idempotent"] is True
        assert retry.json()["reading_id"] == first.json()["reading_id"]
        assert loader.model.calls == 1

        conflict = client.post(
            "/v1/sensor-readings", json=_payload("online-asset", 1, sensor_01=11)
        )
        assert conflict.status_code == 409
        gap = client.post("/v1/sensor-readings", json=_payload("online-asset", 3))
        assert gap.status_code == 409
        assert gap.json()["error"]["code"] == "history_conflict"

        second = client.post("/v1/sensor-readings", json=_payload("online-asset", 2))
        assert second.status_code == 201
        earlier_retry = client.post(
            "/v1/sensor-readings", json=_payload("online-asset", 1, sensor_01=10)
        )
        assert earlier_retry.json()["prediction"] == first.json()["prediction"]
        loader.set_version("2")
        repredicted = client.post("/v1/sensor-readings", json=_payload("online-asset", 2))
        assert repredicted.status_code == 201
        assert repredicted.json()["prediction"]["model_version"] == "2"

        assert client.get("/v1/assets").json()["items"][0]["latest_cycle"] == 2
        assert client.get(f"/v1/assets/{asset_id}").status_code == 200
        health = client.get(f"/v1/assets/{asset_id}/health").json()
        assert health["latest_cycle"] == 2
        assert len(health["prediction_trend"]) == 3
        recent = client.get(f"/v1/predictions/recent?asset_id={asset_id}").json()
        assert [item["prediction"]["model_version"] for item in recent["items"][:2]] == [
            "2",
            "1",
        ]
        assert client.get("/v1/models/current").json()["registry_version"] == "2"
        summary = client.get("/v1/monitoring/summary").json()
        assert summary["reading_count"] == 2
        assert summary["stored_prediction_count"] == 3

    assert db_session.scalar(select(Asset).where(Asset.external_id == "online-asset")) is not None
    assert len(list(db_session.scalars(select(SensorReading)))) == 2
    assert len(list(db_session.scalars(select(Prediction)))) == 3


def test_model_failure_rolls_back_asset_reading_and_prediction(
    db_session: Session, postgres_engine: Engine
) -> None:
    loader = FakeLoader(fail=True)
    with _client(db_session, _settings(postgres_engine), loader) as client:
        response = client.post("/v1/sensor-readings", json=_payload("rollback-asset", 1))
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "model_unavailable"
    assert db_session.scalar(select(Asset).where(Asset.external_id == "rollback-asset")) is None


@pytest.mark.real_data
@pytest.mark.skipif(
    not RAW_TRAIN.exists() or not SPLIT_MANIFEST.exists(), reason="FD001 artifacts unavailable"
)
def test_real_mlflow_champion_http_prediction_matches_offline_feature_row(
    db_session: Session, postgres_engine: Engine
) -> None:
    settings = _settings(postgres_engine)
    settings = settings.model_copy(
        update={
            "data_dir": REPO_DATA_DIR,
            "mlflow_tracking_uri": f"sqlite:///{REPO_DATA_DIR / 'mlflow' / 'mlflow.db'}",
        }
    )
    loader = ChampionModelLoader(settings)
    split = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    replay_assets = split["partitions"]["replay"]
    source = pd.read_parquet(RAW_TRAIN)
    rows = source.loc[source["asset_id"] == replay_assets[0]].sort_values("cycle").head(3)
    responses: list[dict[str, object]] = []
    with _client(db_session, settings, loader) as client:
        for row in rows.to_dict(orient="records"):
            body = {
                "external_asset_id": "real-champion-asset",
                "cycle": int(row["cycle"]),
                "observed_at": f"2026-07-12T12:{int(row['cycle']):02d}:00Z",
                **{name: float(row[name]) for name in OPERATING_SETTING_COLUMNS},
                **{name: float(row[name]) for name in SENSOR_COLUMNS},
                "source": "real-fd001-integration",
                "schema_version": "1",
            }
            response = client.post("/v1/sensor-readings", json=body)
            assert response.status_code == 201
            responses.append(response.json())

    loaded = loader.get()
    from turbine_guard.features.builder import FeatureBuilder

    offline = FeatureBuilder(loaded.feature_config).transform_asset(rows)
    model_input = offline.tail(1).loc[:, list(loaded.feature_columns)]
    expected = loaded.model.predict(model_input).iloc[0]
    actual = cast(dict[str, object], responses[-1])["prediction"]
    assert isinstance(actual, dict)
    assert float(actual["predicted_rul"]) == pytest.approx(float(expected["predicted_rul"]))

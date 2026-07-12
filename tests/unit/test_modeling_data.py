"""Loop 4 data-contract and training-only preprocessing tests."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from turbine_guard.features.manifest import sha256_of
from turbine_guard.modeling.config import CandidateConfig, ModelKind, TrainingConfig
from turbine_guard.modeling.data import (
    DatasetRole,
    ModelDataError,
    load_verified_model_data,
    model_matrix,
)
from turbine_guard.modeling.estimators import build_pipeline


def test_exact_manifest_feature_list_and_role_exclusion(feature_data_dir: Path) -> None:
    data = load_verified_model_data(TrainingConfig(data_dir=feature_data_dir))
    train = data.frame(DatasetRole.TRAIN)

    assert len(data.feature_columns) == 552
    assert tuple(column for column in train.columns if column in data.feature_columns) == (
        data.feature_columns
    )
    assert not {"asset_id", "cycle", "split", "rul"} & set(data.feature_columns)
    assert set(train["asset_id"].unique()) == set(data.split_manifest.partitions["train"])


def test_model_matrix_rejects_reordering_and_infinity(feature_data_dir: Path) -> None:
    data = load_verified_model_data(TrainingConfig(data_dir=feature_data_dir))
    train = data.frame(DatasetRole.TRAIN).copy()
    features = data.feature_columns
    reordered = train[[*train.columns[:4], features[1], features[0], *features[2:]]]
    with pytest.raises(ModelDataError, match="order"):
        model_matrix(reordered, features)

    train.loc[train.index[0], features[0]] = np.inf
    with pytest.raises(ModelDataError, match="infinite"):
        model_matrix(train, features)


def test_ridge_imputer_and_scaler_fit_training_only(
    feature_data_dir: Path,
) -> None:
    config = TrainingConfig(data_dir=feature_data_dir)
    data = load_verified_model_data(config)
    train = data.frame(DatasetRole.TRAIN)
    validation = data.frame(DatasetRole.VALIDATION).copy()
    features = data.feature_columns
    candidate = CandidateConfig("ridge", ModelKind.RIDGE, (("alpha", 1.0),), 1)
    pipeline = build_pipeline(candidate, config)
    pipeline.fit(train[list(features)], train["rul"])

    imputer = pipeline.named_steps["imputer"]
    scaler = pipeline.named_steps["scaler"]
    statistics_before = np.asarray(imputer.statistics_).copy()
    scale_before = np.asarray(scaler.mean_).copy()
    validation.loc[:, features[0]] = 1e12
    pipeline.predict(validation[list(features)])

    np.testing.assert_array_equal(imputer.statistics_, statistics_before)
    np.testing.assert_array_equal(scaler.mean_, scale_before)
    assert statistics_before[0] == pytest.approx(train[features[0]].median())


def test_calibration_and_replay_assets_never_enter_training_fit(
    feature_data_dir: Path,
) -> None:
    data = load_verified_model_data(TrainingConfig(data_dir=feature_data_dir))
    train_assets = set(data.frame(DatasetRole.TRAIN)["asset_id"])
    calibration_assets = set(data.frame(DatasetRole.CALIBRATION)["asset_id"])
    replay_assets = set(data.frame(DatasetRole.REPLAY)["asset_id"])

    assert not train_assets & calibration_assets
    assert not train_assets & replay_assets


def test_schema_mismatch_fails_before_modeling(feature_data_dir: Path) -> None:
    base = feature_data_dir / "features" / "cmapss" / "FD001"
    train_path = base / "train.parquet"
    pd.read_parquet(train_path).assign(unexpected_numeric=1.0).to_parquet(train_path, index=False)
    manifest_path = base / "feature_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(item for item in manifest["outputs"] if item["filename"] == "train.parquet")
    record["sha256"] = sha256_of(train_path)
    record["size_bytes"] = train_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ModelDataError, match="schema"):
        load_verified_model_data(TrainingConfig(data_dir=feature_data_dir))

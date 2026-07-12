"""Deterministic model pipeline and serialization tests."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from turbine_guard.modeling.artifacts import load_joblib, serialize_joblib
from turbine_guard.modeling.config import CandidateConfig, ModelKind, TrainingConfig
from turbine_guard.modeling.estimators import MedianRulRegressor, build_pipeline


@pytest.fixture
def regression_fixture() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(7)
    features = pd.DataFrame(rng.normal(size=(40, 4)), columns=list("abcd"))
    features.loc[0, "b"] = np.nan
    target = pd.Series(20.0 - 2.0 * features["a"].fillna(0.0) + features["c"])
    return features, target


def candidates() -> tuple[CandidateConfig, ...]:
    return (
        CandidateConfig("constant", ModelKind.CONSTANT),
        CandidateConfig("ridge", ModelKind.RIDGE, (("alpha", 1.0),), 1),
        CandidateConfig(
            "tree",
            ModelKind.HIST_GRADIENT_BOOSTING,
            (("max_iter", 8), ("max_leaf_nodes", 7), ("learning_rate", 0.1)),
            2,
        ),
        CandidateConfig(
            "xgb",
            ModelKind.XGBOOST,
            (("n_estimators", 8), ("max_depth", 2), ("learning_rate", 0.1)),
            3,
        ),
    )


def test_constant_baseline_uses_training_median() -> None:
    model = MedianRulRegressor().fit([[1], [2], [3]], [1.0, 9.0, 5.0])
    assert model.predict([[100], [200]]).tolist() == [5.0, 5.0]


@pytest.mark.parametrize("candidate", candidates(), ids=lambda item: item.name)
def test_all_model_approaches_train_and_predict(
    candidate: CandidateConfig,
    regression_fixture: tuple[pd.DataFrame, pd.Series],
) -> None:
    features, target = regression_fixture
    pipeline = build_pipeline(candidate, TrainingConfig())
    prediction = pipeline.fit(features, target).predict(features)

    assert prediction.shape == (len(features),)
    assert np.isfinite(prediction).all()


def test_xgboost_reproducible_with_fixed_seed(
    regression_fixture: tuple[pd.DataFrame, pd.Series],
) -> None:
    features, target = regression_fixture
    candidate = candidates()[-1]
    first = build_pipeline(candidate, TrainingConfig(random_seed=11)).fit(features, target)
    second = build_pipeline(candidate, TrainingConfig(random_seed=11)).fit(features, target)

    np.testing.assert_allclose(first.predict(features), second.predict(features), rtol=0, atol=0)


def test_pipeline_serialization_reload_prediction_equality(
    tmp_path: Path,
    regression_fixture: tuple[pd.DataFrame, pd.Series],
) -> None:
    features, target = regression_fixture
    pipeline = build_pipeline(candidates()[1], TrainingConfig()).fit(features, target)
    expected = pipeline.predict(features)
    path = tmp_path / "pipeline.joblib"
    path.write_bytes(serialize_joblib(pipeline))
    restored = load_joblib(path)

    np.testing.assert_array_equal(restored.predict(features), expected)

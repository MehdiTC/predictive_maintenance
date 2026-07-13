"""Real Loop 4 bundle retraining and identical promotion-holdout comparison."""

import pandas as pd
from sklearn.pipeline import Pipeline

from turbine_guard.modeling.config import CandidateConfig, ModelKind, TrainingConfig
from turbine_guard.modeling.conformal import SplitConformalCalibrator
from turbine_guard.modeling.estimators import MedianRulRegressor
from turbine_guard.modeling.pipeline import ModelBundle
from turbine_guard.monitoring.candidate import compare_candidate, train_candidate


def _champion() -> ModelBundle:
    pipeline = Pipeline((("model", MedianRulRegressor()),))
    pipeline.fit(pd.DataFrame({"feature": [0.0, 1.0]}), [5.0, 5.0])
    conformal = SplitConformalCalibrator(0.9).fit([5.0, 4.0], [5.0, 5.0])
    return ModelBundle(
        pipeline,
        ("feature",),
        "uncapped",
        None,
        3,
        5,
        conformal,
        {"model_kind": "constant_median", "model_configuration": {}},
    )


def test_candidate_champion_and_naive_share_one_holdout() -> None:
    training = pd.DataFrame(
        {
            "asset_id": [1, 1, 2, 2],
            "cycle": [1, 2, 1, 2],
            "rul": [4.0, 3.0, 2.0, 1.0],
            "feature": [0.0, 1.0, 2.0, 3.0],
        }
    )
    holdout = pd.DataFrame(
        {
            "asset_id": [1, 1, 2, 2],
            "cycle": [1, 2, 1, 2],
            "rul": [3.0, 2.0, 1.0, 0.0],
            "feature": [1.0, 2.0, 3.0, 4.0],
        }
    )
    champion = _champion()
    config = TrainingConfig(latency_repeats=1)
    candidate = train_candidate(
        training_frame=training,
        feature_columns=("feature",),
        champion_bundle=champion,
        candidate_config=CandidateConfig(
            "ridge", ModelKind.RIDGE, (("alpha", 1.0),), complexity_rank=1
        ),
        training_config=config,
        metadata={"model_kind": "ridge", "model_configuration": {"alpha": 1.0}},
    )
    comparison = compare_candidate(
        candidate=candidate,
        champion=champion,
        training_frame=training,
        holdout_frame=holdout,
        training_config=config,
    )
    assert comparison.row_count == len(holdout)
    assert comparison.asset_count == 2
    assert len(comparison.holdout_sha256) == 64
    assert set(comparison.candidate) >= {
        "regression",
        "critical",
        "warning",
        "interval",
        "inference_latency_ms",
        "artifact_size_bytes",
    }
    assert set(comparison.champion) == set(comparison.candidate)
    assert "regression" in comparison.naive

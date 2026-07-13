"""Leakage isolation, identical holdout, and every blocking promotion gate."""

import uuid
from dataclasses import replace
from datetime import UTC, datetime

import pandas as pd
import pytest

from turbine_guard.monitoring.candidate import CandidateComparison, promotion_gates
from turbine_guard.monitoring.config import PromotionThresholds
from turbine_guard.monitoring.data import LabeledAssetData
from turbine_guard.monitoring.retraining import (
    assemble_holdout_frame,
    assemble_training_frame,
    split_labeled_assets,
)


def _asset(index: int, rows: int = 10) -> LabeledAssetData:
    return LabeledAssetData(
        asset_id=uuid.UUID(int=index),
        source_asset_id=index,
        external_asset_id=f"asset-{index}",
        labeled_at=datetime.now(UTC),
        frame=pd.DataFrame(
            {
                "asset_id": [index] * rows,
                "cycle": range(1, rows + 1),
                "rul": range(rows - 1, -1, -1),
                "feature": [float(index)] * rows,
            }
        ),
    )


def _thresholds() -> PromotionThresholds:
    return PromotionThresholds(0.02, 0.05, 0.6, 250.0, 0.85, 10.0, 10_000, 1e-12)


def _comparison() -> CandidateComparison:
    candidate = {
        "regression": {"mae": 8.0, "rmse": 9.0, "nasa_score": 90.0},
        "critical": {
            "recall": 0.8,
            "precision": 0.7,
            "false_alarms_per_1000_cycles": 20.0,
        },
        "interval": {"empirical_coverage": 0.9, "average_width": 20.0},
        "inference_latency_ms": 1.0,
        "artifact_size_bytes": 1_000,
    }
    champion = {
        "regression": {"mae": 9.0, "rmse": 10.0, "nasa_score": 100.0},
        "critical": candidate["critical"],
        "interval": candidate["interval"],
        "inference_latency_ms": 1.0,
        "artifact_size_bytes": 1_000,
    }
    naive = {"regression": {"mae": 15.0, "rmse": 16.0, "nasa_score": 200.0}}
    return CandidateComparison("a" * 64, 20, 2, candidate, champion, naive)


def test_retraining_split_is_asset_level_disjoint_and_deterministic() -> None:
    assets = [_asset(index) for index in range(1, 7)]
    first = split_labeled_assets(assets, holdout_fraction=0.3, minimum_holdout_assets=2, seed=42)
    second = split_labeled_assets(
        list(reversed(assets)), holdout_fraction=0.3, minimum_holdout_assets=2, seed=42
    )
    assert [asset.asset_id for asset in first.additions] == [
        asset.asset_id for asset in second.additions
    ]
    assert {asset.asset_id for asset in first.additions}.isdisjoint(
        asset.asset_id for asset in first.holdout
    )
    assert len(first.additions) == 4
    assert len(first.holdout) == 2


def test_too_few_assets_blocks_safe_holdout() -> None:
    with pytest.raises(ValueError, match="Too few"):
        split_labeled_assets(
            [_asset(1), _asset(2)],
            holdout_fraction=0.3,
            minimum_holdout_assets=2,
            seed=42,
        )


def test_original_training_additions_and_holdout_are_isolated() -> None:
    original = pd.DataFrame(
        {"asset_id": [1, 1], "cycle": [1, 2], "rul": [1.0, 0.0], "feature": [0.0, 0.0]}
    )
    additions = (_asset(11), _asset(12))
    holdout = (_asset(13), _asset(14))
    training = assemble_training_frame(original, additions, feature_columns=("feature",))
    evaluation = assemble_holdout_frame(holdout, feature_columns=("feature",))
    assert len(training) == 22
    assert len(evaluation) == 20
    assert set(training["feature"]) == {0.0, 11.0, 12.0}
    assert set(evaluation["feature"]) == {13.0, 14.0}
    assert set(training["feature"]).isdisjoint({13.0, 14.0})


def test_all_promotion_gates_pass_for_acceptable_candidate() -> None:
    result = promotion_gates(
        _comparison(),
        thresholds=_thresholds(),
        data_quality_passes=True,
        enough_labeled_data=True,
        artifact_valid=True,
        reload_equivalence_difference=0.0,
    )
    assert result.passed
    assert not result.blocking_failures


@pytest.mark.parametrize(
    "gate",
    [
        "data_quality",
        "enough_labeled_data",
        "candidate_artifact_valid",
        "beats_naive_baseline",
        "rmse_not_materially_worse",
        "nasa_not_materially_worse",
        "critical_recall",
        "false_alarms",
        "conformal_coverage",
        "inference_latency",
        "artifact_size",
        "mlflow_reload_equivalence",
    ],
)
def test_every_blocking_promotion_gate(gate: str) -> None:
    comparison = _comparison()
    candidate = dict(comparison.candidate)
    candidate["regression"] = dict(candidate["regression"])
    candidate["critical"] = dict(candidate["critical"])
    candidate["interval"] = dict(candidate["interval"])
    kwargs = {
        "data_quality_passes": True,
        "enough_labeled_data": True,
        "artifact_valid": True,
        "reload_equivalence_difference": 0.0,
    }
    if gate == "data_quality":
        kwargs["data_quality_passes"] = False
    elif gate == "enough_labeled_data":
        kwargs["enough_labeled_data"] = False
    elif gate == "candidate_artifact_valid":
        kwargs["artifact_valid"] = False
    elif gate == "beats_naive_baseline":
        candidate["regression"]["rmse"] = 20.0
    elif gate == "rmse_not_materially_worse":
        candidate["regression"]["rmse"] = 11.0
    elif gate == "nasa_not_materially_worse":
        candidate["regression"]["nasa_score"] = 110.0
    elif gate == "critical_recall":
        candidate["critical"]["recall"] = 0.5
    elif gate == "false_alarms":
        candidate["critical"]["false_alarms_per_1000_cycles"] = 300.0
    elif gate == "conformal_coverage":
        candidate["interval"]["empirical_coverage"] = 0.8
    elif gate == "inference_latency":
        candidate["inference_latency_ms"] = 11.0
    elif gate == "artifact_size":
        candidate["artifact_size_bytes"] = 10_001
    elif gate == "mlflow_reload_equivalence":
        kwargs["reload_equivalence_difference"] = 1e-6
    modified = replace(comparison, candidate=candidate)
    result = promotion_gates(modified, thresholds=_thresholds(), **kwargs)
    assert not result.passed
    assert gate in result.blocking_failures

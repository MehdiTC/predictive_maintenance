"""Validation-only champion eligibility, ranking, and tie tests."""

import copy

import pytest

from turbine_guard.modeling.config import SelectionConfig
from turbine_guard.modeling.selection import ChampionSelectionError, select_champion


def candidate(
    candidate_id: str,
    *,
    rmse: float,
    recall: float = 0.8,
    false_alarms: float = 10.0,
    complexity: int = 2,
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "complexity_rank": complexity,
        "common_domain_metrics": {"rmse": rmse, "mae": rmse - 1, "nasa_score": rmse * 2},
        "critical_alert_metrics": {
            "recall": recall,
            "false_alarms_per_1000_cycles": false_alarms,
        },
        "prediction_latency_ms": 0.1,
        "model_size_bytes": 100,
        "target_definition": {"name": "uncapped", "cap": None},
        "alert_thresholds": {"critical_horizon": 30, "warning_horizon": 50},
        "configuration": {},
    }


def test_ineligible_model_rejected_and_metric_ranking() -> None:
    bad = candidate("bad", rmse=5, recall=0.2)
    good = candidate("good", rmse=7)
    result = select_champion([bad, good], SelectionConfig())
    assert result.selected_candidate_id == "good"
    bad_record = next(
        item for item in result.artifact["candidates"] if item["candidate_id"] == "bad"
    )
    assert bad_record["eligible"] is False


def test_simpler_model_wins_within_rmse_tolerance() -> None:
    complex_best = candidate("complex", rmse=10.0, complexity=4)
    simple_close = candidate("simple", rmse=10.1, complexity=1)
    result = select_champion(
        [complex_best, simple_close],
        SelectionConfig(relative_rmse_tolerance=0.02),
    )
    assert result.selected_candidate_id == "simple"


def test_replay_metrics_cannot_change_selection() -> None:
    validation = [candidate("a", rmse=8), candidate("b", rmse=9)]
    first = select_champion(validation, SelectionConfig(relative_rmse_tolerance=0.0))
    with_replay_annotations = copy.deepcopy(validation)
    with_replay_annotations[0]["replay_rmse"] = 999
    with_replay_annotations[1]["replay_rmse"] = 1
    second = select_champion(with_replay_annotations, SelectionConfig(relative_rmse_tolerance=0.0))
    assert first.selected_candidate_id == second.selected_candidate_id == "a"
    assert first.artifact["selection_dataset_role"] == "validation_only"


def test_no_eligible_model_fails() -> None:
    with pytest.raises(ChampionSelectionError, match="No model"):
        select_champion([candidate("bad", rmse=5, recall=0.0)], SelectionConfig())

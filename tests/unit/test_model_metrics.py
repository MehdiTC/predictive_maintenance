"""Hand-calculated regression, alert, aggregation, and slice metric tests."""

import math

import pandas as pd
import pytest

from turbine_guard.modeling.alerts import alert_metrics
from turbine_guard.modeling.metrics import (
    evaluate_regression_frame,
    nasa_asymmetric_score,
    regression_metrics,
)


def test_regression_metrics_known_values() -> None:
    metrics = regression_metrics([1.0, 2.0, 3.0], [2.0, 2.0, 4.0])
    assert metrics["mae"] == pytest.approx(2 / 3)
    assert metrics["rmse"] == pytest.approx(math.sqrt(2 / 3))
    assert metrics["r2"] == pytest.approx(0.0)


def test_nasa_asymmetric_score_hand_calculation() -> None:
    score = nasa_asymmetric_score([10.0, 10.0], [0.0, 20.0])
    expected = math.expm1(10 / 13) + math.expm1(10 / 10)
    assert score == pytest.approx(expected)
    assert nasa_asymmetric_score([10.0], [20.0]) > nasa_asymmetric_score([10.0], [0.0])


def alert_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "asset_id": [1, 1, 1, 1, 2, 2, 2, 2],
            "cycle": [1, 2, 3, 4, 1, 2, 3, 4],
            "y_true_uncapped": [3, 2, 1, 0, 3, 2, 1, 0],
            # Asset 1 alerts from lead 2; asset 2 never alerts.
            "y_pred": [4, 2, 1, 0, 5, 4, 3, 3],
        }
    )


def test_alert_precision_recall_f1_false_alarms_and_missed_failure() -> None:
    metrics = alert_metrics(alert_frame(), horizon=2)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(2 / 3)
    assert metrics["false_alarm_episodes"] == 0
    assert metrics["missed_failures"] == 1
    assert metrics["mean_first_alert_lead_time"] == pytest.approx(2.0)
    assert metrics["timely_warning_asset_percentage"] == pytest.approx(50.0)


def test_alert_episode_collapses_repeated_rows_and_counts_early_alarm() -> None:
    frame = alert_frame()
    frame.loc[0, "y_pred"] = 2  # first alert at true lead 3, early for horizon 2
    metrics = alert_metrics(frame, horizon=2)
    assert metrics["alert_episode_count"] == 1
    assert metrics["false_alarm_episodes"] == 1
    assert metrics["false_alarms_per_1000_cycles"] == pytest.approx(125.0)
    assert metrics["assets_alerted_too_early"] == 1


def test_asset_level_aggregation_and_lifecycle_slices() -> None:
    frame = pd.DataFrame(
        {
            "asset_id": [1, 1, 1, 2, 2, 2, 2],
            "cycle": [1, 2, 3, 1, 2, 3, 4],
            "y_true_uncapped": [2, 1, 0, 3, 2, 1, 0],
            "y_true": [2, 1, 0, 3, 2, 1, 0],
            "y_pred": [2, 1, 0, 2, 2, 1, 0],
        }
    )
    result = evaluate_regression_frame(frame)
    assert len(result["per_asset"]) == 2
    assert {
        record["value"] for record in result["slices"] if record["slice"] == "lifecycle_stage"
    } == {
        "early",
        "middle",
        "late",
    }

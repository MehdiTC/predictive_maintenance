"""Delayed evaluation: hand-calculated metrics, grouping, and aggregation."""

import math
import uuid
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from turbine_guard.database.enums import EvaluationScope, ReplayMode, ReplayRunStatus
from turbine_guard.modeling.config import AlertConfig
from turbine_guard.replay.errors import ReplayOutcomeError
from turbine_guard.replay.evaluation import (
    AGGREGATE_AGGREGATION,
    PER_ASSET_AGGREGATION,
    DelayedEvaluationConfig,
    aggregate_evaluations,
    build_outcome_frame,
    group_by_model,
    per_asset_evaluations,
)
from turbine_guard.replay.state import ReplayRunState, StoredOutcome, StoredPrediction

NOW = datetime(2026, 7, 13, 12, 0, tzinfo=UTC)
CONFIG = DelayedEvaluationConfig(
    alert=AlertConfig(critical_horizon=2, warning_horizon=4, minimum_lead_cycles=1)
)


def _prediction(
    cycle: int,
    y_pred: float,
    *,
    version: str = "1",
    lower: float | None = None,
    upper: float | None = None,
) -> StoredPrediction:
    return StoredPrediction(
        prediction_id=uuid.uuid4(),
        cycle=cycle,
        predicted_rul=y_pred,
        lower_rul=lower,
        upper_rul=upper,
        risk_level="healthy",
        model_name="fake-rul",
        model_version=version,
        model_run_id=f"run-{version}",
        prediction_timestamp=NOW + timedelta(seconds=cycle),
    )


def _labeled(predictions: list[StoredPrediction], final_cycle: int) -> list[StoredOutcome]:
    return [
        StoredOutcome(
            prediction_id=prediction.prediction_id,
            cycle=prediction.cycle,
            realized_rul=final_cycle - prediction.cycle,
        )
        for prediction in predictions
    ]


def _run_state(final_cycle: int = 5, **overrides: object) -> ReplayRunState:
    values: dict[str, object] = {
        "run_id": uuid.uuid4(),
        "dataset_name": "cmapss",
        "dataset_subset": "FD001",
        "source_asset_id": 9,
        "attempt": 1,
        "external_asset_id": "replay-FD001-009",
        "asset_id": uuid.uuid4(),
        "final_cycle": final_cycle,
        "last_confirmed_cycle": final_cycle,
        "status": ReplayRunStatus.RUNNING,
        "mode": ReplayMode.ACCELERATED,
        "cycle_delay_seconds": 0.0,
        "simulated_cycle_duration_seconds": 1.0,
        "replay_started_at": NOW,
        "last_advanced_at": NOW,
        "completed_at": None,
        "failure_event_id": uuid.uuid4(),
        "labels_backfilled_at": NOW,
        "evaluation_completed_at": None,
        "error_message": None,
        "metadata": {},
    }
    values.update(overrides)
    return ReplayRunState(**values)  # type: ignore[arg-type]


def _hand_example() -> tuple[list[StoredPrediction], list[StoredOutcome]]:
    """Cycles 1..5 with realized RUL [4,3,2,1,0] and y_pred [5,3,1,1,0]."""
    lowers = [3.0, 2.0, 0.0, 0.0, 0.0]
    uppers = [6.0, 4.0, 2.0, 2.0, 1.0]
    preds = [5.0, 3.0, 1.0, 1.0, 0.0]
    predictions = [
        _prediction(cycle, preds[cycle - 1], lower=lowers[cycle - 1], upper=uppers[cycle - 1])
        for cycle in range(1, 6)
    ]
    return predictions, _labeled(predictions, final_cycle=5)


class TestHandCalculatedExample:
    def test_regression_alert_and_interval_metrics_match_hand_calculation(self) -> None:
        predictions, outcomes = _hand_example()
        commands = per_asset_evaluations(_run_state(), predictions, outcomes, CONFIG)
        assert len(commands) == 1
        command = commands[0]
        assert command.evaluation_scope is EvaluationScope.REPLAY
        assert command.sample_count == 5
        assert command.mae == pytest.approx(0.4)
        assert command.rmse == pytest.approx(math.sqrt(0.4))
        expected_nasa = float(np.expm1(1 / 10) + np.expm1(1 / 13))
        assert command.nasa_score == pytest.approx(expected_nasa)
        assert command.critical_precision == pytest.approx(1.0)
        assert command.critical_recall == pytest.approx(1.0)
        assert command.interval_coverage == pytest.approx(1.0)

        metrics = command.metrics
        assert metrics["aggregation"] == PER_ASSET_AGGREGATION
        assert metrics["critical"]["missed_failures"] == 0
        assert metrics["critical"]["false_alarm_episodes"] == 0
        assert metrics["critical"]["false_alarms_per_1000_cycles"] == pytest.approx(0.0)
        assert metrics["critical"]["per_asset"][0]["first_alert_cycle"] == 3
        assert metrics["critical"]["per_asset"][0]["first_alert_lead_time"] == pytest.approx(2.0)
        assert metrics["critical"]["per_asset"][0]["timely"] is True
        assert metrics["warning"]["recall"] == pytest.approx(0.8)
        assert metrics["warning"]["per_asset"][0]["first_alert_lead_time"] == pytest.approx(3.0)
        assert metrics["interval"]["average_width"] == pytest.approx(2.0)
        assert metrics["interval"]["empirical_coverage"] == pytest.approx(1.0)
        assert metrics["source_asset_id"] == 9

    def test_missed_failure_is_reported(self) -> None:
        predictions = [_prediction(cycle, 90.0) for cycle in range(1, 6)]
        outcomes = _labeled(predictions, final_cycle=5)
        command = per_asset_evaluations(_run_state(), predictions, outcomes, CONFIG)[0]
        assert command.metrics["critical"]["missed_failures"] == 1
        assert command.metrics["critical"]["per_asset"][0]["first_alert_lead_time"] is None
        assert command.metrics["critical"]["per_asset"][0]["timely"] is False

    def test_predictions_without_intervals_yield_no_interval_metrics(self) -> None:
        predictions = [_prediction(cycle, float(5 - cycle)) for cycle in range(1, 6)]
        outcomes = _labeled(predictions, final_cycle=5)
        command = per_asset_evaluations(_run_state(), predictions, outcomes, CONFIG)[0]
        assert command.interval_coverage is None
        assert command.metrics["interval"] is None


class TestModelVersionGrouping:
    def test_versions_are_never_blended(self) -> None:
        early = [_prediction(cycle, float(6 - cycle), version="1") for cycle in (1, 2, 3)]
        late = [_prediction(cycle, float(6 - cycle), version="2") for cycle in (4, 5)]
        predictions = early + late
        outcomes = _labeled(predictions, final_cycle=5)
        commands = per_asset_evaluations(_run_state(), predictions, outcomes, CONFIG)
        by_version = {command.model_version: command for command in commands}
        assert set(by_version) == {"1", "2"}
        assert by_version["1"].sample_count == 3
        assert by_version["2"].sample_count == 2
        assert by_version["1"].metrics["model_run_ids"] == ["run-1"]
        assert by_version["2"].metrics["model_run_ids"] == ["run-2"]

    def test_group_by_model_splits_frames(self) -> None:
        predictions = [_prediction(1, 4.0, version="1"), _prediction(2, 3.0, version="2")]
        outcomes = _labeled(predictions, final_cycle=5)
        frame = build_outcome_frame(predictions, outcomes, source_asset_id=9)
        groups = group_by_model(frame)
        assert set(groups) == {("fake-rul", "1"), ("fake-rul", "2")}


class TestFrameConstruction:
    def test_missing_label_fails_evaluation(self) -> None:
        predictions = [_prediction(1, 4.0), _prediction(2, 3.0)]
        outcomes = _labeled(predictions[:1], final_cycle=5)
        with pytest.raises(ReplayOutcomeError, match="no realized label"):
            build_outcome_frame(predictions, outcomes, source_asset_id=9)

    def test_evaluation_requires_failure_event(self) -> None:
        predictions, outcomes = _hand_example()
        run = _run_state(failure_event_id=None)
        with pytest.raises(ReplayOutcomeError, match="failure event"):
            per_asset_evaluations(run, predictions, outcomes, CONFIG)


class TestAggregateEvaluation:
    def test_aggregate_groups_by_version_and_records_runs(self) -> None:
        run_a = _run_state(source_asset_id=9, external_asset_id="replay-FD001-009")
        run_b = _run_state(source_asset_id=13, external_asset_id="replay-FD001-013")
        predictions_a, outcomes_a = _hand_example()
        predictions_b = [_prediction(cycle, float(7 - cycle)) for cycle in range(1, 8)]
        outcomes_b = _labeled(predictions_b, final_cycle=7)
        frame_a = build_outcome_frame(predictions_a, outcomes_a, source_asset_id=9)
        frame_b = build_outcome_frame(predictions_b, outcomes_b, source_asset_id=13)
        commands = aggregate_evaluations([run_a, run_b], [frame_a, frame_b], CONFIG)
        assert len(commands) == 1
        command = commands[0]
        assert command.sample_count == 12
        assert command.metrics["aggregation"] == AGGREGATE_AGGREGATION
        assert command.metrics["asset_count"] == 2
        assert sorted(command.metrics["source_asset_ids"]) == [9, 13]
        assert command.metrics["replay_run_ids"] == sorted([str(run_a.run_id), str(run_b.run_id)])
        assert len(command.metrics["critical"]["per_asset"]) == 2

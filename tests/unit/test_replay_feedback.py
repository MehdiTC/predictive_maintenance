"""Realized-label computation: correctness, invariants, and conflict detection."""

import uuid
from datetime import UTC, datetime
from itertools import pairwise

import pytest

from turbine_guard.database.commands import NewPredictionOutcome
from turbine_guard.replay.errors import ReplayOutcomeError
from turbine_guard.replay.feedback import (
    build_outcome_commands,
    realized_rul,
    validate_realized_labels,
)
from turbine_guard.replay.state import StoredPrediction

NOW = datetime(2026, 7, 13, 12, 0, tzinfo=UTC)


def _prediction(cycle: int, *, version: str = "1") -> StoredPrediction:
    return StoredPrediction(
        prediction_id=uuid.uuid4(),
        cycle=cycle,
        predicted_rul=float(100 - cycle),
        lower_rul=float(max(0, 95 - cycle)),
        upper_rul=float(105 - cycle),
        risk_level="healthy",
        model_name="fake-rul",
        model_version=version,
        model_run_id=f"run-{version}",
        prediction_timestamp=NOW,
    )


def _outcome(cycle: int, realized: int, *, final_cycle: int = 5) -> NewPredictionOutcome:
    del final_cycle
    return NewPredictionOutcome(
        prediction_id=uuid.uuid4(),
        maintenance_event_id=uuid.uuid4(),
        asset_id=uuid.uuid4(),
        cycle=cycle,
        realized_rul=realized,
        labeled_at=NOW,
    )


class TestRealizedRul:
    def test_realized_rul_is_final_minus_cycle(self) -> None:
        assert realized_rul(10, 1) == 9
        assert realized_rul(10, 10) == 0

    def test_cycle_beyond_failure_is_impossible(self) -> None:
        with pytest.raises(ReplayOutcomeError, match="impossible"):
            realized_rul(10, 11)

    def test_non_positive_cycles_are_rejected(self) -> None:
        with pytest.raises(ReplayOutcomeError):
            realized_rul(0, 1)
        with pytest.raises(ReplayOutcomeError):
            realized_rul(10, 0)


class TestBuildOutcomeCommands:
    def test_labels_cover_every_prediction_and_end_at_zero(self) -> None:
        predictions = [_prediction(cycle) for cycle in range(1, 6)]
        event_id = uuid.uuid4()
        asset_id = uuid.uuid4()
        commands = build_outcome_commands(
            predictions,
            final_cycle=5,
            asset_id=asset_id,
            maintenance_event_id=event_id,
            labeled_at=NOW,
        )
        assert [command.realized_rul for command in commands] == [4, 3, 2, 1, 0]
        assert all(command.maintenance_event_id == event_id for command in commands)
        assert all(command.asset_id == asset_id for command in commands)
        assert all(command.realized_rul >= 0 for command in commands)

    def test_labels_decrease_by_exactly_one_per_cycle(self) -> None:
        commands = build_outcome_commands(
            [_prediction(cycle) for cycle in range(1, 8)],
            final_cycle=7,
            asset_id=uuid.uuid4(),
            maintenance_event_id=uuid.uuid4(),
            labeled_at=NOW,
        )
        realized = [command.realized_rul for command in commands]
        assert all(a - b == 1 for a, b in pairwise(realized))

    def test_multiple_model_versions_share_the_same_labels(self) -> None:
        predictions = [_prediction(3, version="1"), _prediction(3, version="2")]
        commands = build_outcome_commands(
            predictions,
            final_cycle=5,
            asset_id=uuid.uuid4(),
            maintenance_event_id=uuid.uuid4(),
            labeled_at=NOW,
        )
        assert [command.realized_rul for command in commands] == [2, 2]

    def test_prediction_beyond_failure_cycle_is_conflicting(self) -> None:
        with pytest.raises(ReplayOutcomeError, match="impossible"):
            build_outcome_commands(
                [_prediction(6)],
                final_cycle=5,
                asset_id=uuid.uuid4(),
                maintenance_event_id=uuid.uuid4(),
                labeled_at=NOW,
            )

    def test_empty_predictions_are_rejected(self) -> None:
        with pytest.raises(ReplayOutcomeError, match="no predictions"):
            build_outcome_commands(
                [],
                final_cycle=5,
                asset_id=uuid.uuid4(),
                maintenance_event_id=uuid.uuid4(),
                labeled_at=NOW,
            )


class TestValidateRealizedLabels:
    def test_inconsistent_label_is_detected(self) -> None:
        with pytest.raises(ReplayOutcomeError, match="inconsistent"):
            validate_realized_labels([_outcome(2, 7)], final_cycle=5)

    def test_conflicting_duplicate_cycles_are_detected(self) -> None:
        with pytest.raises(ReplayOutcomeError, match=r"inconsistent|conflicting"):
            validate_realized_labels(
                [_outcome(2, 3), _outcome(2, 4)],
                final_cycle=5,
            )

    def test_consistent_labels_pass(self) -> None:
        validate_realized_labels(
            [_outcome(1, 4), _outcome(2, 3), _outcome(5, 0)],
            final_cycle=5,
        )

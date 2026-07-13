"""Realized RUL label computation for delayed failure outcomes.

For an asset whose observed failure cycle is ``T``, the realized remaining
useful life of a prediction made at cycle ``t`` is ``T - t``. Labels are
computed only after the failure event exists, never mutate the original
predictions, and are validated for physical consistency before persistence.
"""

import uuid
from collections.abc import Sequence
from datetime import datetime
from itertools import pairwise

from turbine_guard.database.commands import NewPredictionOutcome
from turbine_guard.replay.errors import ReplayOutcomeError
from turbine_guard.replay.state import StoredPrediction


def realized_rul(final_cycle: int, cycle: int) -> int:
    """Realized remaining useful life at ``cycle`` for a failure at ``final_cycle``."""
    if final_cycle <= 0 or cycle <= 0:
        raise ReplayOutcomeError("Failure and prediction cycles must be positive.")
    if cycle > final_cycle:
        raise ReplayOutcomeError(
            f"Prediction cycle {cycle} lies beyond the observed failure cycle "
            f"{final_cycle}; the recorded outcome is impossible."
        )
    return final_cycle - cycle


def build_outcome_commands(
    predictions: Sequence[StoredPrediction],
    *,
    final_cycle: int,
    asset_id: uuid.UUID,
    maintenance_event_id: uuid.UUID,
    labeled_at: datetime,
) -> list[NewPredictionOutcome]:
    """Compute one realized label per stored prediction for one failure event."""
    if not predictions:
        raise ReplayOutcomeError("There are no predictions to backfill for this asset.")
    commands = [
        NewPredictionOutcome(
            prediction_id=prediction.prediction_id,
            maintenance_event_id=maintenance_event_id,
            asset_id=asset_id,
            cycle=prediction.cycle,
            realized_rul=realized_rul(final_cycle, prediction.cycle),
            labeled_at=labeled_at,
        )
        for prediction in predictions
    ]
    validate_realized_labels(commands, final_cycle=final_cycle)
    return commands


def validate_realized_labels(commands: Sequence[NewPredictionOutcome], *, final_cycle: int) -> None:
    """Check the physical invariants of a backfilled label set.

    Labels must be non-negative, the final observed cycle must realize zero,
    and the label must fall by exactly one per cycle (equivalently,
    ``cycle + realized_rul == final_cycle`` for every label). Duplicate cycles
    must agree, or the outcome set is internally conflicting.
    """
    realized_by_cycle: dict[int, int] = {}
    for command in commands:
        if command.realized_rul < 0:
            raise ReplayOutcomeError("Realized RUL labels must be non-negative.")
        if command.cycle + command.realized_rul != final_cycle:
            raise ReplayOutcomeError(
                f"Label at cycle {command.cycle} realizes {command.realized_rul}, which "
                f"is inconsistent with failure at cycle {final_cycle}."
            )
        previous = realized_by_cycle.get(command.cycle)
        if previous is not None and previous != command.realized_rul:
            raise ReplayOutcomeError(
                f"Cycle {command.cycle} received conflicting realized labels "
                f"({previous} and {command.realized_rul})."
            )
        realized_by_cycle[command.cycle] = command.realized_rul
    cycles = sorted(realized_by_cycle)
    if final_cycle in realized_by_cycle and realized_by_cycle[final_cycle] != 0:
        raise ReplayOutcomeError("The final observed cycle must realize an RUL of zero.")
    for earlier, later in pairwise(cycles):
        step = later - earlier
        if realized_by_cycle[earlier] - realized_by_cycle[later] != step:
            raise ReplayOutcomeError(
                f"Realized labels between cycles {earlier} and {later} do not decrease "
                "by exactly one per cycle."
            )

"""Explicit validation-only champion selection."""

from dataclasses import dataclass
from typing import Any

from turbine_guard.modeling.config import SelectionConfig


class ChampionSelectionError(RuntimeError):
    """Raised when no candidate satisfies operational eligibility."""


@dataclass(frozen=True)
class ChampionSelection:
    """Selected candidate and the complete auditable decision record."""

    selected_candidate_id: str
    artifact: dict[str, Any]


def select_champion(
    validation_candidates: list[dict[str, Any]], config: SelectionConfig
) -> ChampionSelection:
    """Select using validation metrics only, preferring simplicity within tolerance."""
    if not validation_candidates:
        raise ChampionSelectionError("No validation candidates were supplied.")

    evaluated: list[dict[str, Any]] = []
    for candidate in validation_candidates:
        critical = candidate["critical_alert_metrics"]
        checks = {
            "minimum_critical_recall": (
                float(critical["recall"]) >= config.minimum_critical_recall
            ),
            "maximum_false_alarms_per_1000_cycles": (
                float(critical["false_alarms_per_1000_cycles"])
                <= config.maximum_false_alarms_per_1000_cycles
            ),
        }
        evaluated.append(
            {**candidate, "eligibility_checks": checks, "eligible": all(checks.values())}
        )

    eligible = [candidate for candidate in evaluated if candidate["eligible"]]
    if not eligible:
        raise ChampionSelectionError(
            "No model satisfies the configured validation-only operational requirements."
        )
    best_rmse = min(float(candidate["common_domain_metrics"]["rmse"]) for candidate in eligible)
    tolerance_limit = best_rmse * (1.0 + config.relative_rmse_tolerance)
    tied = [
        candidate
        for candidate in eligible
        if float(candidate["common_domain_metrics"]["rmse"]) <= tolerance_limit
    ]
    selected = min(
        tied,
        key=lambda candidate: (
            int(candidate["complexity_rank"]),
            float(candidate["common_domain_metrics"]["nasa_score"]),
            float(candidate["common_domain_metrics"]["mae"]),
            float(candidate["prediction_latency_ms"]),
            int(candidate["model_size_bytes"]),
            str(candidate["candidate_id"]),
        ),
    )
    rationale = (
        f"Eligible on validation critical recall and false-alarm rate. Its common-domain RMSE "
        f"is within {config.relative_rmse_tolerance:.1%} of the best eligible RMSE; among that "
        "set it has the lowest declared complexity rank, with NASA score, MAE, latency, size, "
        "and candidate ID used as deterministic tie-breakers."
    )
    artifact = {
        "selection_dataset_role": "validation_only",
        "criteria": {
            "minimum_critical_recall": config.minimum_critical_recall,
            "maximum_false_alarms_per_1000_cycles": config.maximum_false_alarms_per_1000_cycles,
            "relative_rmse_tolerance": config.relative_rmse_tolerance,
            "primary_metric": "common_domain_rmse",
            "secondary_metric": "common_domain_nasa_score",
            "tie_preference": "lower declared complexity rank",
        },
        "candidates": evaluated,
        "selected_model": selected["candidate_id"],
        "selected_target_definition": selected["target_definition"],
        "selected_alert_thresholds": selected["alert_thresholds"],
        "rationale": rationale,
    }
    return ChampionSelection(str(selected["candidate_id"]), artifact)

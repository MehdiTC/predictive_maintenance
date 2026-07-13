"""Structured and configurable retraining-trigger decisions."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from turbine_guard.database.enums import DataQualityStatus, DriftStatus
from turbine_guard.monitoring.config import TriggerThresholds


class TriggerAction(StrEnum):
    NO_ACTION = "no_action"
    MONITOR = "monitor"
    RETRAIN = "retrain"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class TriggerDecision:
    action: TriggerAction
    reasons: tuple[str, ...]
    checks: dict[str, bool]
    signals: dict[str, float | int | bool | None]

    def record(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "reasons": list(self.reasons),
            "checks": self.checks,
            "signals": self.signals,
        }


def decide_retraining(
    *,
    thresholds: TriggerThresholds,
    data_quality_status: DataQualityStatus,
    drift_status: DriftStatus,
    drifted_feature_count: int,
    newly_labeled_assets: int,
    newly_labeled_rows: int,
    current_metrics: dict[str, Any] | None,
    baseline_metrics: dict[str, Any] | None,
    interval_elapsed: bool,
    safe_holdout_available: bool,
    manual_force: bool = False,
) -> TriggerDecision:
    """Choose no_action, monitor, retrain, or blocked without bypassing safety checks."""
    performance = current_metrics or {}
    baseline = baseline_metrics or {}
    regression = performance.get("regression", {})
    baseline_regression = baseline.get("regression", baseline)
    critical = performance.get("critical", {})
    baseline_critical = baseline.get("critical", {})
    interval = performance.get("interval") or {}

    rmse_degradation = _relative_increase(regression.get("rmse"), baseline_regression.get("rmse"))
    mae_degradation = _relative_increase(regression.get("mae"), baseline_regression.get("mae"))
    false_alarm_increase = _absolute_increase(
        critical.get("false_alarms_per_1000_cycles"),
        baseline_critical.get("false_alarms_per_1000_cycles"),
    )
    recall = _optional_float(critical.get("recall"))
    coverage = _optional_float(interval.get("empirical_coverage"))
    signals: dict[str, float | int | bool | None] = {
        "newly_labeled_assets": newly_labeled_assets,
        "newly_labeled_rows": newly_labeled_rows,
        "drifted_feature_count": drifted_feature_count,
        "rmse_relative_degradation": rmse_degradation,
        "mae_relative_degradation": mae_degradation,
        "critical_recall": recall,
        "false_alarm_increase": false_alarm_increase,
        "interval_coverage": coverage,
        "interval_elapsed": interval_elapsed,
        "manual_force": manual_force,
    }
    checks = {
        "data_quality_passes": data_quality_status is not DataQualityStatus.FAIL,
        "minimum_assets": newly_labeled_assets >= thresholds.minimum_assets,
        "minimum_rows": newly_labeled_rows >= thresholds.minimum_rows,
        "safe_holdout": safe_holdout_available,
    }
    triggers = {
        "manual_force": manual_force,
        "scheduled_interval": (
            interval_elapsed
            and newly_labeled_assets >= thresholds.minimum_assets
            and newly_labeled_rows >= thresholds.minimum_rows
        ),
        "feature_drift": (
            drift_status is DriftStatus.DETECTED
            and drifted_feature_count >= thresholds.drifted_feature_count
        ),
        "rmse_degradation": (
            rmse_degradation is not None and rmse_degradation > thresholds.performance_degradation
        ),
        "mae_degradation": (
            mae_degradation is not None and mae_degradation > thresholds.performance_degradation
        ),
        "critical_recall": recall is not None and recall < thresholds.minimum_critical_recall,
        "false_alarm_increase": (
            false_alarm_increase is not None
            and false_alarm_increase > thresholds.false_alarm_increase_tolerance
        ),
        "coverage_degradation": coverage is not None and coverage < thresholds.minimum_coverage,
    }
    active = tuple(name for name, value in triggers.items() if value)
    if data_quality_status is DataQualityStatus.FAIL:
        return TriggerDecision(
            TriggerAction.BLOCKED,
            ("data_quality_failure",),
            checks,
            signals,
        )
    if active and not all(checks.values()):
        missing = tuple(name for name, value in checks.items() if not value)
        return TriggerDecision(TriggerAction.BLOCKED, (*active, *missing), checks, signals)
    if active and all(checks.values()):
        return TriggerDecision(TriggerAction.RETRAIN, active, checks, signals)
    if data_quality_status in {
        DataQualityStatus.WARNING,
        DataQualityStatus.INSUFFICIENT_DATA,
    } or drift_status in {DriftStatus.WARNING, DriftStatus.INSUFFICIENT_DATA}:
        return TriggerDecision(TriggerAction.MONITOR, ("watch_thresholds",), checks, signals)
    return TriggerDecision(TriggerAction.NO_ACTION, ("healthy_window",), checks, signals)


def _relative_increase(value: object, baseline: object) -> float | None:
    current = _optional_float(value)
    reference = _optional_float(baseline)
    if current is None or reference is None or reference <= 0:
        return None
    return (current - reference) / reference


def _absolute_increase(value: object, baseline: object) -> float | None:
    current = _optional_float(value)
    reference = _optional_float(baseline)
    return None if current is None or reference is None else current - reference


def _optional_float(value: object) -> float | None:
    return None if value is None else float(str(value))

"""Delayed production performance reports composed from Loop 4 metric functions."""

from dataclasses import dataclass
from typing import Any

import pandas as pd

from turbine_guard.modeling.config import AlertConfig
from turbine_guard.replay.evaluation import DelayedEvaluationConfig, evaluate_group


@dataclass(frozen=True)
class PerformanceResult:
    sample_count: int
    metrics: dict[str, Any]


def delayed_performance_report(
    frame: pd.DataFrame, alert_config: AlertConfig | None = None
) -> PerformanceResult:
    """Calculate regression, alert, lead-time, conformal, and output-distribution metrics."""
    if frame.empty:
        return PerformanceResult(0, {"status": "insufficient_data"})
    results = evaluate_group(
        frame,
        DelayedEvaluationConfig(alert=alert_config or AlertConfig()),
    )
    risk = pd.cut(
        frame["y_pred"],
        bins=[float("-inf"), 30.0, 50.0, float("inf")],
        labels=["critical", "warning", "healthy"],
    )
    metrics: dict[str, Any] = {
        "status": "available",
        "regression": results["regression"],
        "critical": _bounded_alert_record(results["critical"]),
        "warning": _bounded_alert_record(results["warning"]),
        "interval": results["interval"],
        "prediction_distribution": _distribution(frame["y_pred"]),
        "risk_distribution": {str(key): int(value) for key, value in risk.value_counts().items()},
        "asset_count": int(frame["asset_id"].nunique()),
    }
    return PerformanceResult(len(frame), metrics)


def _bounded_alert_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key != "per_asset"}


def _distribution(series: pd.Series) -> dict[str, float]:
    return {
        "minimum": float(series.min()),
        "p10": float(series.quantile(0.10)),
        "median": float(series.median()),
        "p90": float(series.quantile(0.90)),
        "maximum": float(series.max()),
        "mean": float(series.mean()),
        "std": float(series.std(ddof=0)),
    }

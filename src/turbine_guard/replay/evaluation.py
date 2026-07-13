"""Delayed evaluation of stored online predictions against realized outcomes.

All metric formulas are the Loop 4 implementations (`modeling.metrics`,
`modeling.alerts`, `modeling.conformal`); this module only assembles frames
from persisted predictions and realized labels and groups them by the model
identity actually stored with each prediction. Replay evaluation is evidence
about online behavior — it never feeds back into Loop 4 champion selection.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from turbine_guard.data.schema import ASSET_ID_COLUMN, CYCLE_COLUMN
from turbine_guard.database.commands import NewModelEvaluation
from turbine_guard.database.enums import EvaluationScope
from turbine_guard.modeling.alerts import alert_metrics
from turbine_guard.modeling.config import AlertConfig
from turbine_guard.modeling.conformal import interval_metrics
from turbine_guard.modeling.metrics import regression_metrics
from turbine_guard.replay.errors import ReplayOutcomeError
from turbine_guard.replay.state import ReplayRunState, StoredOutcome, StoredPrediction

PER_ASSET_AGGREGATION = "replay_asset"
AGGREGATE_AGGREGATION = "replay_aggregate"

_FRAME_COLUMNS = (
    ASSET_ID_COLUMN,
    CYCLE_COLUMN,
    "y_true",
    "y_true_uncapped",
    "y_pred",
    "lower",
    "upper",
    "model_name",
    "model_version",
    "model_run_id",
    "prediction_timestamp",
)


@dataclass(frozen=True)
class DelayedEvaluationConfig:
    """Alert policy used when scoring realized outcomes; Loop 4 defaults."""

    alert: AlertConfig = field(default_factory=AlertConfig)


def build_outcome_frame(
    predictions: Sequence[StoredPrediction],
    outcomes: Sequence[StoredOutcome],
    *,
    source_asset_id: int,
) -> pd.DataFrame:
    """Join immutable predictions to their realized labels for one asset."""
    realized = {outcome.prediction_id: outcome for outcome in outcomes}
    rows: list[dict[str, Any]] = []
    for prediction in predictions:
        outcome = realized.get(prediction.prediction_id)
        if outcome is None:
            raise ReplayOutcomeError(
                f"Prediction at cycle {prediction.cycle} has no realized label; "
                "evaluation requires a completed backfill."
            )
        if outcome.cycle != prediction.cycle:
            raise ReplayOutcomeError(
                f"Realized label cycle {outcome.cycle} does not match prediction "
                f"cycle {prediction.cycle}."
            )
        rows.append(
            {
                ASSET_ID_COLUMN: source_asset_id,
                CYCLE_COLUMN: prediction.cycle,
                "y_true": float(outcome.realized_rul),
                "y_true_uncapped": float(outcome.realized_rul),
                "y_pred": float(prediction.predicted_rul),
                "lower": prediction.lower_rul,
                "upper": prediction.upper_rul,
                "model_name": prediction.model_name,
                "model_version": prediction.model_version,
                "model_run_id": prediction.model_run_id,
                "prediction_timestamp": prediction.prediction_timestamp,
            }
        )
    if not rows:
        raise ReplayOutcomeError("No labeled predictions are available for evaluation.")
    frame = pd.DataFrame(rows, columns=list(_FRAME_COLUMNS)).astype(
        {"lower": "float64", "upper": "float64"}
    )
    return frame.sort_values([ASSET_ID_COLUMN, CYCLE_COLUMN], kind="stable", ignore_index=True)


def group_by_model(frame: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
    """Split labeled rows by the exact model identity stored with each prediction."""
    return {
        (str(name), str(version)): group.reset_index(drop=True)
        for (name, version), group in frame.groupby(["model_name", "model_version"], sort=True)
    }


def evaluate_group(frame: pd.DataFrame, config: DelayedEvaluationConfig) -> dict[str, Any]:
    """Score one model version's labeled predictions with the Loop 4 formulas."""
    regression = regression_metrics(frame["y_true"], frame["y_pred"])
    critical = alert_metrics(
        frame,
        horizon=config.alert.critical_horizon,
        minimum_lead_cycles=config.alert.minimum_lead_cycles,
    )
    warning = alert_metrics(
        frame,
        horizon=config.alert.warning_horizon,
        minimum_lead_cycles=config.alert.minimum_lead_cycles,
    )
    with_interval = frame[frame["lower"].notna() & frame["upper"].notna()]
    interval = (
        interval_metrics(with_interval["y_true"], with_interval["lower"], with_interval["upper"])
        if not with_interval.empty
        else None
    )
    return {
        "regression": regression,
        "critical": critical,
        "warning": warning,
        "interval": interval,
    }


def per_asset_evaluations(
    run: ReplayRunState,
    predictions: Sequence[StoredPrediction],
    outcomes: Sequence[StoredOutcome],
    config: DelayedEvaluationConfig | None = None,
) -> list[NewModelEvaluation]:
    """Delayed per-asset evaluation commands, one per stored model version."""
    if run.failure_event_id is None:
        raise ReplayOutcomeError("Delayed evaluation requires an emitted failure event.")
    evaluation_config = config or DelayedEvaluationConfig()
    frame = build_outcome_frame(predictions, outcomes, source_asset_id=run.source_asset_id)
    commands: list[NewModelEvaluation] = []
    for (model_name, model_version), group in group_by_model(frame).items():
        results = evaluate_group(group, evaluation_config)
        metrics = _metrics_payload(
            results,
            evaluation_config,
            aggregation=PER_ASSET_AGGREGATION,
            extra={
                "replay_run_id": str(run.run_id),
                "source_asset_id": run.source_asset_id,
                "external_asset_id": run.external_asset_id,
                "attempt": run.attempt,
                "failure_event_id": str(run.failure_event_id),
                "model_run_ids": sorted(
                    {str(value) for value in group["model_run_id"].dropna().unique()}
                ),
            },
        )
        commands.append(
            _evaluation_command(
                model_name, model_version, run.dataset_subset, group, results, metrics
            )
        )
    return commands


def aggregate_evaluations(
    runs: Sequence[ReplayRunState],
    frames: Iterable[pd.DataFrame],
    config: DelayedEvaluationConfig | None = None,
) -> list[NewModelEvaluation]:
    """Aggregate delayed evaluation across completed replay runs by model version.

    Metrics from different model versions are never combined into one row; the
    aggregation label and the contributing run IDs are recorded explicitly.
    """
    evaluation_config = config or DelayedEvaluationConfig()
    combined = pd.concat(list(frames), ignore_index=True)
    if combined.empty:
        raise ReplayOutcomeError("Aggregate evaluation received no labeled predictions.")
    run_ids = sorted(str(run.run_id) for run in runs)
    source_assets = sorted({run.source_asset_id for run in runs})
    subset = runs[0].dataset_subset if runs else None
    commands: list[NewModelEvaluation] = []
    for (model_name, model_version), group in group_by_model(combined).items():
        results = evaluate_group(group, evaluation_config)
        metrics = _metrics_payload(
            results,
            evaluation_config,
            aggregation=AGGREGATE_AGGREGATION,
            extra={
                "replay_run_ids": run_ids,
                "source_asset_ids": source_assets,
                "asset_count": int(group[ASSET_ID_COLUMN].nunique()),
                "model_run_ids": sorted(
                    {str(value) for value in group["model_run_id"].dropna().unique()}
                ),
            },
        )
        commands.append(
            _evaluation_command(model_name, model_version, subset, group, results, metrics)
        )
    return commands


def _evaluation_command(
    model_name: str,
    model_version: str,
    dataset_subset: str | None,
    group: pd.DataFrame,
    results: dict[str, Any],
    metrics: dict[str, Any],
) -> NewModelEvaluation:
    timestamps = pd.to_datetime(group["prediction_timestamp"])
    window_start = timestamps.min().to_pydatetime()
    window_end = timestamps.max().to_pydatetime()
    interval = results["interval"]
    return NewModelEvaluation(
        model_name=model_name,
        model_version=model_version,
        evaluation_scope=EvaluationScope.REPLAY,
        dataset_subset=dataset_subset,
        window_start=window_start,
        window_end=window_end,
        sample_count=len(group),
        mae=results["regression"]["mae"],
        rmse=results["regression"]["rmse"],
        nasa_score=results["regression"]["nasa_score"],
        critical_precision=results["critical"]["precision"],
        critical_recall=results["critical"]["recall"],
        interval_coverage=None if interval is None else interval["empirical_coverage"],
        metrics=metrics,
    )


def _metrics_payload(
    results: dict[str, Any],
    config: DelayedEvaluationConfig,
    *,
    aggregation: str,
    extra: dict[str, Any],
) -> dict[str, Any]:
    interval = results["interval"]
    payload: dict[str, Any] = {
        "aggregation": aggregation,
        "alert_config": {
            "critical_horizon": config.alert.critical_horizon,
            "warning_horizon": config.alert.warning_horizon,
            "minimum_lead_cycles": config.alert.minimum_lead_cycles,
        },
        "r2": results["regression"]["r2"],
        "critical": _trim_alerts(results["critical"]),
        "warning": _trim_alerts(results["warning"]),
        "interval": None
        if interval is None
        else {
            "empirical_coverage": interval["empirical_coverage"],
            "average_width": interval["average_width"],
            "median_width": interval["median_width"],
            "row_count": interval["row_count"],
        },
    }
    payload.update(extra)
    return payload


def _trim_alerts(alerts: dict[str, Any]) -> dict[str, Any]:
    """Keep the operational alert summary JSON-friendly and bounded."""
    return {
        "precision": alerts["precision"],
        "recall": alerts["recall"],
        "f1": alerts["f1"],
        "pr_auc": alerts["pr_auc"],
        "alert_episode_count": alerts["alert_episode_count"],
        "false_alarm_episodes": alerts["false_alarm_episodes"],
        "false_alarms_per_1000_cycles": alerts["false_alarms_per_1000_cycles"],
        "missed_failures": alerts["missed_failures"],
        "mean_first_alert_lead_time": alerts["mean_first_alert_lead_time"],
        "median_first_alert_lead_time": alerts["median_first_alert_lead_time"],
        "timely_warning_asset_percentage": alerts["timely_warning_asset_percentage"],
        "assets_alerted_too_early": alerts["assets_alerted_too_early"],
        "assets_alerted_too_late": alerts["assets_alerted_too_late"],
        "per_asset": alerts["per_asset"],
    }

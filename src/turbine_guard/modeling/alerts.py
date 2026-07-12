"""Maintenance-alert classification and collapsed episode metrics."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

from turbine_guard.data.schema import ASSET_ID_COLUMN, CYCLE_COLUMN


def alert_metrics(
    frame: pd.DataFrame,
    *,
    horizon: int,
    minimum_lead_cycles: int = 1,
) -> dict[str, Any]:
    """Evaluate row alerts and operational first-alert episodes.

    A row is an alert when ``y_pred <= horizon``. Consecutive alert rows for
    an asset form one episode. Operational lead time is the true uncapped RUL
    at the first episode start. Starts above the horizon are early; starts
    below ``minimum_lead_cycles`` are late. An asset with no usable alert, or
    only a late first alert, is a missed failure.
    """
    if horizon <= 0:
        raise ValueError("Alert horizon must be positive.")
    required = {ASSET_ID_COLUMN, CYCLE_COLUMN, "y_true_uncapped", "y_pred"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Alert frame is missing columns: {sorted(missing)}.")

    work = frame.sort_values([ASSET_ID_COLUMN, CYCLE_COLUMN], kind="stable").copy()
    actual = (work["y_true_uncapped"] <= horizon).to_numpy(dtype="bool")
    predicted = (work["y_pred"] <= horizon).to_numpy(dtype="bool")
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual,
        predicted,
        average="binary",
        zero_division=0,
    )
    pr_auc = (
        float(average_precision_score(actual, -work["y_pred"].to_numpy(dtype="float64")))
        if bool(actual.any())
        else None
    )

    work["alert"] = predicted
    previous = work.groupby(ASSET_ID_COLUMN, sort=False)["alert"].shift(fill_value=False)
    work["episode_start"] = work["alert"] & ~previous
    starts = work[work["episode_start"]]

    per_asset: list[dict[str, Any]] = []
    early_episode_count = int((starts["y_true_uncapped"] > horizon).sum())
    for asset_id, group in work.groupby(ASSET_ID_COLUMN, sort=True):
        first = group[group["episode_start"]].head(1)
        lead = None if first.empty else float(first.iloc[0]["y_true_uncapped"])
        too_early = lead is not None and lead > horizon
        too_late = lead is not None and lead < minimum_lead_cycles
        timely = lead is not None and minimum_lead_cycles <= lead <= horizon
        missed = lead is None or too_late
        per_asset.append(
            {
                "asset_id": int(str(asset_id)),
                "first_alert_cycle": None if first.empty else int(first.iloc[0][CYCLE_COLUMN]),
                "first_alert_lead_time": lead,
                "alert_episode_count": int(group["episode_start"].sum()),
                "timely": timely,
                "too_early": too_early,
                "too_late": too_late,
                "missed_failure": missed,
            }
        )

    leads = [
        float(record["first_alert_lead_time"])
        for record in per_asset
        if record["first_alert_lead_time"] is not None
    ]
    asset_count = len(per_asset)
    false_positive_rows = int((predicted & ~actual).sum())
    false_negative_rows = int((~predicted & actual).sum())
    return {
        "definition": {
            "horizon_cycles": horizon,
            "minimum_lead_cycles": minimum_lead_cycles,
            "row_alert": "predicted_rul <= horizon",
            "episode": "consecutive alert rows collapsed; first episode controls intervention",
        },
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": pr_auc,
        "false_positive_rows": false_positive_rows,
        "false_negative_rows": false_negative_rows,
        "alert_episode_count": int(work["episode_start"].sum()),
        "false_alarm_episodes": early_episode_count,
        "false_alarms_per_1000_cycles": 1000.0 * early_episode_count / len(work),
        "missed_failures": sum(bool(record["missed_failure"]) for record in per_asset),
        "mean_first_alert_lead_time": float(np.mean(leads)) if leads else None,
        "median_first_alert_lead_time": float(np.median(leads)) if leads else None,
        "timely_warning_asset_percentage": (
            100.0 * sum(bool(record["timely"]) for record in per_asset) / asset_count
            if asset_count
            else 0.0
        ),
        "assets_alerted_too_early": sum(bool(record["too_early"]) for record in per_asset),
        "assets_alerted_too_late": sum(bool(record["too_late"]) for record in per_asset),
        "per_asset": per_asset,
    }

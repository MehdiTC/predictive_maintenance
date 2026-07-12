"""Reusable regression metrics and lifecycle/asset slice evaluation."""

from collections.abc import Iterable
from math import sqrt
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from turbine_guard.data.schema import ASSET_ID_COLUMN, CYCLE_COLUMN


def nasa_asymmetric_score(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """Return the NASA C-MAPSS asymmetric score.

    With error ``d = prediction - truth``, underprediction is penalized as
    ``exp(-d / 13) - 1`` and overprediction as ``exp(d / 10) - 1``. The
    latter is steeper because predicting too much remaining life is riskier.
    """
    truth = np.asarray(list(y_true), dtype="float64")
    prediction = np.asarray(list(y_pred), dtype="float64")
    if truth.shape != prediction.shape or truth.size == 0:
        raise ValueError("NASA score requires non-empty arrays with equal shape.")
    error = prediction - truth
    terms = np.where(error < 0, np.expm1(-error / 13.0), np.expm1(error / 10.0))
    return float(np.sum(terms))


def regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float | None]:
    """Calculate MAE, RMSE, contextual R-squared, and NASA score."""
    truth = np.asarray(list(y_true), dtype="float64")
    prediction = np.asarray(list(y_pred), dtype="float64")
    if truth.shape != prediction.shape or truth.size == 0:
        raise ValueError("Regression metrics require non-empty arrays with equal shape.")
    if not bool(np.isfinite(truth).all() and np.isfinite(prediction).all()):
        raise ValueError("Regression metrics require finite values.")
    r2 = float(r2_score(truth, prediction)) if truth.size >= 2 else None
    return {
        "mae": float(mean_absolute_error(truth, prediction)),
        "rmse": sqrt(float(mean_squared_error(truth, prediction))),
        "r2": r2,
        "nasa_score": nasa_asymmetric_score(truth, prediction),
    }


def evaluate_regression_frame(frame: pd.DataFrame) -> dict[str, Any]:
    """Evaluate row-weighted, per-asset, asset-balanced, and lifecycle slices.

    Required columns are ``asset_id``, ``cycle``, ``y_true``, ``y_true_uncapped``,
    and ``y_pred``. Lifecycle stage and total trajectory length use realized
    labels for evaluation only; neither is a model feature.
    """
    required = {
        ASSET_ID_COLUMN,
        CYCLE_COLUMN,
        "y_true",
        "y_true_uncapped",
        "y_pred",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Regression evaluation frame is missing columns: {sorted(missing)}.")

    row_weighted = regression_metrics(frame["y_true"], frame["y_pred"])
    per_asset: list[dict[str, Any]] = []
    for asset_id, group in frame.groupby(ASSET_ID_COLUMN, sort=True):
        per_asset.append(
            {
                "asset_id": int(str(asset_id)),
                "row_count": len(group),
                **regression_metrics(group["y_true"], group["y_pred"]),
            }
        )
    asset_balanced = _mean_metric_records(per_asset)
    slices = lifecycle_slice_metrics(frame)
    return {
        "row_weighted": row_weighted,
        "asset_balanced": asset_balanced,
        "per_asset": per_asset,
        "slices": slices,
    }


def lifecycle_slice_metrics(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Regression metrics by early/middle/late stage and trajectory-length group."""
    work = frame.copy()
    total_life = work[CYCLE_COLUMN].astype("float64") + work["y_true_uncapped"]
    progress = work[CYCLE_COLUMN].astype("float64") / total_life.replace(0.0, np.nan)
    work["lifecycle_stage"] = pd.cut(
        progress,
        bins=[-np.inf, 1.0 / 3.0, 2.0 / 3.0, np.inf],
        labels=["early", "middle", "late"],
        include_lowest=True,
    ).astype("string")

    work["realized_total_life"] = work[CYCLE_COLUMN].astype("float64") + work["y_true_uncapped"]
    asset_lengths = work.groupby(ASSET_ID_COLUMN)["realized_total_life"].max()
    median_length = float(asset_lengths.median())
    length_map = {
        int(str(asset)): "short" if float(length) <= median_length else "long"
        for asset, length in asset_lengths.items()
    }
    work["trajectory_length_group"] = work[ASSET_ID_COLUMN].map(length_map)

    records: list[dict[str, Any]] = []
    for slice_name in ("lifecycle_stage", "trajectory_length_group"):
        for value, group in work.groupby(slice_name, observed=True, sort=True):
            records.append(
                {
                    "slice": slice_name,
                    "value": str(value),
                    "row_count": len(group),
                    "asset_count": int(group[ASSET_ID_COLUMN].nunique()),
                    **regression_metrics(group["y_true"], group["y_pred"]),
                }
            )
    return records


def common_domain_metrics(frame: pd.DataFrame, maximum_rul: int) -> dict[str, float | None]:
    """Comparable validation metrics where capped and uncapped truth are identical."""
    common = frame[frame["y_true_uncapped"] <= maximum_rul]
    if common.empty:
        raise ValueError("No rows fall inside the configured common target domain.")
    return regression_metrics(common["y_true_uncapped"], common["y_pred"])


def _mean_metric_records(records: list[dict[str, Any]]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for metric in ("mae", "rmse", "r2", "nasa_score"):
        values = [float(record[metric]) for record in records if record[metric] is not None]
        result[metric] = float(np.mean(values)) if values else None
    return result

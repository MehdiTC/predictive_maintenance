"""Split conformal RUL intervals calibrated only on the calibration role."""

from dataclasses import dataclass
from math import ceil
from typing import Any

import numpy as np
import pandas as pd


def finite_sample_residual_quantile(residuals: np.ndarray, coverage: float) -> float:
    """Finite-sample corrected order statistic for absolute residuals.

    The selected one-indexed rank is ``ceil((n + 1) * coverage)``. When the
    nominal rank exceeds ``n``, the maximum residual is used; this is the most
    conservative finite interval available from the observed calibration set.
    """
    values = np.asarray(residuals, dtype="float64")
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Conformal calibration requires a non-empty residual vector.")
    if not 0.0 < coverage < 1.0:
        raise ValueError("Conformal coverage must be strictly between 0 and 1.")
    if not bool(np.isfinite(values).all()) or bool((values < 0).any()):
        raise ValueError("Conformal residuals must be finite and non-negative.")
    rank = min(int(values.size), ceil((values.size + 1) * coverage))
    return float(np.sort(values)[rank - 1])


@dataclass
class SplitConformalCalibrator:
    """Symmetric absolute-residual split conformal calibrator."""

    coverage: float = 0.90
    residual_quantile_: float | None = None
    calibration_size_: int = 0

    def fit(self, y_true: Any, y_pred: Any) -> "SplitConformalCalibrator":
        """Fit the interval radius from calibration residuals only."""
        truth = np.asarray(y_true, dtype="float64")
        prediction = np.asarray(y_pred, dtype="float64")
        if truth.shape != prediction.shape:
            raise ValueError("Calibration truth and predictions must have equal shape.")
        residuals = np.abs(truth - prediction)
        self.residual_quantile_ = finite_sample_residual_quantile(residuals, self.coverage)
        self.calibration_size_ = int(residuals.size)
        return self

    def intervals(self, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
        """Return lower/upper bounds, clipping physically impossible lower RUL at zero."""
        if self.residual_quantile_ is None:
            raise ValueError("SplitConformalCalibrator has not been fitted.")
        prediction = np.asarray(y_pred, dtype="float64")
        lower = np.maximum(0.0, prediction - self.residual_quantile_)
        upper = prediction + self.residual_quantile_
        return lower, upper

    def record(self) -> dict[str, float | int | str]:
        """Machine-readable fitted calibration information."""
        if self.residual_quantile_ is None:
            raise ValueError("SplitConformalCalibrator has not been fitted.")
        return {
            "method": "row_level_absolute_residual_split_conformal",
            "coverage": self.coverage,
            "residual_quantile": self.residual_quantile_,
            "calibration_rows": self.calibration_size_,
            "limitation": (
                "Rows within an asset trajectory are temporally dependent; nominal finite-sample "
                "exchangeability guarantees therefore do not strictly apply."
            ),
        }


def interval_metrics(
    y_true: Any,
    lower: Any,
    upper: Any,
    lifecycle_stage: pd.Series | None = None,
) -> dict[str, Any]:
    """Empirical interval coverage and width overall and optionally by lifecycle stage."""
    truth = np.asarray(y_true, dtype="float64")
    low = np.asarray(lower, dtype="float64")
    high = np.asarray(upper, dtype="float64")
    if not (truth.shape == low.shape == high.shape) or truth.size == 0:
        raise ValueError("Interval metrics require non-empty arrays with equal shape.")
    covered = (truth >= low) & (truth <= high)
    width = high - low
    result: dict[str, Any] = {
        "row_count": int(truth.size),
        "empirical_coverage": float(np.mean(covered)),
        "average_width": float(np.mean(width)),
        "median_width": float(np.median(width)),
    }
    if lifecycle_stage is not None:
        if len(lifecycle_stage) != truth.size:
            raise ValueError("Lifecycle-stage labels must align with interval rows.")
        stages: list[dict[str, Any]] = []
        labels = lifecycle_stage.astype("string").to_numpy()
        for stage in sorted({str(value) for value in labels}):
            mask = labels == stage
            stages.append(
                {
                    "stage": stage,
                    "row_count": int(mask.sum()),
                    "empirical_coverage": float(np.mean(covered[mask])),
                    "average_width": float(np.mean(width[mask])),
                    "median_width": float(np.median(width[mask])),
                }
            )
        result["by_lifecycle_stage"] = stages
    return result


def lifecycle_stages(cycle: Any, uncapped_rul: Any) -> pd.Series:
    """Truth-derived early/middle/late stage labels for evaluation only."""
    cycle_values = np.asarray(cycle, dtype="float64")
    rul_values = np.asarray(uncapped_rul, dtype="float64")
    progress = cycle_values / (cycle_values + rul_values)
    return pd.Series(
        np.where(progress <= 1 / 3, "early", np.where(progress <= 2 / 3, "middle", "late")),
        dtype="string",
    )

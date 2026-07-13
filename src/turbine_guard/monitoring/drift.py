"""Transparent PSI, Wasserstein, missingness, mean, and standard-deviation drift."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from turbine_guard.database.enums import DriftStatus
from turbine_guard.monitoring.config import DriftThresholds
from turbine_guard.monitoring.reference import FeatureReference, TrainingReference

_EPSILON = 1e-6


@dataclass(frozen=True)
class DriftResult:
    status: DriftStatus
    drifted_feature_count: int
    max_psi: float | None
    max_wasserstein: float | None
    details: dict[str, Any]


def feature_drift_report(
    current: pd.DataFrame,
    *,
    reference: TrainingReference,
    thresholds: DriftThresholds,
) -> DriftResult:
    """Compare one feature window against the champion's training-only distributions."""
    if tuple(current.columns) != reference.feature_columns:
        raise ValueError("Current feature order does not match the training reference.")
    if len(current) < thresholds.minimum_rows:
        return DriftResult(
            DriftStatus.INSUFFICIENT_DATA,
            0,
            None,
            None,
            {
                "row_count": len(current),
                "minimum_rows": thresholds.minimum_rows,
                "features": [],
            },
        )

    records: list[dict[str, Any]] = []
    for column in reference.feature_columns:
        records.append(
            _feature_metrics(current[column], reference.features[column], thresholds, column)
        )
    drifted = [record for record in records if record["drifted"]]
    warning = [record for record in records if record["warning"]]
    if drifted:
        status = DriftStatus.DETECTED
    elif warning:
        status = DriftStatus.WARNING
    else:
        status = DriftStatus.NOT_DETECTED
    psi_values = [float(record["psi"]) for record in records if record["psi"] is not None]
    wasserstein_values = [
        float(record["wasserstein"]) for record in records if record["wasserstein"] is not None
    ]
    return DriftResult(
        status=status,
        drifted_feature_count=len(drifted),
        max_psi=max(psi_values, default=None),
        max_wasserstein=max(wasserstein_values, default=None),
        details={
            "row_count": len(current),
            "reference_row_count": reference.row_count,
            "reference_model_version": reference.model_version,
            "reference_feature_version": reference.feature_version,
            "thresholds": {
                "psi_warning": thresholds.psi_warning,
                "psi_detected": thresholds.psi_detected,
                "normalized_wasserstein": thresholds.normalized_wasserstein,
                "missingness_shift": thresholds.missingness_shift,
                "mean_shift": thresholds.mean_shift,
                "std_shift": thresholds.std_shift,
            },
            "features": records,
        },
    )


def population_stability_index(current: np.ndarray, reference: FeatureReference) -> float | None:
    """PSI using training-quantile bins and epsilon-smoothed proportions."""
    finite = current[np.isfinite(current)]
    if finite.size == 0 or not reference.bin_probabilities:
        return None
    edges = np.concatenate(([-np.inf], np.asarray(reference.bin_edges, dtype="float64"), [np.inf]))
    counts, _ = np.histogram(finite, bins=edges)
    actual = counts.astype("float64") / finite.size
    expected = np.asarray(reference.bin_probabilities, dtype="float64")
    actual = np.clip(actual, _EPSILON, None)
    expected = np.clip(expected, _EPSILON, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def wasserstein_distance(current: np.ndarray, reference: FeatureReference) -> float | None:
    """One-dimensional W1 distance via the integral of empirical quantile differences."""
    finite = current[np.isfinite(current)]
    if finite.size == 0 or not reference.quantiles:
        return None
    probabilities = np.linspace(0.0, 1.0, len(reference.quantiles))
    current_quantiles = np.quantile(finite, probabilities)
    difference = np.abs(current_quantiles - np.asarray(reference.quantiles, dtype="float64"))
    return float(np.trapezoid(difference, probabilities))


def _feature_metrics(
    series: pd.Series,
    reference: FeatureReference,
    thresholds: DriftThresholds,
    name: str,
) -> dict[str, Any]:
    values = series.to_numpy(dtype="float64")
    finite = values[np.isfinite(values)]
    missing_rate = float(1.0 - finite.size / values.size)
    missingness_shift = abs(missing_rate - reference.missing_rate)
    if finite.size < thresholds.minimum_non_null or reference.mean is None or reference.std is None:
        return {
            "feature": name,
            "non_null_count": int(finite.size),
            "psi": None,
            "wasserstein": None,
            "normalized_wasserstein": None,
            "missingness_shift": missingness_shift,
            "mean_shift": None,
            "std_shift": None,
            "warning": missingness_shift >= thresholds.missingness_shift,
            "drifted": missingness_shift >= thresholds.missingness_shift,
            "insufficient_non_null": True,
        }
    current_mean = float(np.mean(finite))
    current_std = float(np.std(finite, ddof=0))
    scale = max(reference.std, abs(reference.mean) * 1e-12, 1e-12)
    mean_shift = abs(current_mean - reference.mean) / scale
    std_shift = abs(current_std - reference.std) / scale
    psi = population_stability_index(values, reference)
    wasserstein = wasserstein_distance(values, reference)
    normalized_wasserstein = None if wasserstein is None else wasserstein / scale
    detected_checks = {
        "psi": psi is not None and psi >= thresholds.psi_detected,
        "wasserstein": (
            normalized_wasserstein is not None
            and normalized_wasserstein >= thresholds.normalized_wasserstein
        ),
        "missingness": missingness_shift >= thresholds.missingness_shift,
        "mean": mean_shift >= thresholds.mean_shift,
        "std": std_shift >= thresholds.std_shift,
    }
    warning = (psi is not None and psi >= thresholds.psi_warning) or any(detected_checks.values())
    return {
        "feature": name,
        "non_null_count": int(finite.size),
        "reference_missing_rate": reference.missing_rate,
        "current_missing_rate": missing_rate,
        "reference_mean": reference.mean,
        "current_mean": current_mean,
        "reference_std": reference.std,
        "current_std": current_std,
        "psi": psi,
        "wasserstein": wasserstein,
        "normalized_wasserstein": normalized_wasserstein,
        "missingness_shift": missingness_shift,
        "mean_shift": mean_shift,
        "std_shift": std_shift,
        "triggered_checks": [name for name, failed in detected_checks.items() if failed],
        "warning": warning,
        "drifted": any(detected_checks.values()),
        "insufficient_non_null": False,
    }

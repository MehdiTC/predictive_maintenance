"""Split conformal finite-sample, coverage, clipping, and serialization tests."""

from pathlib import Path

import numpy as np
import pytest

from turbine_guard.modeling.artifacts import load_joblib, serialize_joblib
from turbine_guard.modeling.conformal import (
    SplitConformalCalibrator,
    finite_sample_residual_quantile,
    interval_metrics,
)


def test_finite_sample_corrected_residual_quantile() -> None:
    residuals = np.array([1.0, 2.0, 3.0, 4.0])
    # ceil((4 + 1) * 0.6) = 3 -> third order statistic.
    assert finite_sample_residual_quantile(residuals, 0.6) == 3.0


def test_unattainable_rank_uses_conservative_maximum() -> None:
    assert finite_sample_residual_quantile(np.array([1.0, 5.0]), 0.95) == 5.0


def test_bounds_clip_lower_and_coverage_width() -> None:
    calibrator = SplitConformalCalibrator(coverage=0.75).fit(
        [1.0, 2.0, 3.0, 4.0],
        [0.0, 2.0, 5.0, 4.0],
    )
    lower, upper = calibrator.intervals([0.5, 3.0])
    assert lower[0] == 0.0
    metrics = interval_metrics([1.0, 3.0], lower, upper)
    assert metrics["empirical_coverage"] == 1.0
    assert metrics["average_width"] >= metrics["median_width"] - 1e-12


def test_calibrator_serialization_reload(tmp_path: Path) -> None:
    calibrator = SplitConformalCalibrator(0.8).fit([1, 2, 3, 4, 5], [2, 2, 2, 4, 4])
    expected = calibrator.intervals([2.0, 7.0])
    path = tmp_path / "conformal.joblib"
    path.write_bytes(serialize_joblib(calibrator))
    restored = load_joblib(path)
    actual = restored.intervals([2.0, 7.0])
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


def test_fit_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="equal shape"):
        SplitConformalCalibrator().fit([1, 2], [1])

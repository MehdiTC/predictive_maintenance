"""Unit coverage for Loop 9 quality, drift, performance, and trigger decisions."""

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from turbine_guard.data.schema import OPERATING_SETTING_COLUMNS, SENSOR_COLUMNS
from turbine_guard.database.enums import DataQualityStatus, DriftStatus
from turbine_guard.monitoring.config import DriftThresholds, TriggerThresholds
from turbine_guard.monitoring.decisions import TriggerAction, decide_retraining
from turbine_guard.monitoring.drift import (
    feature_drift_report,
    population_stability_index,
    wasserstein_distance,
)
from turbine_guard.monitoring.performance import delayed_performance_report
from turbine_guard.monitoring.quality import data_quality_report
from turbine_guard.monitoring.reference import (
    FeatureReference,
    TrainingReference,
    build_training_reference,
)


def _feature_reference(values: np.ndarray) -> FeatureReference:
    internal = np.quantile(values, np.linspace(0.1, 0.9, 9))
    edges = np.concatenate(([-np.inf], internal, [np.inf]))
    counts, _ = np.histogram(values, bins=edges)
    return FeatureReference(
        count=len(values),
        missing_rate=0.0,
        mean=float(values.mean()),
        std=float(values.std()),
        minimum=float(values.min()),
        maximum=float(values.max()),
        bin_edges=tuple(internal),
        bin_probabilities=tuple(counts / len(values)),
        quantiles=tuple(np.quantile(values, np.linspace(0.0, 1.0, 101))),
    )


def _reference(columns: tuple[str, ...], values: np.ndarray | None = None) -> TrainingReference:
    source = np.arange(100, dtype="float64") if values is None else values
    return TrainingReference(
        reference_version="1",
        created_at=datetime.now(UTC),
        model_name="model",
        model_version="7",
        feature_version="1",
        feature_manifest_sha256="a" * 64,
        training_parquet_sha256="b" * 64,
        row_count=len(source),
        asset_count=10,
        feature_columns=columns,
        quantile_probabilities=tuple(np.linspace(0.0, 1.0, 101)),
        features={column: _feature_reference(source) for column in columns},
    )


def _quality_frame(*, unhealthy: bool = False) -> pd.DataFrame:
    measurements = (*OPERATING_SETTING_COLUMNS, *SENSOR_COLUMNS)
    rows = []
    for cycle in range(1, 26):
        row = {"asset_id": 1, "cycle": cycle}
        row.update(dict.fromkeys(measurements, 49.5))
        rows.append(row)
    frame = pd.DataFrame(rows)
    if unhealthy:
        frame.loc[0, "sensor_01"] = np.nan
        frame.loc[1, "sensor_02"] = 1_000.0
        frame.loc[2, "cycle"] = 1
    return frame


def _quality_reference() -> TrainingReference:
    columns = tuple(f"{column}_current" for column in (*OPERATING_SETTING_COLUMNS, *SENSOR_COLUMNS))
    return _reference(columns)


def _drift_thresholds() -> DriftThresholds:
    return DriftThresholds(
        minimum_rows=50,
        minimum_non_null=10,
        psi_warning=0.1,
        psi_detected=0.25,
        normalized_wasserstein=0.2,
        missingness_shift=0.05,
        mean_shift=0.5,
        std_shift=0.5,
    )


def _trigger_thresholds() -> TriggerThresholds:
    return TriggerThresholds(
        minimum_assets=5,
        minimum_rows=500,
        minimum_holdout_assets=2,
        drifted_feature_count=3,
        performance_degradation=0.15,
        minimum_critical_recall=0.6,
        false_alarm_increase_tolerance=50.0,
        minimum_coverage=0.85,
        interval_days=30,
    )


def test_healthy_and_unhealthy_data_quality_windows() -> None:
    healthy = data_quality_report(
        _quality_frame(),
        reference=_quality_reference(),
        minimum_rows=20,
        minimum_assets=1,
        sufficient_history_cycles=20,
        out_of_range_stddevs=8.0,
    )
    assert healthy.status is DataQualityStatus.PASS
    assert healthy.failure_count == 0

    unhealthy = data_quality_report(
        _quality_frame(unhealthy=True),
        reference=_quality_reference(),
        minimum_rows=20,
        minimum_assets=1,
        sufficient_history_cycles=20,
        out_of_range_stddevs=8.0,
        rejected_records=2,
    )
    assert unhealthy.status is DataQualityStatus.FAIL
    assert unhealthy.details["missing_value_count"] == 1
    assert unhealthy.details["duplicate_record_count"] == 1
    assert unhealthy.details["out_of_range_value_count"] >= 1


def test_insufficient_quality_window_is_explicit() -> None:
    result = data_quality_report(
        _quality_frame().head(3),
        reference=_quality_reference(),
        minimum_rows=20,
        minimum_assets=1,
        sufficient_history_cycles=20,
        out_of_range_stddevs=8.0,
    )
    assert result.status is DataQualityStatus.INSUFFICIENT_DATA


def test_no_drift_and_deliberately_induced_drift() -> None:
    reference = _reference(("feature",))
    stable = feature_drift_report(
        pd.DataFrame({"feature": np.arange(100, dtype="float64")}),
        reference=reference,
        thresholds=_drift_thresholds(),
    )
    assert stable.status is DriftStatus.NOT_DETECTED
    assert stable.drifted_feature_count == 0

    shifted = feature_drift_report(
        pd.DataFrame({"feature": np.arange(100, dtype="float64") + 100.0}),
        reference=reference,
        thresholds=_drift_thresholds(),
    )
    assert shifted.status is DriftStatus.DETECTED
    assert shifted.drifted_feature_count == 1
    feature = shifted.details["features"][0]
    assert feature["psi"] > 0.25
    assert feature["wasserstein"] == pytest.approx(100.0)
    assert "mean" in feature["triggered_checks"]


def test_psi_wasserstein_and_missingness_shift_are_reported() -> None:
    reference = _feature_reference(np.arange(100, dtype="float64"))
    current = np.concatenate((np.arange(50, dtype="float64") + 50.0, [np.nan] * 10))
    assert population_stability_index(current, reference) is not None
    assert wasserstein_distance(current, reference) is not None
    report = feature_drift_report(
        pd.DataFrame({"feature": current}),
        reference=_reference(("feature",)),
        thresholds=_drift_thresholds(),
    )
    assert report.details["features"][0]["missingness_shift"] == pytest.approx(1 / 6)


def test_delayed_performance_reuses_full_metric_set_and_shows_degradation() -> None:
    frame = pd.DataFrame(
        {
            "asset_id": [1] * 60,
            "cycle": range(1, 61),
            "y_true": np.arange(59, -1, -1, dtype="float64"),
            "y_true_uncapped": np.arange(59, -1, -1, dtype="float64"),
            "y_pred": np.full(60, 80.0),
            "lower": np.full(60, 75.0),
            "upper": np.full(60, 85.0),
        }
    )
    result = delayed_performance_report(frame)
    assert result.metrics["regression"]["rmse"] > 40
    assert result.metrics["critical"]["recall"] == 0.0
    assert result.metrics["interval"]["empirical_coverage"] < 0.2
    assert set(result.metrics["risk_distribution"]) == {"healthy", "warning", "critical"}


@pytest.mark.parametrize(
    ("expected", "kwargs"),
    [
        (TriggerAction.NO_ACTION, {}),
        (TriggerAction.MONITOR, {"drift_status": DriftStatus.WARNING}),
        (
            TriggerAction.RETRAIN,
            {"manual_force": True, "newly_labeled_assets": 5, "newly_labeled_rows": 500},
        ),
        (TriggerAction.BLOCKED, {"manual_force": True}),
        (TriggerAction.BLOCKED, {"data_quality_status": DataQualityStatus.FAIL}),
    ],
)
def test_all_trigger_decisions(expected: TriggerAction, kwargs: dict[str, object]) -> None:
    arguments: dict[str, object] = {
        "thresholds": _trigger_thresholds(),
        "data_quality_status": DataQualityStatus.PASS,
        "drift_status": DriftStatus.NOT_DETECTED,
        "drifted_feature_count": 0,
        "newly_labeled_assets": 0,
        "newly_labeled_rows": 0,
        "current_metrics": None,
        "baseline_metrics": None,
        "interval_elapsed": False,
        "safe_holdout_available": True,
        "manual_force": False,
    }
    arguments.update(kwargs)
    decision = decide_retraining(**arguments)  # type: ignore[arg-type]
    assert decision.action is expected


def test_performance_drift_recall_false_alarm_and_coverage_triggers() -> None:
    current = {
        "regression": {"mae": 13.0, "rmse": 13.0},
        "critical": {"recall": 0.5, "false_alarms_per_1000_cycles": 90.0},
        "interval": {"empirical_coverage": 0.7},
    }
    baseline = {
        "regression": {"mae": 10.0, "rmse": 10.0},
        "critical": {"recall": 0.8, "false_alarms_per_1000_cycles": 10.0},
        "interval": {"empirical_coverage": 0.9},
    }
    decision = decide_retraining(
        thresholds=_trigger_thresholds(),
        data_quality_status=DataQualityStatus.PASS,
        drift_status=DriftStatus.DETECTED,
        drifted_feature_count=4,
        newly_labeled_assets=5,
        newly_labeled_rows=500,
        current_metrics=current,
        baseline_metrics=baseline,
        interval_elapsed=False,
        safe_holdout_available=True,
    )
    assert decision.action is TriggerAction.RETRAIN
    assert {
        "feature_drift",
        "rmse_degradation",
        "mae_degradation",
        "critical_recall",
        "false_alarm_increase",
        "coverage_degradation",
    } <= set(decision.reasons)


def test_training_reference_reads_only_training_role_and_is_versioned(
    feature_data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = pd.read_parquet
    calls: list[Path] = []

    def read(path: object, *args: object, **kwargs: object) -> pd.DataFrame:
        calls.append(Path(str(path)))
        return original(path, *args, **kwargs)

    monkeypatch.setattr("turbine_guard.monitoring.reference.pd.read_parquet", read)
    first = build_training_reference(
        data_dir=feature_data_dir,
        model_name="model",
        model_version="7",
        expected_feature_version="1",
    )
    second = build_training_reference(
        data_dir=feature_data_dir,
        model_name="model",
        model_version="7",
        expected_feature_version="1",
    )
    assert [path.name for path in calls] == ["train.parquet"]
    assert first.reference.row_count > 0
    assert first.reference.model_version == "7"
    assert first.sha256 == second.sha256
    assert first.path == second.path

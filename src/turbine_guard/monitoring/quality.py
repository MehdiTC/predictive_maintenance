"""Deterministic data-quality reports for operational sensor windows."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from turbine_guard.data.schema import (
    ASSET_ID_COLUMN,
    CYCLE_COLUMN,
    OPERATING_SETTING_COLUMNS,
    SENSOR_COLUMNS,
)
from turbine_guard.database.enums import DataQualityStatus
from turbine_guard.monitoring.reference import TrainingReference

_MEASUREMENTS = (*OPERATING_SETTING_COLUMNS, *SENSOR_COLUMNS)


@dataclass(frozen=True)
class DataQualityResult:
    status: DataQualityStatus
    record_count: int
    asset_count: int
    failure_count: int
    details: dict[str, Any]


def data_quality_report(
    frame: pd.DataFrame,
    *,
    reference: TrainingReference,
    minimum_rows: int,
    minimum_assets: int,
    sufficient_history_cycles: int,
    out_of_range_stddevs: float,
    rejected_records: int = 0,
) -> DataQualityResult:
    """Validate counts, schema completeness, sequence integrity, ranges, and availability."""
    required = {ASSET_ID_COLUMN, CYCLE_COLUMN, *_MEASUREMENTS}
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        return DataQualityResult(
            status=DataQualityStatus.FAIL,
            record_count=len(frame),
            asset_count=0,
            failure_count=len(missing_columns),
            details={"missing_columns": missing_columns},
        )

    record_count = len(frame)
    asset_count = int(frame[ASSET_ID_COLUMN].nunique()) if record_count else 0
    duplicate_count = int(frame.duplicated([ASSET_ID_COLUMN, CYCLE_COLUMN]).sum())
    values = frame.loc[:, list(_MEASUREMENTS)].to_numpy(dtype="float64")
    missing_value_count = int(np.isnan(values).sum())
    non_finite_count = int((~np.isfinite(values) & ~np.isnan(values)).sum())
    out_of_order_assets, gap_count = _sequence_failures(frame)
    out_of_range_count, out_of_range_by_column = _range_failures(
        frame, reference, out_of_range_stddevs
    )
    maximum_cycle: pd.Series[Any] = (
        frame.groupby(ASSET_ID_COLUMN)[CYCLE_COLUMN].max()
        if record_count
        else pd.Series(dtype="int64")
    )
    insufficient_history_assets = (
        int((maximum_cycle < sufficient_history_cycles).sum()) if record_count else 0
    )
    availability = {
        column: float(frame[column].notna().mean()) if record_count else 0.0
        for column in _MEASUREMENTS
    }
    hard_failures = (
        rejected_records
        + duplicate_count
        + missing_value_count
        + non_finite_count
        + out_of_order_assets
        + gap_count
        + out_of_range_count
    )
    if hard_failures:
        status = DataQualityStatus.FAIL
    elif record_count < minimum_rows or asset_count < minimum_assets:
        status = DataQualityStatus.INSUFFICIENT_DATA
    elif insufficient_history_assets:
        status = DataQualityStatus.WARNING
    else:
        status = DataQualityStatus.PASS
    details: dict[str, Any] = {
        "incoming_record_count": record_count,
        "rejected_record_count": rejected_records,
        "duplicate_record_count": duplicate_count,
        "missing_value_count": missing_value_count,
        "non_finite_value_count": non_finite_count,
        "out_of_range_value_count": out_of_range_count,
        "out_of_range_by_column": out_of_range_by_column,
        "out_of_order_asset_count": out_of_order_assets,
        "cycle_gap_count": gap_count,
        "insufficient_history_asset_count": insufficient_history_assets,
        "sufficient_history_cycles": sufficient_history_cycles,
        "sensor_availability": availability,
        "rejected_record_source": (
            "explicit_window_input; PostgreSQL stores accepted records only"
        ),
        "reference_model_version": reference.model_version,
        "reference_feature_version": reference.feature_version,
    }
    return DataQualityResult(status, record_count, asset_count, hard_failures, details)


def _sequence_failures(frame: pd.DataFrame) -> tuple[int, int]:
    out_of_order_assets = 0
    gap_count = 0
    for _, group in frame.groupby(ASSET_ID_COLUMN, sort=False):
        cycles = group[CYCLE_COLUMN].to_numpy(dtype="int64")
        if cycles.size > 1 and bool((np.diff(cycles) < 0).any()):
            out_of_order_assets += 1
        unique = np.unique(cycles)
        if unique.size > 1:
            gap_count += int(np.maximum(0, np.diff(unique) - 1).sum())
    return out_of_order_assets, gap_count


def _range_failures(
    frame: pd.DataFrame,
    reference: TrainingReference,
    stddevs: float,
) -> tuple[int, dict[str, int]]:
    result: dict[str, int] = {}
    for column in _MEASUREMENTS:
        feature = reference.features.get(f"{column}_current")
        if feature is None or feature.mean is None or feature.std is None:
            continue
        scale = max(feature.std, 1e-12)
        lower = feature.mean - stddevs * scale
        upper = feature.mean + stddevs * scale
        count = int(((frame[column] < lower) | (frame[column] > upper)).sum())
        if count:
            result[column] = count
    return sum(result.values()), result

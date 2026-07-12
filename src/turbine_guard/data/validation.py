"""Structural and semantic validation of parsed C-MAPSS frames.

Validation produces structured, machine-readable results (pydantic models)
instead of scattered assertions. Checks are either *required* — a failure
must block publication of processed outputs — or *warnings*, which are
recorded without failing the run (for example constant sensor columns, which
are a real property of the dataset rather than a defect).

General structural validation applies to any C-MAPSS-shaped frame. The
canonical FD001 row/asset counts are a separate, optional
:class:`CanonicalProfile` so the validator is not usable only for one exact
dataset.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from turbine_guard.data.schema import (
    ASSET_ID_COLUMN,
    CYCLE_COLUMN,
    RUL_COLUMN,
    RUL_DTYPES,
    TRAJECTORY_DTYPES,
    TRAJECTORY_FLOAT_COLUMNS,
)

logger = logging.getLogger(__name__)

NEAR_CONSTANT_RELATIVE_STD = 1e-4
"""A column is near-constant when std / max(|mean|, eps) falls below this."""


@dataclass(frozen=True)
class CanonicalProfile:
    """Expected high-level counts for a specific, known acquisition."""

    subset: str
    train_rows: int
    train_assets: int
    test_rows: int
    test_assets: int
    rul_values: int


FD001_CANONICAL_PROFILE = CanonicalProfile(
    subset="FD001",
    train_rows=20_631,
    train_assets=100,
    test_rows=13_096,
    test_assets=100,
    rul_values=100,
)

CANONICAL_PROFILES: dict[str, CanonicalProfile] = {"FD001": FD001_CANONICAL_PROFILE}


class ValidationCheck(BaseModel):
    """One named validation check and its outcome."""

    model_config = ConfigDict(frozen=True)

    name: str
    passed: bool
    required: bool
    message: str


class TrajectoryStats(BaseModel):
    """Descriptive statistics for a validated trajectory frame."""

    model_config = ConfigDict(frozen=True)

    row_count: int
    column_count: int
    asset_count: int
    trajectory_length_min: int
    trajectory_length_max: int
    trajectory_length_median: float
    duplicate_pair_count: int
    missing_values_per_column: dict[str, int]
    non_finite_values_per_column: dict[str, int]
    constant_columns: tuple[str, ...]
    near_constant_columns: tuple[str, ...]
    column_ranges: dict[str, tuple[float, float]]
    """Observed (min, max) per numeric column; reported, never enforced."""


class RulStats(BaseModel):
    """Descriptive statistics for the validated official RUL frame."""

    model_config = ConfigDict(frozen=True)

    row_count: int
    column_count: int
    missing_value_count: int
    minimum: int | None
    maximum: int | None
    median: float | None


class DatasetValidation(BaseModel):
    """Validation outcome for one parsed dataset (train, test, or rul)."""

    model_config = ConfigDict(frozen=True)

    dataset: str
    checks: tuple[ValidationCheck, ...]
    warnings: tuple[str, ...]
    trajectory_stats: TrajectoryStats | None = None
    rul_stats: RulStats | None = None

    @property
    def passed(self) -> bool:
        """True when every required check passed."""
        return all(check.passed for check in self.checks if check.required)

    @property
    def failed_checks(self) -> tuple[ValidationCheck, ...]:
        """Required checks that failed."""
        return tuple(c for c in self.checks if c.required and not c.passed)


def validate_trajectory_frame(
    frame: pd.DataFrame,
    dataset: str,
    expected_rows: int | None = None,
    expected_assets: int | None = None,
) -> DatasetValidation:
    """Validate a parsed trajectory frame against the canonical contract.

    ``expected_rows`` / ``expected_assets`` come from an optional
    :class:`CanonicalProfile`; when ``None``, only general structural and
    semantic checks run.
    """
    checks: list[ValidationCheck] = []
    warnings: list[str] = []

    checks.extend(_schema_checks(frame, TRAJECTORY_DTYPES))
    schema_ok = all(check.passed for check in checks)
    stats: TrajectoryStats | None = None

    if schema_ok and len(frame) > 0:
        checks.extend(_asset_cycle_checks(frame))
        checks.extend(_numeric_checks(frame, list(TRAJECTORY_FLOAT_COLUMNS)))
        stats = _trajectory_stats(frame)
        if stats.constant_columns:
            warnings.append(
                "Constant columns (kept, not deleted): " + ", ".join(stats.constant_columns)
            )
        if stats.near_constant_columns:
            warnings.append(
                "Near-constant columns (relative std < "
                f"{NEAR_CONSTANT_RELATIVE_STD:g}): " + ", ".join(stats.near_constant_columns)
            )
        if expected_rows is not None:
            checks.append(
                _check(
                    "canonical_row_count",
                    len(frame) == expected_rows,
                    f"expected {expected_rows} rows, found {len(frame)}",
                )
            )
        if expected_assets is not None:
            checks.append(
                _check(
                    "canonical_asset_count",
                    stats.asset_count == expected_assets,
                    f"expected {expected_assets} assets, found {stats.asset_count}",
                )
            )

    result = DatasetValidation(
        dataset=dataset,
        checks=tuple(checks),
        warnings=tuple(warnings),
        trajectory_stats=stats,
    )
    logger.info(
        "trajectory_frame_validated",
        extra={
            "dataset": dataset,
            "passed": result.passed,
            "failed_checks": [check.name for check in result.failed_checks],
            "warning_count": len(warnings),
        },
    )
    return result


def validate_rul_frame(
    frame: pd.DataFrame,
    dataset: str = "rul",
    expected_values: int | None = None,
) -> DatasetValidation:
    """Validate the parsed official RUL frame (structure only, no labels)."""
    checks: list[ValidationCheck] = []
    checks.extend(_schema_checks(frame, RUL_DTYPES))
    schema_ok = all(check.passed for check in checks)
    stats: RulStats | None = None

    if schema_ok and len(frame) > 0:
        values = frame[RUL_COLUMN]
        checks.append(
            _check(
                "rul_values_non_negative",
                bool((values >= 0).all()),
                "official RUL values must be non-negative integers",
            )
        )
        if expected_values is not None:
            checks.append(
                _check(
                    "canonical_rul_count",
                    len(frame) == expected_values,
                    f"expected {expected_values} RUL values, found {len(frame)}",
                )
            )
        stats = RulStats(
            row_count=len(frame),
            column_count=frame.shape[1],
            missing_value_count=int(frame.isna().sum().sum()),
            minimum=int(values.min()),
            maximum=int(values.max()),
            median=float(values.median()),
        )

    result = DatasetValidation(
        dataset=dataset,
        checks=tuple(checks),
        warnings=(),
        rul_stats=stats,
    )
    logger.info(
        "rul_frame_validated",
        extra={"dataset": dataset, "passed": result.passed},
    )
    return result


def _check(name: str, passed: bool, failure_detail: str) -> ValidationCheck:
    """Build a required check; the failure detail is kept even when passing."""
    return ValidationCheck(
        name=name,
        passed=passed,
        required=True,
        message="ok" if passed else failure_detail,
    )


def _schema_checks(frame: pd.DataFrame, dtypes: dict[str, str]) -> list[ValidationCheck]:
    """Frame-level structural checks against a canonical schema."""
    expected = list(dtypes)
    actual = list(frame.columns)
    checks = [
        _check("not_empty", len(frame) > 0, "dataset contains no rows"),
        _check(
            "column_count",
            frame.shape[1] == len(expected),
            f"expected {len(expected)} columns, found {frame.shape[1]}",
        ),
        _check(
            "required_columns_present",
            not set(expected) - set(actual),
            f"missing columns: {sorted(set(expected) - set(actual))}",
        ),
        _check(
            "no_unexpected_columns",
            not set(actual) - set(expected),
            f"unexpected columns: {sorted(set(actual) - set(expected))}",
        ),
        _check(
            "column_order",
            actual == expected,
            "columns are not in canonical order",
        ),
    ]
    shared = [column for column in expected if column in frame.columns]
    wrong = [column for column in shared if str(frame[column].dtype) != dtypes[column]]
    checks.append(
        _check(
            "dtypes_match_schema",
            not wrong,
            "columns with wrong dtype: "
            + ", ".join(f"{c} is {frame[c].dtype}, expected {dtypes[c]}" for c in wrong),
        )
    )
    return checks


def _asset_cycle_checks(frame: pd.DataFrame) -> list[ValidationCheck]:
    """Asset and cycle integrity checks.

    Correctness must not depend on source row order: cycles are checked per
    asset after grouping. Given unique cycles, an asset's cycles are strictly
    increasing and contiguous from 1 exactly when min == 1 and
    max == count == distinct count.
    """
    asset_ids = frame[ASSET_ID_COLUMN]
    cycles = frame[CYCLE_COLUMN]

    duplicate_count = int(frame.duplicated(subset=[ASSET_ID_COLUMN, CYCLE_COLUMN]).sum())

    grouped = frame.groupby(ASSET_ID_COLUMN)[CYCLE_COLUMN]
    sizes = grouped.size()
    contiguous = (grouped.min() == 1) & (grouped.max() == sizes) & (grouped.nunique() == sizes)
    broken_assets = sorted(int(asset) for asset in sizes.index[~contiguous])

    return [
        _check(
            "asset_ids_positive",
            bool((asset_ids >= 1).all()),
            f"minimum asset_id is {int(asset_ids.min())}, expected >= 1",
        ),
        _check(
            "cycles_positive",
            bool((cycles >= 1).all()),
            f"minimum cycle is {int(cycles.min())}, expected >= 1",
        ),
        _check(
            "unique_asset_cycle_pairs",
            duplicate_count == 0,
            f"{duplicate_count} duplicate (asset_id, cycle) pairs",
        ),
        _check(
            "cycles_contiguous_from_one",
            not broken_assets,
            "assets whose cycles are not exactly 1..n: "
            + ", ".join(str(asset) for asset in broken_assets[:10]),
        ),
    ]


def _numeric_checks(frame: pd.DataFrame, float_columns: list[str]) -> list[ValidationCheck]:
    """Missing- and non-finite-value checks over the numeric payload columns."""
    missing = frame[float_columns].isna().sum()
    non_finite = _non_finite_counts(frame, float_columns)
    missing_total = int(missing.sum())
    non_finite_total = sum(non_finite.values())
    return [
        _check(
            "no_missing_values",
            missing_total == 0,
            f"{missing_total} missing values in: " + ", ".join(sorted(missing.index[missing > 0])),
        ),
        _check(
            "all_values_finite",
            non_finite_total == 0,
            f"{non_finite_total} non-finite values in: " + ", ".join(sorted(non_finite)),
        ),
    ]


def _non_finite_counts(frame: pd.DataFrame, columns: list[str]) -> dict[str, int]:
    """Per-column count of NaN or infinite values (nonzero entries only)."""
    counts: dict[str, int] = {}
    for column in columns:
        bad = int((~np.isfinite(frame[column].to_numpy(dtype="float64"))).sum())
        if bad:
            counts[column] = bad
    return counts


def _trajectory_stats(frame: pd.DataFrame) -> TrajectoryStats:
    """Descriptive statistics reported alongside the checks."""
    lengths = frame.groupby(ASSET_ID_COLUMN).size()
    float_columns = list(TRAJECTORY_FLOAT_COLUMNS)
    missing = frame[float_columns].isna().sum()

    constant: list[str] = []
    near_constant: list[str] = []
    ranges: dict[str, tuple[float, float]] = {}
    for column in float_columns:
        values = frame[column]
        values = values[np.isfinite(values)]
        if values.empty:
            continue
        ranges[column] = (float(values.min()), float(values.max()))
        std = float(values.std())
        mean = float(values.mean())
        if std == 0.0:
            constant.append(column)
        elif std / max(abs(mean), 1e-12) < NEAR_CONSTANT_RELATIVE_STD:
            near_constant.append(column)

    return TrajectoryStats(
        row_count=len(frame),
        column_count=frame.shape[1],
        asset_count=int(frame[ASSET_ID_COLUMN].nunique()),
        trajectory_length_min=int(lengths.min()),
        trajectory_length_max=int(lengths.max()),
        trajectory_length_median=float(lengths.median()),
        duplicate_pair_count=int(frame.duplicated(subset=[ASSET_ID_COLUMN, CYCLE_COLUMN]).sum()),
        missing_values_per_column={str(c): int(n) for c, n in missing.items() if n > 0},
        non_finite_values_per_column=_non_finite_counts(frame, float_columns),
        constant_columns=tuple(constant),
        near_constant_columns=tuple(near_constant),
        column_ranges=ranges,
    )

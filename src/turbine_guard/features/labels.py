"""Remaining Useful Life (RUL) label generation for training trajectories.

For a training asset ``i`` observed to fail at its final cycle ``T_i``, the RUL
at cycle ``t`` is ``T_i - t``. Because training trajectories are complete
run-to-failure records with contiguous cycles ``1..T_i`` (enforced by Loop 2
validation), ``T_i`` is simply the asset's maximum observed cycle and the
uncapped RUL decreases by exactly one each cycle, reaching zero at failure.

The official *test* RUL values are handled separately: they are the true RUL at
each truncated test trajectory's final cycle and come from the official file.
No per-row test labels are fabricated from unavailable future cycles.
"""

import logging

import pandas as pd

from turbine_guard.data.schema import ASSET_ID_COLUMN, CYCLE_COLUMN
from turbine_guard.data.schema import RUL_COLUMN as OFFICIAL_RUL_COLUMN
from turbine_guard.features.config import RUL_CAPPED_COLUMN, RUL_COLUMN, RulConfig

logger = logging.getLogger(__name__)

FINAL_CYCLE_COLUMN = "final_cycle"
"""In the test benchmark table: the last observed cycle of a test trajectory."""


class LabelError(ValueError):
    """Raised when RUL labels cannot be generated or fail validation."""


def add_rul_labels(frame: pd.DataFrame, config: RulConfig | None = None) -> pd.DataFrame:
    """Return ``frame`` with an uncapped ``rul`` column (and optional cap).

    ``T_i`` is each asset's maximum cycle; ``rul = T_i - cycle`` as ``int64``.
    When ``config.cap`` is set, an additional ``rul_capped = min(rul, cap)``
    column is produced while the uncapped ``rul`` is always preserved. The
    input row order is not relied upon; the returned frame keeps the caller's
    row order (labels are joined back per asset).
    """
    config = config or RulConfig()
    _require_columns(frame, (ASSET_ID_COLUMN, CYCLE_COLUMN))

    failure_cycle = frame.groupby(ASSET_ID_COLUMN)[CYCLE_COLUMN].transform("max")
    rul = (failure_cycle - frame[CYCLE_COLUMN]).astype("int64")

    labelled = frame.copy()
    labelled[RUL_COLUMN] = rul
    if config.produces_capped:
        assert config.cap is not None  # narrowed by produces_capped
        labelled[RUL_CAPPED_COLUMN] = rul.clip(upper=config.cap).astype("int64")

    logger.info(
        "rul_labels_generated",
        extra={
            "row_count": len(labelled),
            "asset_count": int(frame[ASSET_ID_COLUMN].nunique()),
            "capped": config.produces_capped,
            "cap": config.cap,
        },
    )
    return labelled


def validate_rul_labels(frame: pd.DataFrame, config: RulConfig | None = None) -> None:
    """Validate generated RUL labels, raising :class:`LabelError` on any defect.

    Checks that the uncapped RUL is non-negative everywhere, that every asset
    reaches RUL 0 at its final cycle, and that RUL decreases by exactly 1 as
    cycle increases by 1 within each asset. When a capped target is expected,
    checks it equals ``min(rul, cap)``.
    """
    config = config or RulConfig()
    _require_columns(frame, (ASSET_ID_COLUMN, CYCLE_COLUMN, RUL_COLUMN))

    if bool((frame[RUL_COLUMN] < 0).any()):
        raise LabelError("RUL labels contain negative values.")

    ordered = frame.sort_values([ASSET_ID_COLUMN, CYCLE_COLUMN], kind="stable")
    grouped = ordered.groupby(ASSET_ID_COLUMN)

    final_rul = grouped[RUL_COLUMN].last()
    non_zero_final = final_rul[final_rul != 0]
    if not non_zero_final.empty:
        assets = ", ".join(str(a) for a in non_zero_final.index[:10])
        raise LabelError(f"Assets whose final-cycle RUL is not 0: {assets}.")

    # Within an asset (sorted by cycle) consecutive cycles differ by +1 and RUL
    # must differ by exactly -1; the first row of each asset is NaN after diff.
    cycle_step = grouped[CYCLE_COLUMN].diff()
    rul_step = grouped[RUL_COLUMN].diff()
    interior = cycle_step.notna()
    bad = interior & ~((cycle_step == 1) & (rul_step == -1))
    if bool(bad.any()):
        offenders = ordered.loc[bad, ASSET_ID_COLUMN].unique()[:10]
        assets = ", ".join(str(a) for a in offenders)
        raise LabelError(
            "RUL must decrease by exactly 1 as cycle increases by 1; violated for "
            f"assets: {assets}. (Trajectories must have contiguous cycles.)"
        )

    if config.produces_capped:
        assert config.cap is not None
        if RUL_CAPPED_COLUMN not in frame.columns:
            raise LabelError(f"Expected a '{RUL_CAPPED_COLUMN}' column but it is missing.")
        expected = frame[RUL_COLUMN].clip(upper=config.cap)
        if not bool((frame[RUL_CAPPED_COLUMN] == expected).all()):
            raise LabelError(f"'{RUL_CAPPED_COLUMN}' does not equal min(rul, {config.cap}).")


def build_test_benchmark_labels(
    test_frame: pd.DataFrame, official_rul: pd.DataFrame
) -> pd.DataFrame:
    """Build the official test-set RUL benchmark, one row per test asset.

    The official RUL file gives one value per test unit in file order; per the
    dataset readme, file row ``i`` corresponds to test ``asset_id`` ``i + 1``.
    This positional correspondence cannot be verified from file content alone,
    so it is asserted explicitly here.

    Each output row is a test asset's *final observed cycle* and the true RUL
    at that cycle. This is an **evaluation benchmark only** — it is the RUL at
    truncation, not a per-cycle training label, and must never be used to fit
    or select models. Returned columns: ``asset_id``, ``final_cycle``, ``rul``.
    """
    _require_columns(test_frame, (ASSET_ID_COLUMN, CYCLE_COLUMN))
    _require_columns(official_rul, (OFFICIAL_RUL_COLUMN,))

    assets = sorted(int(a) for a in test_frame[ASSET_ID_COLUMN].unique())
    if len(assets) != len(official_rul):
        raise LabelError(
            f"Official RUL file has {len(official_rul)} values but the test set has "
            f"{len(assets)} assets; positional correspondence cannot hold."
        )
    if assets != list(range(1, len(assets) + 1)):
        raise LabelError(
            "Test asset IDs are not the contiguous range 1..N; the official RUL "
            "file's positional (row i -> asset i+1) correspondence is unsafe."
        )

    final_cycle = (
        test_frame.groupby(ASSET_ID_COLUMN)[CYCLE_COLUMN].max().reindex(assets).astype("int64")
    )
    benchmark = pd.DataFrame(
        {
            ASSET_ID_COLUMN: pd.Series(assets, dtype="int64"),
            FINAL_CYCLE_COLUMN: final_cycle.to_numpy(),
            RUL_COLUMN: official_rul[OFFICIAL_RUL_COLUMN].to_numpy().astype("int64"),
        }
    )
    logger.info("test_benchmark_labels_built", extra={"asset_count": len(benchmark)})
    return benchmark


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise LabelError(f"Frame is missing required columns: {missing}.")

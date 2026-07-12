"""The shared, stateless, leakage-safe feature generator.

:class:`FeatureBuilder` is the single feature-generation implementation used by
offline training-data creation now and by online inference, monitoring, and
retraining later. There is intentionally no separate "notebook" or "serving"
copy of this logic.

Leakage safety is structural:

* Every feature at cycle ``t`` is a function only of that asset's observations
  at cycles ``<= t``. Windows are *trailing* (right-aligned) and end at ``t``;
  no centered or forward-looking windows exist.
* All calculations are grouped by ``asset_id`` and never cross asset
  boundaries.
* The builder holds no fitted state — feature values for an asset depend solely
  on that asset's own trajectory, so results are independent of any other asset
  or split. This makes training-serving consistency and fit-isolation
  guarantees hold by construction rather than by discipline.

Because there is no fitted state, the offline batch path and the single-asset
incremental path (:class:`IncrementalFeatureState`) share the exact same code:
the incremental row at cycle ``t`` is the last row of the batch transform over
that asset's observations through ``t``.
"""

import logging
from collections.abc import Mapping
from typing import Any

import pandas as pd

from turbine_guard.data.schema import ASSET_ID_COLUMN, CYCLE_COLUMN
from turbine_guard.features.config import WINDOWED_FAMILIES, FeatureConfig

logger = logging.getLogger(__name__)

IDENTIFIER_COLUMNS: tuple[str, ...] = (ASSET_ID_COLUMN, CYCLE_COLUMN)

_SeriesMap = dict[str, "pd.Series[Any]"]


class FeatureError(ValueError):
    """Raised when features cannot be generated from the given input."""


class FeatureBuilder:
    """Generate trailing-window features from validated trajectory frames."""

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self._config = config or FeatureConfig()

    @property
    def config(self) -> FeatureConfig:
        """The immutable configuration this builder was constructed with."""
        return self._config

    def feature_columns(self) -> tuple[str, ...]:
        """Return the exact ordered model-feature column names.

        Deterministic and stable: iterate source columns in canonical order and,
        within each, emit families in a fixed order (current, delta, then each
        enabled windowed family across ascending windows, then EWM across
        ascending spans). This is the authoritative column contract that the
        feature manifest records and that models consume.
        """
        config = self._config
        names: list[str] = []
        for column in config.source_columns:
            if "current" in config.families:
                names.append(f"{column}_current")
            if "delta" in config.families:
                names.append(f"{column}_delta_1")
            for family in WINDOWED_FAMILIES:
                if family in config.families:
                    names.extend(f"{column}_{family}_w{window}" for window in config.windows)
            if "ewm_mean" in config.families:
                names.extend(f"{column}_ewm_mean_s{span}" for span in config.ewm_spans)
        return tuple(names)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Generate features for a multi-asset trajectory frame.

        The input is sorted by ``(asset_id, cycle)`` internally, so source row
        order is never relied upon. Returns a frame with the identifier columns
        ``asset_id`` and ``cycle`` followed by the feature columns in
        :meth:`feature_columns` order. Structurally undefined early-cycle values
        (e.g. the first-cycle delta) are left as nulls.
        """
        self._require_source_columns(frame)
        ordered = frame.sort_values(list(IDENTIFIER_COLUMNS), kind="stable").reset_index(drop=True)
        computed = self._compute(ordered)
        # Assemble the whole frame at once (not column-by-column) so the wide
        # feature block stays a single consolidated pandas block.
        data: dict[str, object] = {
            ASSET_ID_COLUMN: ordered[ASSET_ID_COLUMN].to_numpy(),
            CYCLE_COLUMN: ordered[CYCLE_COLUMN].to_numpy(),
        }
        for name in self.feature_columns():
            data[name] = computed[name].to_numpy()
        result = pd.DataFrame(data)
        logger.info(
            "features_generated",
            extra={
                "row_count": len(result),
                "asset_count": int(ordered[ASSET_ID_COLUMN].nunique()),
                "feature_count": len(self.feature_columns()),
            },
        )
        return result

    def transform_asset(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Generate features for a single asset's observations.

        Convenience wrapper over :meth:`transform` for one asset; used by the
        incremental path. Raises when the frame spans more than one asset.
        """
        if ASSET_ID_COLUMN in frame.columns and int(frame[ASSET_ID_COLUMN].nunique()) > 1:
            raise FeatureError("transform_asset expects observations for a single asset.")
        return self.transform(frame)

    def _compute(self, ordered: pd.DataFrame) -> _SeriesMap:
        """Compute every enabled feature Series, aligned to ``ordered``'s index."""
        config = self._config
        grouped = ordered.groupby(ASSET_ID_COLUMN, sort=False)
        window_stats = self._window_cycle_stats(ordered) if config.uses_windows else {}

        out: _SeriesMap = {}
        for column in config.source_columns:
            values = ordered[column].astype("float64")
            if "current" in config.families:
                out[f"{column}_current"] = values
            if "delta" in config.families:
                out[f"{column}_delta_1"] = grouped[column].diff(1).astype("float64")
            if config.uses_windows:
                self._add_windowed(out, ordered, column, window_stats)
            if config.uses_ewm:
                for span in config.ewm_spans:
                    ewm = grouped[column].ewm(span=span).mean()
                    out[f"{column}_ewm_mean_s{span}"] = self._align(ewm, ordered.index)
        return out

    def _add_windowed(
        self,
        out: _SeriesMap,
        ordered: pd.DataFrame,
        column: str,
        window_stats: dict[int, _SeriesMap],
    ) -> None:
        """Add every enabled windowed family for one source column."""
        families = self._config.families
        grouped_column = ordered.groupby(ASSET_ID_COLUMN, sort=False)[column]
        for window in self._config.windows:
            rolling = grouped_column.rolling(window, min_periods=self._config.min_periods)
            if "roll_mean" in families:
                out[f"{column}_roll_mean_w{window}"] = self._align(rolling.mean(), ordered.index)
            if "roll_std" in families:
                out[f"{column}_roll_std_w{window}"] = self._align(rolling.std(), ordered.index)
            roll_min = self._align(rolling.min(), ordered.index) if self._needs_extrema() else None
            roll_max = self._align(rolling.max(), ordered.index) if self._needs_extrema() else None
            if "roll_min" in families and roll_min is not None:
                out[f"{column}_roll_min_w{window}"] = roll_min
            if "roll_max" in families and roll_max is not None:
                out[f"{column}_roll_max_w{window}"] = roll_max
            if "roll_range" in families and roll_min is not None and roll_max is not None:
                out[f"{column}_roll_range_w{window}"] = roll_max - roll_min
            if "roll_slope" in families:
                out[f"{column}_roll_slope_w{window}"] = self._roll_slope(
                    ordered, column, window, window_stats[window]
                )

    def _needs_extrema(self) -> bool:
        families = self._config.families
        return any(family in families for family in ("roll_min", "roll_max", "roll_range"))

    def _window_cycle_stats(self, ordered: pd.DataFrame) -> dict[int, _SeriesMap]:
        """Precompute per-window trailing sums of the cycle axis, shared by slope.

        The independent variable of the rolling linear fit is the cycle number,
        which is common to every source column, so its trailing sum, sum of
        squares, and count are computed once per window and reused.
        """
        work = ordered[[ASSET_ID_COLUMN]].copy()
        work["__x"] = ordered[CYCLE_COLUMN].astype("float64")
        work["__xx"] = work["__x"] * work["__x"]
        grouped = work.groupby(ASSET_ID_COLUMN, sort=False)
        min_periods = self._config.min_periods
        stats: dict[int, _SeriesMap] = {}
        for window in self._config.windows:
            summed = grouped[["__x", "__xx"]].rolling(window, min_periods=min_periods).sum()
            count = grouped["__x"].rolling(window, min_periods=min_periods).count()
            stats[window] = {
                "sum_x": self._align(summed["__x"], ordered.index),
                "sum_xx": self._align(summed["__xx"], ordered.index),
                "count": self._align(count, ordered.index),
            }
        return stats

    def _roll_slope(
        self,
        ordered: pd.DataFrame,
        column: str,
        window: int,
        cycle_stats: _SeriesMap,
    ) -> "pd.Series[Any]":
        """Trailing ordinary-least-squares slope of ``column`` against cycle.

        Uses only observations inside the trailing window (cycle vs. value).
        A window with a single observation, or a degenerate fit, yields a
        deterministic slope of 0.0; windows below ``min_periods`` remain null.
        Source values are finite (guaranteed by Loop 2 validation), so the
        observation count equals the cycle count.
        """
        x = ordered[CYCLE_COLUMN].astype("float64")
        y = ordered[column].astype("float64")
        work = ordered[[ASSET_ID_COLUMN]].copy()
        work["__y"] = y
        work["__xy"] = x * y
        grouped = work.groupby(ASSET_ID_COLUMN, sort=False)
        rolling = grouped[["__y", "__xy"]].rolling(window, min_periods=self._config.min_periods)
        summed = rolling.sum()
        sum_y = self._align(summed["__y"], ordered.index)
        sum_xy = self._align(summed["__xy"], ordered.index)

        count = cycle_stats["count"]
        sum_x = cycle_stats["sum_x"]
        sum_xx = cycle_stats["sum_xx"]
        numerator = count * sum_xy - sum_x * sum_y
        denominator = count * sum_xx - sum_x * sum_x
        slope = numerator / denominator.where(denominator != 0)
        degenerate = (denominator == 0) & count.notna()
        return slope.mask(degenerate, 0.0).astype("float64")

    @staticmethod
    def _align(series: "pd.Series[Any]", index: pd.Index) -> "pd.Series[Any]":
        """Drop the group key a grouped-rolling result carries and realign."""
        return series.reset_index(level=0, drop=True).reindex(index)

    def _require_source_columns(self, frame: pd.DataFrame) -> None:
        required = (*IDENTIFIER_COLUMNS, *self._config.source_columns)
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise FeatureError(f"Input frame is missing required columns: {missing}.")


class IncrementalFeatureState:
    """Single-asset incremental feature generation for online inference.

    Observations are fed one cycle at a time in increasing cycle order; each
    :meth:`update` returns the current-cycle feature row, identical to the
    corresponding offline batch row. The state keeps the asset's observation
    history and recomputes through the shared :class:`FeatureBuilder`, so the
    online and offline feature definitions can never drift apart. It never
    requires future observations.

    Trajectories in this dataset are short, so retaining full history is cheap
    and keeps the exponentially-weighted features exactly equal to the batch
    values. A production system with very long histories could instead keep a
    bounded window plus a running EWM accumulator; that optimization is
    deliberately out of scope here.
    """

    def __init__(self, builder: FeatureBuilder, asset_id: int) -> None:
        self._builder = builder
        self._asset_id = int(asset_id)
        self._history: list[dict[str, float]] = []
        self._last_cycle: int | None = None

    @classmethod
    def from_history(
        cls, builder: FeatureBuilder, asset_id: int, observations: pd.DataFrame
    ) -> "IncrementalFeatureState":
        """Reconstruct state from a past trajectory (e.g. after a restart)."""
        state = cls(builder, asset_id)
        ordered = observations.sort_values(CYCLE_COLUMN, kind="stable")
        for record in ordered.to_dict(orient="records"):
            state._append({str(key): value for key, value in record.items()})
        return state

    def update(self, observation: Mapping[str, float]) -> "pd.Series[Any]":
        """Ingest one observation and return the current-cycle feature row."""
        self._append(dict(observation))
        frame = pd.DataFrame(self._history)
        frame[ASSET_ID_COLUMN] = self._asset_id
        return self._builder.transform_asset(frame).iloc[-1]

    @property
    def cycles_seen(self) -> int:
        """Number of observations ingested so far."""
        return len(self._history)

    def _append(self, observation: Mapping[str, float]) -> None:
        cycle = int(observation[CYCLE_COLUMN])
        if self._last_cycle is not None and cycle <= self._last_cycle:
            raise FeatureError(
                f"Observations must arrive in strictly increasing cycle order; got cycle "
                f"{cycle} after {self._last_cycle}."
            )
        record = {key: value for key, value in observation.items() if key != ASSET_ID_COLUMN}
        self._history.append(record)
        self._last_cycle = cycle

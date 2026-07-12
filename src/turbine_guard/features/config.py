"""Typed configuration for Loop 3 labels, splits, and features.

Configuration lives in frozen dataclasses (the same convention as
:class:`turbine_guard.data.acquisition.AcquisitionConfig` and
:class:`turbine_guard.data.processing.ProcessingConfig`) rather than scattered
module constants. There is no YAML configuration layer in the repository yet;
introducing one would be a new pattern, so the established typed-dataclass
convention is used and documented in ADR 0002.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

from turbine_guard.data.schema import OPERATING_SETTING_COLUMNS, SENSOR_COLUMNS

# Canonical target and metadata column names produced by Loop 3.
RUL_COLUMN = "rul"
"""Uncapped Remaining Useful Life target (cycles remaining until failure)."""

RUL_CAPPED_COLUMN = "rul_capped"
"""Optional piecewise-linear capped RUL target; present only when enabled."""

SPLIT_COLUMN = "split"
"""Metadata column naming the partition a model-ready row belongs to."""

FEATURE_VERSION = "1"
"""Version of the feature *definition*; bump when feature semantics change."""

SPLIT_VERSION = "1"
"""Version of the split *strategy*; bump when the partitioning changes."""

# Source columns features are derived from: the three operating settings and
# the twenty-one sensors, in canonical order. Identifiers (asset_id, cycle)
# and targets are never treated as feature sources.
SOURCE_COLUMNS: tuple[str, ...] = (*OPERATING_SETTING_COLUMNS, *SENSOR_COLUMNS)

# Feature-family identifiers. Windowed families consume ``windows``; EWM
# families consume ``ewm_spans``; ``current`` and ``delta`` take no window.
WINDOWED_FAMILIES: tuple[str, ...] = (
    "roll_mean",
    "roll_std",
    "roll_min",
    "roll_max",
    "roll_range",
    "roll_slope",
)
EWM_FAMILIES: tuple[str, ...] = ("ewm_mean",)
POINT_FAMILIES: tuple[str, ...] = ("current", "delta")
ALL_FAMILIES: tuple[str, ...] = (*POINT_FAMILIES, *WINDOWED_FAMILIES, *EWM_FAMILIES)


@dataclass(frozen=True)
class FeatureConfig:
    """Controls which leakage-safe features the :class:`FeatureBuilder` emits.

    All windows are *trailing* and end at the current cycle. ``min_periods``
    governs how many observations a window needs before a value is produced;
    with the default of 1, rolling mean/min/max are defined from the first
    cycle, while families that need at least two points (delta, rolling std,
    rolling slope) are structurally undefined (null / zero) on the first cycle.
    """

    feature_version: str = FEATURE_VERSION
    source_columns: tuple[str, ...] = SOURCE_COLUMNS
    families: tuple[str, ...] = ALL_FAMILIES
    windows: tuple[int, ...] = (5, 10, 20)
    ewm_spans: tuple[int, ...] = (5, 10, 20)
    min_periods: int = 1

    def __post_init__(self) -> None:
        unknown = tuple(f for f in self.families if f not in ALL_FAMILIES)
        if unknown:
            raise ValueError(
                f"Unknown feature families {unknown}; valid families are {ALL_FAMILIES}."
            )
        if not self.source_columns:
            raise ValueError("At least one source column is required.")
        if any(window < 2 for window in self.windows):
            raise ValueError("Rolling windows must be >= 2 cycles.")
        if any(span < 1 for span in self.ewm_spans):
            raise ValueError("EWM spans must be >= 1.")
        if self.min_periods < 1:
            raise ValueError("min_periods must be >= 1.")

    @property
    def uses_windows(self) -> bool:
        """Whether any enabled family consumes ``windows``."""
        return any(family in WINDOWED_FAMILIES for family in self.families)

    @property
    def uses_ewm(self) -> bool:
        """Whether any enabled family consumes ``ewm_spans``."""
        return any(family in EWM_FAMILIES for family in self.families)


@dataclass(frozen=True)
class SplitConfig:
    """Controls the deterministic asset-level partitioning of training assets.

    Fractions are converted to integer asset counts with largest-remainder
    rounding so the counts always sum to the number of training assets. The
    partition order for tie-breaking is fixed: train, validation, calibration,
    replay.
    """

    seed: int = 42
    split_version: str = SPLIT_VERSION
    strategy: str = "seeded_permutation"
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    calibration_fraction: float = 0.05
    replay_fraction: float = 0.10

    def __post_init__(self) -> None:
        total = (
            self.train_fraction
            + self.validation_fraction
            + self.calibration_fraction
            + self.replay_fraction
        )
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Split fractions must sum to 1.0, got {total}.")
        if min(self.fractions.values()) < 0:
            raise ValueError("Split fractions must be non-negative.")

    @property
    def fractions(self) -> Mapping[str, float]:
        """Ordered mapping of partition name to fraction (fixed order)."""
        return {
            "train": self.train_fraction,
            "validation": self.validation_fraction,
            "calibration": self.calibration_fraction,
            "replay": self.replay_fraction,
        }


@dataclass(frozen=True)
class RulConfig:
    """Controls RUL label generation.

    ``cap`` is optional; when set, a ``rul_capped`` column is produced in
    addition to the always-present uncapped ``rul`` column. The cap models the
    common assumption that early-life RUL is not meaningfully predictable and
    is clipped to a constant healthy value.
    """

    cap: int | None = None

    def __post_init__(self) -> None:
        if self.cap is not None and self.cap < 0:
            raise ValueError("RUL cap must be non-negative.")

    @property
    def produces_capped(self) -> bool:
        """Whether a ``rul_capped`` column will be generated."""
        return self.cap is not None


@dataclass(frozen=True)
class BuildConfig:
    """Aggregate configuration for one feature-build run."""

    features: FeatureConfig = field(default_factory=FeatureConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    rul: RulConfig = field(default_factory=RulConfig)

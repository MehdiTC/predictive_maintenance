"""Deterministic asset-level splitting of training assets.

Splits are always by ``asset_id`` — never by individual row — so that an
asset's entire trajectory lands in exactly one partition. This prevents the
classic time-series leak where rows of the same trajectory appear in both
train and validation.

Determinism: the unique training asset IDs are sorted ascending and permuted
with ``numpy.random.default_rng(seed)`` (a reproducible PCG64 stream), then
sliced into contiguous blocks whose sizes come from the configured fractions
via largest-remainder rounding. The same seed and asset set always yield the
same partitions.

The four partitions have distinct roles that later loops depend on:

* ``train`` — fits models and any preprocessing.
* ``validation`` — model selection and tuning.
* ``calibration`` — held out for conformal prediction intervals (Loop 4);
  kept separate from train and validation.
* ``replay`` — held out entirely from initial training so it can be streamed
  as "live" sensor data in Loop 8; never used for fitting, feature selection,
  calibration, or threshold tuning.

The official NASA *test* set is not touched here; it remains a separate,
untouched benchmark.
"""

import logging
from collections.abc import Mapping

import numpy as np
import pandas as pd

from turbine_guard.data.schema import ASSET_ID_COLUMN
from turbine_guard.features.config import SplitConfig

logger = logging.getLogger(__name__)

PARTITION_NAMES: tuple[str, ...] = ("train", "validation", "calibration", "replay")


class SplitError(ValueError):
    """Raised when an asset-level split cannot be produced or is inconsistent."""


class AssetSplit:
    """An immutable assignment of asset IDs to named partitions.

    Construction validates that partitions are disjoint and cover exactly the
    input asset set, so a constructed :class:`AssetSplit` is always internally
    consistent.
    """

    def __init__(self, partitions: Mapping[str, tuple[int, ...]]) -> None:
        self._partitions: dict[str, tuple[int, ...]] = {
            name: tuple(int(asset) for asset in partitions.get(name, ()))
            for name in PARTITION_NAMES
        }
        self._validate()

    def _validate(self) -> None:
        seen: set[int] = set()
        for name, assets in self._partitions.items():
            as_set = set(assets)
            if len(as_set) != len(assets):
                raise SplitError(f"Partition '{name}' contains duplicate asset IDs.")
            overlap = seen & as_set
            if overlap:
                raise SplitError(f"Asset(s) {sorted(overlap)} appear in more than one partition.")
            seen |= as_set

    @property
    def partitions(self) -> Mapping[str, tuple[int, ...]]:
        """Mapping of partition name to sorted asset-ID tuple."""
        return dict(self._partitions)

    @property
    def all_assets(self) -> tuple[int, ...]:
        """Every assigned asset ID, sorted ascending."""
        return tuple(sorted(asset for assets in self._partitions.values() for asset in assets))

    def assets(self, partition: str) -> tuple[int, ...]:
        """Sorted asset IDs assigned to ``partition``."""
        if partition not in self._partitions:
            raise SplitError(f"Unknown partition '{partition}'.")
        return self._partitions[partition]

    def counts(self) -> dict[str, int]:
        """Asset count per partition."""
        return {name: len(assets) for name, assets in self._partitions.items()}


def assign_counts(n_assets: int, config: SplitConfig) -> dict[str, int]:
    """Convert fractions into integer per-partition counts that sum to ``n``.

    Uses largest-remainder rounding: each partition gets ``floor(fraction * n)``
    assets, then the leftover assets are handed out one at a time to the
    partitions with the largest fractional remainders, breaking ties by the
    fixed partition order (train, validation, calibration, replay).
    """
    if n_assets < 0:
        raise SplitError("Asset count must be non-negative.")
    exact = {name: fraction * n_assets for name, fraction in config.fractions.items()}
    counts = {name: int(np.floor(value)) for name, value in exact.items()}
    remainder = n_assets - sum(counts.values())
    if remainder:
        order = sorted(
            PARTITION_NAMES,
            key=lambda name: (-(exact[name] - counts[name]), PARTITION_NAMES.index(name)),
        )
        for name in order[:remainder]:
            counts[name] += 1
    return counts


def make_asset_split(assets: pd.Series | pd.Index | np.ndarray, config: SplitConfig) -> AssetSplit:
    """Deterministically partition the unique asset IDs in ``assets``.

    ``assets`` may contain repeated IDs (e.g. the ``asset_id`` column of a
    trajectory frame); only the distinct IDs are partitioned, guaranteeing an
    asset-level (never row-level) split.
    """
    unique = np.array(sorted({int(a) for a in np.asarray(assets)}), dtype="int64")
    if unique.size == 0:
        raise SplitError("No assets to split.")

    counts = assign_counts(int(unique.size), config)
    rng = np.random.default_rng(config.seed)
    order = rng.permutation(unique)

    partitions: dict[str, tuple[int, ...]] = {}
    start = 0
    for name in PARTITION_NAMES:
        end = start + counts[name]
        partitions[name] = tuple(sorted(int(a) for a in order[start:end]))
        start = end

    split = AssetSplit(partitions)
    logger.info(
        "asset_split_created",
        extra={
            "seed": config.seed,
            "strategy": config.strategy,
            "counts": split.counts(),
        },
    )
    return split


def split_of(frame: pd.DataFrame, split: AssetSplit, partition: str) -> pd.DataFrame:
    """Return the subset of ``frame`` whose assets belong to ``partition``."""
    wanted = set(split.assets(partition))
    return frame[frame[ASSET_ID_COLUMN].isin(wanted)].reset_index(drop=True)

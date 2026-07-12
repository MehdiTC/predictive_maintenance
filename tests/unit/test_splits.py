"""Tests for deterministic asset-level splitting."""

import numpy as np
import pytest
from tests.conftest import make_trajectory_frame

from turbine_guard.features.config import SplitConfig
from turbine_guard.features.splits import (
    PARTITION_NAMES,
    AssetSplit,
    SplitError,
    assign_counts,
    make_asset_split,
    split_of,
)

ASSETS = np.arange(1, 21)  # 20 assets -> 14/3/1/2 under default fractions


def test_counts_sum_to_asset_count_and_are_expected() -> None:
    counts = assign_counts(20, SplitConfig())
    assert counts == {"train": 14, "validation": 3, "calibration": 1, "replay": 2}
    assert sum(counts.values()) == 20


def test_counts_largest_remainder_still_sums() -> None:
    # 13 assets do not divide evenly; counts must still total 13.
    counts = assign_counts(13, SplitConfig())
    assert sum(counts.values()) == 13


def test_split_is_deterministic_for_fixed_seed() -> None:
    first = make_asset_split(ASSETS, SplitConfig(seed=7))
    second = make_asset_split(ASSETS, SplitConfig(seed=7))
    assert first.partitions == second.partitions


def test_different_seed_changes_partitions() -> None:
    a = make_asset_split(ASSETS, SplitConfig(seed=1))
    b = make_asset_split(ASSETS, SplitConfig(seed=2))
    assert a.partitions != b.partitions


def test_no_asset_in_more_than_one_partition() -> None:
    split = make_asset_split(ASSETS, SplitConfig())
    seen: list[int] = []
    for name in PARTITION_NAMES:
        seen.extend(split.assets(name))
    assert len(seen) == len(set(seen))


def test_all_assets_covered_exactly_once() -> None:
    split = make_asset_split(ASSETS, SplitConfig())
    assert split.all_assets == tuple(range(1, 21))


def test_split_is_asset_level_not_row_level() -> None:
    # Every row of an asset must land in the same partition.
    frame = make_trajectory_frame(dict.fromkeys(range(1, 21), 5))
    split = make_asset_split(frame["asset_id"], SplitConfig())
    for name in PARTITION_NAMES:
        partition_frame = split_of(frame, split, name)
        assets_in_frame = set(partition_frame["asset_id"].unique())
        assert assets_in_frame == set(split.assets(name))


def test_repeated_asset_ids_are_deduplicated() -> None:
    frame = make_trajectory_frame(dict.fromkeys(range(1, 21), 4))
    from_column = make_asset_split(frame["asset_id"], SplitConfig(seed=3))
    from_unique = make_asset_split(np.arange(1, 21), SplitConfig(seed=3))
    assert from_column.partitions == from_unique.partitions


def test_replay_and_calibration_are_isolated_from_train_and_validation() -> None:
    split = make_asset_split(ASSETS, SplitConfig())
    train_val = set(split.assets("train")) | set(split.assets("validation"))
    assert not (set(split.assets("replay")) & train_val)
    assert not (set(split.assets("calibration")) & train_val)
    assert not (set(split.assets("replay")) & set(split.assets("calibration")))


def test_asset_split_rejects_overlap() -> None:
    with pytest.raises(SplitError, match="more than one partition"):
        AssetSplit({"train": (1, 2), "validation": (2, 3)})


def test_make_asset_split_rejects_empty() -> None:
    with pytest.raises(SplitError, match="No assets"):
        make_asset_split(np.array([], dtype="int64"), SplitConfig())


def test_config_rejects_fractions_not_summing_to_one() -> None:
    with pytest.raises(ValueError, match=r"sum to 1\.0"):
        SplitConfig(train_fraction=0.5, validation_fraction=0.1)

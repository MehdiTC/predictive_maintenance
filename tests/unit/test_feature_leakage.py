"""Explicit leakage-protection tests for the feature pipeline.

These prove the core Loop 3 guarantee: a feature at cycle ``t`` can only depend
on observations at cycles ``<= t`` for the same asset, and no fitted state
crosses split boundaries.
"""

import numpy as np
import pandas as pd
from tests.conftest import make_trajectory_frame

from turbine_guard.features.builder import FeatureBuilder
from turbine_guard.features.config import FeatureConfig, SplitConfig
from turbine_guard.features.splits import make_asset_split, split_of


def _single_asset(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"asset_id": 1, "cycle": range(1, len(values) + 1), "s": values})


def builder() -> FeatureBuilder:
    return FeatureBuilder(FeatureConfig(source_columns=("s",), windows=(3,), ewm_spans=(3,)))


def test_future_row_mutation_does_not_change_earlier_features() -> None:
    frame = _single_asset([1.0, 2.0, 3.0, 4.0, 5.0])
    cut = 3  # inspect features at cycle 3

    original = builder().transform(frame)
    at_t = original[original["cycle"] == cut].reset_index(drop=True)

    mutated = frame.copy()
    mutated.loc[mutated["cycle"] > cut, "s"] = [999.0, -999.0]
    remutated = builder().transform(mutated)
    at_t_after = remutated[remutated["cycle"] == cut].reset_index(drop=True)

    pd.testing.assert_frame_equal(at_t, at_t_after)


def test_future_row_append_does_not_change_earlier_features() -> None:
    through_t = _single_asset([1.0, 2.0, 3.0])
    extended = _single_asset([1.0, 2.0, 3.0, 42.0, 43.0])

    features_through_t = builder().transform(through_t)
    features_extended = builder().transform(extended)
    overlap = features_extended[features_extended["cycle"] <= 3].reset_index(drop=True)

    pd.testing.assert_frame_equal(features_through_t, overlap)


def test_cross_asset_isolation() -> None:
    base = make_trajectory_frame({1: 5, 2: 5})
    mutated = base.copy()
    sensor = "sensor_02"
    mutated.loc[mutated["asset_id"] == 2, sensor] = 12345.0

    full = FeatureBuilder(FeatureConfig())
    a1_base = full.transform(base).query("asset_id == 1").reset_index(drop=True)
    a1_mut = full.transform(mutated).query("asset_id == 1").reset_index(drop=True)

    pd.testing.assert_frame_equal(a1_base, a1_mut)


def test_builder_is_stateless_no_fitted_state() -> None:
    # Features for a subset are identical whether or not other assets are present
    # in the input frame; the builder fits nothing on the full population.
    frame = make_trajectory_frame({1: 6, 2: 6, 3: 6})
    full = FeatureBuilder(FeatureConfig())

    from_all = full.transform(frame).query("asset_id == 2").reset_index(drop=True)
    from_one = full.transform(frame[frame["asset_id"] == 2]).reset_index(drop=True)

    pd.testing.assert_frame_equal(from_all, from_one)


def test_replay_assets_excluded_from_training_partition() -> None:
    frame = make_trajectory_frame(dict.fromkeys(range(1, 21), 5))
    split = make_asset_split(frame["asset_id"], SplitConfig())

    train_frame = split_of(frame, split, "train")
    replay_assets = set(split.assets("replay"))

    assert not (set(train_frame["asset_id"].unique()) & replay_assets)
    # Replay assets are also absent from validation and calibration.
    for partition in ("validation", "calibration"):
        partition_assets = set(split_of(frame, split, partition)["asset_id"].unique())
        assert not (partition_assets & replay_assets)


def test_no_feature_uses_future_via_exhaustive_perturbation() -> None:
    # For every cycle t, perturbing exactly one strictly-future value must leave
    # the entire feature row at t bit-for-bit identical.
    values = [1.0, 3.0, 2.0, 5.0, 4.0, 6.0]
    baseline = builder().transform(_single_asset(values))
    feature_cols = list(builder().feature_columns())

    for t in range(1, len(values)):
        perturbed = list(values)
        perturbed[t] = perturbed[t] + 1000.0  # change a future cycle only
        after = builder().transform(_single_asset(perturbed))
        before_rows = baseline[baseline["cycle"] <= t][feature_cols].to_numpy()
        after_rows = after[after["cycle"] <= t][feature_cols].to_numpy()
        assert np.array_equal(before_rows, after_rows, equal_nan=True), f"leak at t={t}"

"""Tests for the shared FeatureBuilder feature definitions."""

import numpy as np
import pandas as pd
import pytest

from turbine_guard.features.builder import FeatureBuilder, FeatureError
from turbine_guard.features.config import FeatureConfig


def single_column_frame(asset_values: dict[int, list[float]]) -> pd.DataFrame:
    """Build a minimal (asset_id, cycle, s) frame from per-asset value lists."""
    rows = [
        {"asset_id": asset, "cycle": index + 1, "s": value}
        for asset, values in asset_values.items()
        for index, value in enumerate(values)
    ]
    return pd.DataFrame(rows)


def builder_for(**overrides: object) -> FeatureBuilder:
    config = FeatureConfig(source_columns=("s",), windows=(3,), ewm_spans=(3,), **overrides)
    return FeatureBuilder(config)


def test_feature_columns_are_canonical_and_ordered() -> None:
    builder = builder_for()
    assert builder.feature_columns() == (
        "s_current",
        "s_delta_1",
        "s_roll_mean_w3",
        "s_roll_std_w3",
        "s_roll_min_w3",
        "s_roll_max_w3",
        "s_roll_range_w3",
        "s_roll_slope_w3",
        "s_ewm_mean_s3",
    )


def test_output_has_identifiers_then_features_in_order() -> None:
    builder = builder_for()
    out = builder.transform(single_column_frame({1: [1.0, 2.0, 3.0]}))
    assert list(out.columns) == ["asset_id", "cycle", *builder.feature_columns()]


def test_current_value_feature() -> None:
    out = builder_for().transform(single_column_frame({1: [5.0, 6.0, 7.0]}))
    assert out["s_current"].tolist() == [5.0, 6.0, 7.0]


def test_lag_difference_first_cycle_null() -> None:
    out = builder_for().transform(single_column_frame({1: [1.0, 4.0, 9.0]}))
    assert pd.isna(out.loc[0, "s_delta_1"])
    assert out.loc[1, "s_delta_1"] == 3.0
    assert out.loc[2, "s_delta_1"] == 5.0


def test_rolling_mean_std_min_max_range() -> None:
    out = builder_for().transform(single_column_frame({1: [1.0, 2.0, 4.0, 7.0]}))
    # Window ends at cycle 4 over (2, 4, 7).
    assert out.loc[3, "s_roll_mean_w3"] == pytest.approx(np.mean([2, 4, 7]))
    assert out.loc[3, "s_roll_std_w3"] == pytest.approx(np.std([2, 4, 7], ddof=1))
    assert out.loc[3, "s_roll_min_w3"] == 2.0
    assert out.loc[3, "s_roll_max_w3"] == 7.0
    assert out.loc[3, "s_roll_range_w3"] == 5.0


def test_rolling_std_first_cycle_null() -> None:
    out = builder_for().transform(single_column_frame({1: [3.0, 3.0]}))
    assert pd.isna(out.loc[0, "s_roll_std_w3"])  # one observation -> undefined


def test_rolling_slope_matches_ols() -> None:
    out = builder_for().transform(single_column_frame({1: [1.0, 2.0, 4.0, 7.0]}))
    expected = np.polyfit([2, 3, 4], [2, 4, 7], 1)[0]
    assert out.loc[3, "s_roll_slope_w3"] == pytest.approx(expected)


def test_rolling_slope_single_point_is_zero() -> None:
    out = builder_for().transform(single_column_frame({1: [5.0]}))
    assert out.loc[0, "s_roll_slope_w3"] == 0.0


def test_rolling_slope_constant_sequence_is_zero() -> None:
    out = builder_for().transform(single_column_frame({1: [4.0, 4.0, 4.0]}))
    assert (out["s_roll_slope_w3"] == 0.0).all()


def test_ewm_mean_matches_pandas() -> None:
    values = [1.0, 2.0, 4.0, 7.0]
    out = builder_for().transform(single_column_frame({1: values}))
    expected = pd.Series(values).ewm(span=3).mean()
    assert out["s_ewm_mean_s3"].to_numpy() == pytest.approx(expected.to_numpy())


def test_unsorted_input_yields_sorted_grouped_features() -> None:
    frame = single_column_frame({1: [1.0, 2.0, 4.0], 2: [10.0, 20.0]})
    shuffled = frame.sample(frac=1.0, random_state=0).reset_index(drop=True)

    out = builder_for().transform(shuffled)

    assert out["asset_id"].tolist() == [1, 1, 1, 2, 2]
    assert out["cycle"].tolist() == [1, 2, 3, 1, 2]
    # Delta is computed within each asset even though rows arrived interleaved.
    asset2 = out[out["asset_id"] == 2].reset_index(drop=True)
    assert pd.isna(asset2.loc[0, "s_delta_1"])
    assert asset2.loc[1, "s_delta_1"] == 10.0


def test_windows_do_not_cross_asset_boundaries() -> None:
    # Asset 2's first cycle must not see asset 1's values.
    out = builder_for().transform(single_column_frame({1: [100.0, 100.0, 100.0], 2: [1.0]}))
    asset2 = out[out["asset_id"] == 2].reset_index(drop=True)
    assert asset2.loc[0, "s_roll_mean_w3"] == 1.0
    assert asset2.loc[0, "s_roll_max_w3"] == 1.0


def test_changing_one_asset_does_not_change_another() -> None:
    base = single_column_frame({1: [1.0, 2.0, 3.0], 2: [5.0, 6.0, 7.0]})
    mutated = base.copy()
    mutated.loc[mutated["asset_id"] == 1, "s"] = [99.0, 98.0, 97.0]

    builder = builder_for()
    out_base = builder.transform(base)
    out_mut = builder.transform(mutated)

    a2_base = out_base[out_base["asset_id"] == 2].reset_index(drop=True)
    a2_mut = out_mut[out_mut["asset_id"] == 2].reset_index(drop=True)
    pd.testing.assert_frame_equal(a2_base, a2_mut)


def test_min_periods_configuration_changes_early_cycles() -> None:
    values = {1: [1.0, 2.0, 3.0]}
    default = builder_for().transform(single_column_frame(values))
    strict = builder_for(min_periods=2).transform(single_column_frame(values))

    # With min_periods=1 the first-cycle rolling mean is defined; with 2 it is null.
    assert default.loc[0, "s_roll_mean_w3"] == 1.0
    assert pd.isna(strict.loc[0, "s_roll_mean_w3"])


def test_enabling_fewer_families_shrinks_output() -> None:
    builder = FeatureBuilder(FeatureConfig(source_columns=("s",), families=("current", "delta")))
    out = builder.transform(single_column_frame({1: [1.0, 2.0]}))
    assert builder.feature_columns() == ("s_current", "s_delta_1")
    assert list(out.columns) == ["asset_id", "cycle", "s_current", "s_delta_1"]


def test_missing_source_column_raises() -> None:
    builder = FeatureBuilder(FeatureConfig(source_columns=("missing",)))
    with pytest.raises(FeatureError, match="missing required columns"):
        builder.transform(single_column_frame({1: [1.0]}))


def test_default_config_produces_expected_feature_count() -> None:
    builder = FeatureBuilder()  # 24 source columns, all families, 3 windows, 3 spans
    # per column: current + delta + 6 windowed families x 3 windows + ewm x 3 = 23
    assert len(builder.feature_columns()) == 552  # 24 columns x 23 features

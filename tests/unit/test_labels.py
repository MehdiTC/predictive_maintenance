"""Tests for RUL label generation and the official test benchmark."""

import pandas as pd
import pytest
from tests.conftest import make_trajectory_frame

from turbine_guard.features.config import RUL_CAPPED_COLUMN, RUL_COLUMN, RulConfig
from turbine_guard.features.labels import (
    FINAL_CYCLE_COLUMN,
    LabelError,
    add_rul_labels,
    build_test_benchmark_labels,
    validate_rul_labels,
)


def test_uncapped_rul_values_are_time_to_failure() -> None:
    frame = make_trajectory_frame({1: 4, 2: 3})

    labelled = add_rul_labels(frame)

    asset1 = labelled[labelled["asset_id"] == 1].sort_values("cycle")
    assert asset1[RUL_COLUMN].tolist() == [3, 2, 1, 0]
    asset2 = labelled[labelled["asset_id"] == 2].sort_values("cycle")
    assert asset2[RUL_COLUMN].tolist() == [2, 1, 0]
    assert str(labelled[RUL_COLUMN].dtype) == "int64"


def test_final_cycle_has_zero_rul() -> None:
    frame = make_trajectory_frame({1: 5, 2: 8})

    labelled = add_rul_labels(frame)

    finals = labelled.sort_values("cycle").groupby("asset_id")[RUL_COLUMN].last()
    assert (finals == 0).all()


def test_rul_decreases_by_one_per_cycle() -> None:
    frame = make_trajectory_frame({1: 6})

    labelled = add_rul_labels(frame).sort_values("cycle")

    assert labelled[RUL_COLUMN].diff().dropna().eq(-1).all()


def test_capped_rul_preserves_uncapped_and_clips() -> None:
    frame = make_trajectory_frame({1: 10})

    labelled = add_rul_labels(frame, RulConfig(cap=3))

    assert RUL_COLUMN in labelled.columns
    assert labelled[RUL_COLUMN].max() == 9  # uncapped preserved
    assert labelled[RUL_CAPPED_COLUMN].max() == 3
    assert (labelled[RUL_CAPPED_COLUMN] == labelled[RUL_COLUMN].clip(upper=3)).all()
    assert str(labelled[RUL_CAPPED_COLUMN].dtype) == "int64"


def test_no_capped_column_when_cap_disabled() -> None:
    labelled = add_rul_labels(make_trajectory_frame({1: 3}))
    assert RUL_CAPPED_COLUMN not in labelled.columns


def test_labels_are_row_order_independent() -> None:
    frame = make_trajectory_frame({1: 4, 2: 3}).sample(frac=1.0, random_state=1)

    labelled = add_rul_labels(frame)

    # The RUL for a given (asset, cycle) does not depend on input row order.
    lookup = labelled.set_index(["asset_id", "cycle"])[RUL_COLUMN]
    assert lookup[(1, 1)] == 3
    assert lookup[(2, 3)] == 0


def test_validate_accepts_valid_labels() -> None:
    labelled = add_rul_labels(make_trajectory_frame({1: 4, 2: 6}))
    validate_rul_labels(labelled)  # must not raise


def test_validate_rejects_non_contiguous_trajectory() -> None:
    frame = make_trajectory_frame({1: 4})
    frame = frame[frame["cycle"] != 2]  # drop a cycle -> RUL no longer steps by 1

    labelled = add_rul_labels(frame)

    with pytest.raises(LabelError, match="decrease by exactly 1"):
        validate_rul_labels(labelled)


def test_validate_rejects_negative_rul() -> None:
    labelled = add_rul_labels(make_trajectory_frame({1: 3}))
    labelled.loc[0, RUL_COLUMN] = -1

    with pytest.raises(LabelError, match="negative"):
        validate_rul_labels(labelled)


def test_validate_capped_mismatch_detected() -> None:
    labelled = add_rul_labels(make_trajectory_frame({1: 5}), RulConfig(cap=2))
    labelled.loc[labelled.index[0], RUL_CAPPED_COLUMN] = 99

    with pytest.raises(LabelError, match="rul_capped"):
        validate_rul_labels(labelled, RulConfig(cap=2))


def test_build_test_benchmark_labels() -> None:
    test_frame = make_trajectory_frame({1: 3, 2: 5})
    official = pd.DataFrame({"rul": [12, 7]})

    benchmark = build_test_benchmark_labels(test_frame, official)

    assert list(benchmark.columns) == ["asset_id", FINAL_CYCLE_COLUMN, RUL_COLUMN]
    assert benchmark["asset_id"].tolist() == [1, 2]
    assert benchmark[FINAL_CYCLE_COLUMN].tolist() == [3, 5]  # each asset's last cycle
    assert benchmark[RUL_COLUMN].tolist() == [12, 7]  # official values, positional


def test_build_test_benchmark_rejects_count_mismatch() -> None:
    test_frame = make_trajectory_frame({1: 3, 2: 5})
    official = pd.DataFrame({"rul": [12, 7, 9]})

    with pytest.raises(LabelError, match="positional correspondence"):
        build_test_benchmark_labels(test_frame, official)


def test_build_test_benchmark_rejects_non_contiguous_assets() -> None:
    test_frame = make_trajectory_frame({1: 3, 5: 5})  # ids not 1..N
    official = pd.DataFrame({"rul": [12, 7]})

    with pytest.raises(LabelError, match="contiguous range"):
        build_test_benchmark_labels(test_frame, official)

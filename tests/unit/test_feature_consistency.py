"""Offline-versus-incremental (training-serving) consistency tests."""

import numpy as np
import pandas as pd
import pytest
from tests.conftest import make_trajectory_frame

from turbine_guard.features.builder import (
    FeatureBuilder,
    FeatureError,
    IncrementalFeatureState,
)
from turbine_guard.features.config import FeatureConfig


def builder() -> FeatureBuilder:
    return FeatureBuilder(FeatureConfig())


def _asset_records(frame: pd.DataFrame, asset_id: int) -> list[dict[str, float]]:
    asset = frame[frame["asset_id"] == asset_id].sort_values("cycle")
    return [
        {key: value for key, value in record.items() if key != "asset_id"}
        for record in asset.to_dict(orient="records")
    ]


def test_incremental_matches_offline_at_every_cycle() -> None:
    frame = make_trajectory_frame({1: 12})
    offline = builder().transform(frame).sort_values("cycle").reset_index(drop=True)

    state = IncrementalFeatureState(builder(), asset_id=1)
    for index, observation in enumerate(_asset_records(frame, 1)):
        row = state.update(observation)
        expected = offline.loc[index]
        np.testing.assert_allclose(
            row[list(builder().feature_columns())].to_numpy(dtype="float64"),
            expected[list(builder().feature_columns())].to_numpy(dtype="float64"),
            equal_nan=True,
        )
        assert int(row["cycle"]) == index + 1


def test_incremental_final_row_matches_offline_final_row() -> None:
    frame = make_trajectory_frame({7: 15})
    offline = builder().transform(frame).sort_values("cycle").reset_index(drop=True)

    state = IncrementalFeatureState(builder(), asset_id=7)
    row = pd.Series(dtype="float64")
    for observation in _asset_records(frame, 7):
        row = state.update(observation)

    expected = offline.iloc[-1]
    cols = list(builder().feature_columns())
    np.testing.assert_allclose(
        row[cols].to_numpy(dtype="float64"),
        expected[cols].to_numpy(dtype="float64"),
        equal_nan=True,
    )


def test_state_reconstruction_from_history_matches_streaming() -> None:
    frame = make_trajectory_frame({3: 10})
    records = _asset_records(frame, 3)

    streamed = IncrementalFeatureState(builder(), asset_id=3)
    for observation in records[:6]:
        streamed.update(observation)
    next_row_streamed = streamed.update(records[6])

    history = frame[frame["asset_id"] == 3].sort_values("cycle").head(6)
    restored = IncrementalFeatureState.from_history(builder(), asset_id=3, observations=history)
    assert restored.cycles_seen == 6
    next_row_restored = restored.update(records[6])

    cols = list(builder().feature_columns())
    np.testing.assert_allclose(
        next_row_streamed[cols].to_numpy(dtype="float64"),
        next_row_restored[cols].to_numpy(dtype="float64"),
        equal_nan=True,
    )


def test_incremental_rejects_out_of_order_cycles() -> None:
    state = IncrementalFeatureState(builder(), asset_id=1)
    frame = make_trajectory_frame({1: 3})
    records = _asset_records(frame, 1)
    state.update(records[2])  # cycle 3 first
    with pytest.raises(FeatureError, match="strictly increasing"):
        state.update(records[0])  # cycle 1 afterwards


def test_incremental_never_requires_future_observations() -> None:
    # After a single observation, a current-cycle row is produced without any
    # future data (rolling/lag values are simply structurally null/degenerate).
    state = IncrementalFeatureState(builder(), asset_id=1)
    frame = make_trajectory_frame({1: 1})
    row = state.update(_asset_records(frame, 1)[0])
    assert int(row["cycle"]) == 1

"""Replay source verification: split membership, integrity, payload generation."""

import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from turbine_guard.data.schema import (
    CYCLE_COLUMN,
    OPERATING_SETTING_COLUMNS,
    SENSOR_COLUMNS,
    TRAJECTORY_COLUMNS,
)
from turbine_guard.features.manifest import load_split_manifest
from turbine_guard.replay.client import build_reading_request
from turbine_guard.replay.errors import ReplaySourceError
from turbine_guard.replay.source import ReplaySource, ReplaySourceConfig

EPOCH = datetime(2026, 7, 13, 12, 0, tzinfo=UTC)


@pytest.fixture
def source(feature_data_dir: Path) -> ReplaySource:
    return ReplaySource(ReplaySourceConfig(data_dir=feature_data_dir))


def _config(data_dir: Path) -> ReplaySourceConfig:
    return ReplaySourceConfig(data_dir=data_dir)


def _replay_and_train_assets(data_dir: Path) -> tuple[list[int], list[int]]:
    manifest = load_split_manifest(_config(data_dir).split_manifest_path)
    return sorted(manifest.partitions["replay"]), sorted(manifest.partitions["train"])


class TestReplaySplitEnforcement:
    def test_replay_asset_ids_match_verified_manifest(
        self, source: ReplaySource, feature_data_dir: Path
    ) -> None:
        replay_assets, _ = _replay_and_train_assets(feature_data_dir)
        assert list(source.replay_asset_ids()) == replay_assets

    def test_train_split_asset_is_rejected(
        self, source: ReplaySource, feature_data_dir: Path
    ) -> None:
        _, train_assets = _replay_and_train_assets(feature_data_dir)
        with pytest.raises(ReplaySourceError, match="'train' split"):
            source.load_trajectory(train_assets[0])

    def test_missing_asset_is_rejected(self, source: ReplaySource) -> None:
        with pytest.raises(ReplaySourceError, match="does not exist in the split manifest"):
            source.load_trajectory(9999)


class TestTrajectoryLoading:
    def test_valid_replay_trajectory_loads_completely(
        self, source: ReplaySource, feature_data_dir: Path
    ) -> None:
        replay_assets, _ = _replay_and_train_assets(feature_data_dir)
        asset_id = replay_assets[0]
        trajectory = source.load_trajectory(asset_id)
        assert trajectory.source_asset_id == asset_id
        assert trajectory.final_cycle == 20 + asset_id  # conftest fixture lengths
        assert tuple(trajectory.frame.columns) == TRAJECTORY_COLUMNS
        assert trajectory.frame[CYCLE_COLUMN].to_list() == list(
            range(1, trajectory.final_cycle + 1)
        )
        assert set(trajectory.source_checksums) == {
            "processing_report_sha256",
            "split_manifest_sha256",
            "trajectory_parquet_sha256",
        }

    def test_row_returns_exactly_one_cycle(
        self, source: ReplaySource, feature_data_dir: Path
    ) -> None:
        replay_assets, _ = _replay_and_train_assets(feature_data_dir)
        trajectory = source.load_trajectory(replay_assets[0])
        row = trajectory.row(3)
        assert row[CYCLE_COLUMN] == 3.0
        with pytest.raises(ReplaySourceError):
            trajectory.row(trajectory.final_cycle + 1)


class TestIntegrityProtection:
    def test_tampered_trajectory_parquet_is_rejected(
        self, source: ReplaySource, feature_data_dir: Path
    ) -> None:
        parquet = _config(feature_data_dir).train_parquet_path
        with parquet.open("ab") as stream:
            stream.write(b"tampered")
        replay_assets, _ = _replay_and_train_assets(feature_data_dir)
        with pytest.raises(ReplaySourceError, match="tampered input"):
            source.load_trajectory(replay_assets[0])

    def test_tampered_split_manifest_is_rejected(
        self, source: ReplaySource, feature_data_dir: Path
    ) -> None:
        manifest_path = _config(feature_data_dir).split_manifest_path
        manifest_path.write_text(manifest_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with pytest.raises(ReplaySourceError, match="tampered"):
            source.replay_asset_ids()

    def test_missing_feature_manifest_is_rejected(
        self, source: ReplaySource, feature_data_dir: Path
    ) -> None:
        _config(feature_data_dir).feature_manifest_path.unlink()
        with pytest.raises(ReplaySourceError, match="missing"):
            source.replay_asset_ids()

    def test_noncontiguous_trajectory_is_rejected(
        self, source: ReplaySource, feature_data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        replay_assets, _ = _replay_and_train_assets(feature_data_dir)
        asset_id = replay_assets[0]
        original = pd.read_parquet(_config(feature_data_dir).train_parquet_path)
        doctored = original[~((original["asset_id"] == asset_id) & (original[CYCLE_COLUMN] == 2))]
        monkeypatch.setattr("turbine_guard.replay.source.pd.read_parquet", lambda _: doctored)
        with pytest.raises(ReplaySourceError, match="not contiguous"):
            source.load_trajectory(asset_id)

    def test_nonfinite_trajectory_is_rejected(
        self, source: ReplaySource, feature_data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        replay_assets, _ = _replay_and_train_assets(feature_data_dir)
        asset_id = replay_assets[0]
        doctored = pd.read_parquet(_config(feature_data_dir).train_parquet_path).copy()
        mask = (doctored["asset_id"] == asset_id) & (doctored[CYCLE_COLUMN] == 1)
        doctored.loc[mask, SENSOR_COLUMNS[0]] = float("nan")
        monkeypatch.setattr("turbine_guard.replay.source.pd.read_parquet", lambda _: doctored)
        with pytest.raises(ReplaySourceError, match="non-finite"):
            source.load_trajectory(asset_id)


class TestPayloadGeneration:
    def test_request_payload_matches_source_row_and_contract(
        self, source: ReplaySource, feature_data_dir: Path
    ) -> None:
        replay_assets, _ = _replay_and_train_assets(feature_data_dir)
        trajectory = source.load_trajectory(replay_assets[0])
        run_id = uuid.uuid4()
        request = build_reading_request(
            trajectory,
            4,
            run_id=run_id,
            external_asset_id="replay-FD001-001",
            replay_started_at=EPOCH,
            simulated_cycle_duration_seconds=2.0,
        )
        row = trajectory.row(4)
        assert request.cycle == 4
        assert request.external_asset_id == "replay-FD001-001"
        assert request.observed_at == EPOCH + timedelta(seconds=6.0)
        assert request.source == "replay"
        assert request.ingestion_id == f"replay-run:{run_id}:cycle:4"
        for name in (*OPERATING_SETTING_COLUMNS, *SENSOR_COLUMNS):
            assert getattr(request, name) == row[name]

    def test_payload_is_deterministic_for_retries(
        self, source: ReplaySource, feature_data_dir: Path
    ) -> None:
        replay_assets, _ = _replay_and_train_assets(feature_data_dir)
        trajectory = source.load_trajectory(replay_assets[0])
        run_id = uuid.uuid4()
        kwargs = {
            "run_id": run_id,
            "external_asset_id": "replay-FD001-001",
            "replay_started_at": EPOCH,
            "simulated_cycle_duration_seconds": 1.0,
        }
        first = build_reading_request(trajectory, 2, **kwargs)
        second = build_reading_request(trajectory, 2, **kwargs)
        assert first.model_dump(mode="json") == second.model_dump(mode="json")

    def test_future_row_mutation_cannot_change_earlier_payload(
        self, source: ReplaySource, feature_data_dir: Path
    ) -> None:
        replay_assets, _ = _replay_and_train_assets(feature_data_dir)
        trajectory = source.load_trajectory(replay_assets[0])
        kwargs = {
            "run_id": uuid.uuid4(),
            "external_asset_id": "replay-FD001-001",
            "replay_started_at": EPOCH,
            "simulated_cycle_duration_seconds": 1.0,
        }
        before = build_reading_request(trajectory, 3, **kwargs).model_dump(mode="json")
        future_mask = trajectory.frame[CYCLE_COLUMN] > 3
        trajectory.frame.loc[future_mask, SENSOR_COLUMNS[3]] = 999_999.0
        after = build_reading_request(trajectory, 3, **kwargs).model_dump(mode="json")
        assert before == after

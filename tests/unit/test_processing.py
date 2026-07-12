"""Tests for processing raw files into validated Parquet outputs."""

import hashlib
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest
from tests.conftest import make_trajectory_line

from turbine_guard.data.acquisition import AcquisitionConfig, acquire
from turbine_guard.data.processing import (
    ProcessingConfig,
    ProcessingError,
    ProcessingStatus,
    load_report,
    process,
)
from turbine_guard.data.schema import TRAJECTORY_COLUMNS, TRAJECTORY_DTYPES

ArchiveFactory = Callable[..., Path]


def config_for(data_dir: Path, **overrides: bool) -> ProcessingConfig:
    return ProcessingConfig(data_dir=data_dir, validate_canonical=False, **overrides)


def test_process_writes_outputs_and_report(acquired_data_dir: Path) -> None:
    result = process(config_for(acquired_data_dir))

    assert result.status is ProcessingStatus.PROCESSED
    assert result.report.passed
    for path in result.output_paths:
        assert path.exists()
    assert result.report_path.exists()

    reloaded = load_report(result.report_path)
    assert reloaded == result.report
    assert reloaded.source_archive_sha256
    assert len(reloaded.inputs) == 3
    assert len(reloaded.outputs) == 3


def test_output_checksums_match_files(acquired_data_dir: Path) -> None:
    result = process(config_for(acquired_data_dir))

    for record in result.report.outputs:
        path = result.report_path.parent / record.filename
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == record.sha256


def test_parquet_round_trip_schema_and_values(acquired_data_dir: Path) -> None:
    result = process(config_for(acquired_data_dir))
    train_path = next(path for path in result.output_paths if path.name.startswith("train"))

    frame = pd.read_parquet(train_path)

    assert list(frame.columns) == list(TRAJECTORY_COLUMNS)
    for column, dtype in TRAJECTORY_DTYPES.items():
        assert str(frame[column].dtype) == dtype, column
    assert len(frame) == 7
    assert frame["asset_id"].tolist() == [1, 1, 1, 1, 2, 2, 2]
    assert frame["cycle"].tolist() == [1, 2, 3, 4, 1, 2, 3]
    first_fields = make_trajectory_line(1, 1).split()
    assert frame.loc[0, "sensor_21"] == float(first_fields[25])


def test_output_rows_sorted_even_for_out_of_order_input(
    tmp_path: Path,
    cmapss_archive_factory: ArchiveFactory,
    full_cmapss_contents: dict[str, str],
) -> None:
    lines = [make_trajectory_line(2, 1), make_trajectory_line(1, 2), make_trajectory_line(1, 1)]
    contents = {**full_cmapss_contents, "train_FD001.txt": "\n".join(lines) + "\n"}
    archive = cmapss_archive_factory(contents=contents)
    data_dir = tmp_path / "data"
    acquire(AcquisitionConfig(data_dir=data_dir, source_url=archive.as_uri()))

    result = process(config_for(data_dir))
    train_path = next(path for path in result.output_paths if path.name.startswith("train"))
    frame = pd.read_parquet(train_path)

    assert frame["asset_id"].tolist() == [1, 1, 2]
    assert frame["cycle"].tolist() == [1, 2, 1]


def test_rerun_is_idempotent_and_does_not_rewrite(acquired_data_dir: Path) -> None:
    first = process(config_for(acquired_data_dir))
    mtimes = {path: path.stat().st_mtime_ns for path in first.output_paths}

    second = process(config_for(acquired_data_dir))

    assert second.status is ProcessingStatus.ALREADY_PROCESSED
    assert second.report == first.report
    assert {path: path.stat().st_mtime_ns for path in second.output_paths} == mtimes


def test_force_rebuilds_outputs(acquired_data_dir: Path) -> None:
    process(config_for(acquired_data_dir))

    result = process(config_for(acquired_data_dir, force=True))

    assert result.status is ProcessingStatus.PROCESSED


def test_tampered_output_detected(acquired_data_dir: Path) -> None:
    result = process(config_for(acquired_data_dir))
    victim = result.output_paths[0]
    victim.write_bytes(victim.read_bytes() + b"tampered")

    with pytest.raises(ProcessingError, match="Checksum mismatch"):
        process(config_for(acquired_data_dir))

    recovered = process(config_for(acquired_data_dir, force=True))
    assert recovered.status is ProcessingStatus.PROCESSED


def test_missing_output_detected(acquired_data_dir: Path) -> None:
    result = process(config_for(acquired_data_dir))
    result.output_paths[1].unlink()

    with pytest.raises(ProcessingError, match="missing"):
        process(config_for(acquired_data_dir))


def test_unreadable_report_rejected(acquired_data_dir: Path) -> None:
    result = process(config_for(acquired_data_dir))
    result.report_path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ProcessingError, match="could not be read"):
        process(config_for(acquired_data_dir))


def test_stale_report_triggers_reprocessing(acquired_data_dir: Path) -> None:
    result = process(config_for(acquired_data_dir))
    stale = result.report_path.read_text(encoding="utf-8").replace(
        result.report.inputs[0].sha256, "0" * 64
    )
    result.report_path.write_text(stale, encoding="utf-8")

    rerun = process(config_for(acquired_data_dir))

    assert rerun.status is ProcessingStatus.PROCESSED


def test_missing_acquisition_rejected(tmp_path: Path) -> None:
    with pytest.raises(ProcessingError, match="acquisition"):
        process(config_for(tmp_path / "data"))


def test_invalid_raw_data_blocks_publication(
    tmp_path: Path,
    cmapss_archive_factory: ArchiveFactory,
    full_cmapss_contents: dict[str, str],
) -> None:
    duplicate = make_trajectory_line(1, 1)
    contents = {**full_cmapss_contents, "train_FD001.txt": duplicate + "\n" + duplicate + "\n"}
    archive = cmapss_archive_factory(contents=contents)
    data_dir = tmp_path / "data"
    acquire(AcquisitionConfig(data_dir=data_dir, source_url=archive.as_uri()))
    config = config_for(data_dir)

    with pytest.raises(ProcessingError, match="unique_asset_cycle_pairs"):
        process(config)

    assert not any(config.processed_dir.glob("*.parquet"))
    assert not config.report_path.exists()


def test_malformed_raw_data_rejected(
    tmp_path: Path,
    cmapss_archive_factory: ArchiveFactory,
    full_cmapss_contents: dict[str, str],
) -> None:
    contents = {**full_cmapss_contents, "test_FD001.txt": "1 1 only three\n"}
    archive = cmapss_archive_factory(contents=contents)
    data_dir = tmp_path / "data"
    acquire(AcquisitionConfig(data_dir=data_dir, source_url=archive.as_uri()))

    with pytest.raises(ProcessingError, match="Parsing failed"):
        process(config_for(data_dir))


def test_canonical_profile_enforced_for_fd001(acquired_data_dir: Path) -> None:
    with pytest.raises(ProcessingError, match="canonical"):
        process(ProcessingConfig(data_dir=acquired_data_dir, validate_canonical=True))


def test_rul_test_asset_cross_check(
    tmp_path: Path,
    cmapss_archive_factory: ArchiveFactory,
    full_cmapss_contents: dict[str, str],
) -> None:
    contents = {**full_cmapss_contents, "RUL_FD001.txt": "112\n98\n7\n"}
    archive = cmapss_archive_factory(contents=contents)
    data_dir = tmp_path / "data"
    acquire(AcquisitionConfig(data_dir=data_dir, source_url=archive.as_uri()))

    with pytest.raises(ProcessingError, match="rul_count_matches_test_assets"):
        process(config_for(data_dir))


def test_raw_files_unchanged_by_processing(acquired_data_dir: Path) -> None:
    raw_dir = acquired_data_dir / "raw" / "cmapss" / "FD001"
    before = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in raw_dir.iterdir()
    }

    process(config_for(acquired_data_dir))

    after = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in raw_dir.iterdir()}
    assert after == before


def test_constant_column_warnings_recorded_in_report(acquired_data_dir: Path) -> None:
    result = process(config_for(acquired_data_dir))

    assert any("Constant columns" in warning for warning in result.report.warnings)

"""Tests for manifest persistence."""

from datetime import UTC, datetime
from pathlib import Path

from turbine_guard.data.manifest import (
    AcquisitionManifest,
    FileRecord,
    load_manifest,
    write_manifest,
)


def _sample_manifest() -> AcquisitionManifest:
    return AcquisitionManifest(
        acquisition_version="1",
        dataset_name="NASA C-MAPSS Turbofan Engine Degradation Simulation",
        dataset_subset="FD001",
        source_name="NASA Prognostics Center of Excellence data repository",
        source_url="file:///tmp/archive.zip",
        retrieved_at=datetime(2026, 7, 12, 12, 0, 0, tzinfo=UTC),
        acquired_by="turbine-guard 0.1.0",
        git_commit=None,
        archive=FileRecord(filename="archive.zip", sha256="ab" * 32, size_bytes=10),
        files=(
            FileRecord(
                filename="train_FD001.txt",
                sha256="cd" * 32,
                size_bytes=100,
                record_count=5,
                asset_count=2,
            ),
        ),
        notes="Simulated turbofan degradation data; sensor channels are anonymous.",
    )


def test_write_and_load_round_trip(tmp_path: Path) -> None:
    manifest = _sample_manifest()
    path = tmp_path / "manifests" / "cmapss_fd001.json"

    write_manifest(manifest, path)
    loaded = load_manifest(path)

    assert loaded == manifest


def test_write_is_atomic_and_leaves_no_temp_file(tmp_path: Path) -> None:
    path = tmp_path / "cmapss_fd001.json"

    write_manifest(_sample_manifest(), path)

    assert path.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_written_manifest_is_readable_json_with_trailing_newline(tmp_path: Path) -> None:
    path = tmp_path / "cmapss_fd001.json"

    write_manifest(_sample_manifest(), path)
    text = path.read_text(encoding="utf-8")

    assert text.endswith("\n")
    assert '"dataset_subset": "FD001"' in text

"""Tests for C-MAPSS dataset acquisition. Fully offline: file:// URLs only."""

import dataclasses
import hashlib
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from turbine_guard.data.acquisition import (
    AcquisitionConfig,
    AcquisitionError,
    AcquisitionStatus,
    acquire,
    current_git_commit,
)
from turbine_guard.data.manifest import load_manifest

ArchiveFactory = Callable[..., Path]

SUBSET_FILES = ("train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt")


def _config(data_dir: Path, archive: Path, **overrides: bool) -> AcquisitionConfig:
    return AcquisitionConfig(data_dir=data_dir, source_url=archive.as_uri(), **overrides)


def test_acquires_flat_archive(tmp_path: Path, cmapss_archive_factory: ArchiveFactory) -> None:
    config = _config(tmp_path / "data", cmapss_archive_factory())

    result = acquire(config)

    assert result.status is AcquisitionStatus.ACQUIRED
    for filename in SUBSET_FILES:
        assert (config.subset_dir / filename).exists()
    assert result.manifest_path.exists()


def test_acquires_nested_archive(tmp_path: Path, cmapss_archive_factory: ArchiveFactory) -> None:
    config = _config(tmp_path / "data", cmapss_archive_factory(nested=True))

    result = acquire(config)

    assert result.status is AcquisitionStatus.ACQUIRED
    for filename in SUBSET_FILES:
        assert (config.subset_dir / filename).exists()


def test_manifest_records_required_provenance_fields(
    tmp_path: Path,
    cmapss_archive_factory: ArchiveFactory,
    cmapss_member_contents: dict[str, str],
) -> None:
    archive = cmapss_archive_factory()
    config = _config(tmp_path / "data", archive)

    result = acquire(config)
    manifest = load_manifest(result.manifest_path)

    assert manifest.dataset_name == "NASA C-MAPSS Turbofan Engine Degradation Simulation"
    assert manifest.dataset_subset == "FD001"
    assert manifest.source_url == archive.as_uri()
    assert manifest.acquisition_version == "1"
    assert manifest.retrieved_at.tzinfo is not None
    assert "turbine-guard" in manifest.acquired_by
    assert "anonymous" in manifest.notes

    archive_bytes = archive.read_bytes()
    assert manifest.archive.sha256 == hashlib.sha256(archive_bytes).hexdigest()
    assert manifest.archive.size_bytes == len(archive_bytes)

    by_name = {record.filename: record for record in manifest.files}
    assert set(by_name) == set(SUBSET_FILES)
    for filename, content in cmapss_member_contents.items():
        record = by_name[filename]
        assert record.sha256 == hashlib.sha256(content.encode()).hexdigest()
        assert record.size_bytes == len(content.encode())

    assert by_name["train_FD001.txt"].record_count == 5
    assert by_name["train_FD001.txt"].asset_count == 2
    assert by_name["test_FD001.txt"].record_count == 3
    assert by_name["test_FD001.txt"].asset_count == 2
    assert by_name["RUL_FD001.txt"].record_count == 2
    assert by_name["RUL_FD001.txt"].asset_count is None


def test_raw_files_are_read_only(tmp_path: Path, cmapss_archive_factory: ArchiveFactory) -> None:
    config = _config(tmp_path / "data", cmapss_archive_factory())

    acquire(config)

    for filename in SUBSET_FILES:
        mode = (config.subset_dir / filename).stat().st_mode
        assert mode & 0o222 == 0, f"{filename} should be read-only"


def test_rerun_is_idempotent(tmp_path: Path, cmapss_archive_factory: ArchiveFactory) -> None:
    config = _config(tmp_path / "data", cmapss_archive_factory())

    first = acquire(config)
    manifest_text = first.manifest_path.read_text(encoding="utf-8")
    mtimes = {
        filename: (config.subset_dir / filename).stat().st_mtime_ns for filename in SUBSET_FILES
    }

    second = acquire(config)

    assert second.status is AcquisitionStatus.ALREADY_ACQUIRED
    assert second.manifest_path.read_text(encoding="utf-8") == manifest_text
    for filename in SUBSET_FILES:
        assert (config.subset_dir / filename).stat().st_mtime_ns == mtimes[filename]


def test_rerun_uses_cached_archive_when_source_is_gone(
    tmp_path: Path, cmapss_archive_factory: ArchiveFactory
) -> None:
    archive = cmapss_archive_factory()
    config = _config(tmp_path / "data", archive)
    acquire(config)

    archive.unlink()  # the original source disappears
    config.manifest_path.unlink()  # raw layer lost its manifest
    shutil.rmtree(config.subset_dir)

    result = acquire(config)  # must succeed from the cached archive alone

    assert result.status is AcquisitionStatus.ACQUIRED


def test_tampered_raw_file_is_detected(
    tmp_path: Path, cmapss_archive_factory: ArchiveFactory
) -> None:
    config = _config(tmp_path / "data", cmapss_archive_factory())
    acquire(config)

    tampered = config.subset_dir / "train_FD001.txt"
    tampered.chmod(0o644)
    tampered.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(AcquisitionError, match="Checksum mismatch"):
        acquire(config)


def test_missing_raw_file_is_detected(
    tmp_path: Path, cmapss_archive_factory: ArchiveFactory
) -> None:
    config = _config(tmp_path / "data", cmapss_archive_factory())
    acquire(config)

    (config.subset_dir / "RUL_FD001.txt").unlink()

    with pytest.raises(AcquisitionError, match="missing"):
        acquire(config)


def test_force_replaces_tampered_raw_layer(
    tmp_path: Path,
    cmapss_archive_factory: ArchiveFactory,
    cmapss_member_contents: dict[str, str],
) -> None:
    config = _config(tmp_path / "data", cmapss_archive_factory())
    acquire(config)

    tampered = config.subset_dir / "train_FD001.txt"
    tampered.chmod(0o644)
    tampered.write_text("tampered\n", encoding="utf-8")

    result = acquire(dataclasses.replace(config, force=True))

    assert result.status is AcquisitionStatus.ACQUIRED
    expected = hashlib.sha256(cmapss_member_contents["train_FD001.txt"].encode()).hexdigest()
    restored = hashlib.sha256(tampered.read_bytes()).hexdigest()
    assert restored == expected


def test_archive_missing_member_raises(
    tmp_path: Path, cmapss_archive_factory: ArchiveFactory
) -> None:
    archive = cmapss_archive_factory(omit=("RUL_FD001.txt",))
    config = _config(tmp_path / "data", archive)

    with pytest.raises(AcquisitionError, match=r"RUL_FD001\.txt"):
        acquire(config)


def test_unreachable_source_raises_clear_error(tmp_path: Path) -> None:
    config = AcquisitionConfig(
        data_dir=tmp_path / "data",
        source_url=(tmp_path / "does_not_exist.zip").as_uri(),
    )

    with pytest.raises(AcquisitionError, match="Could not download"):
        acquire(config)


def test_invalid_archive_raises_clear_error(tmp_path: Path) -> None:
    bogus = tmp_path / "bogus.zip"
    bogus.write_text("this is not a zip file", encoding="utf-8")
    config = _config(tmp_path / "data", bogus)

    with pytest.raises(AcquisitionError, match="not a valid zip"):
        acquire(config)


def test_git_commit_is_none_outside_a_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert current_git_commit() is None


def test_git_commit_resolves_inside_a_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "-q"], check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=test@example.com",
            "-c",
            "user.name=test",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--allow-empty",
            "-q",
            "-m",
            "initial",
        ],
        check=True,
    )

    commit = current_git_commit()

    assert commit is not None
    assert len(commit) == 40

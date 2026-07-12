"""Tests for the acquisition command-line interface."""

from collections.abc import Callable
from pathlib import Path

from turbine_guard.data.cli import main

ArchiveFactory = Callable[..., Path]


def test_cli_acquires_and_exits_zero(
    tmp_path: Path, cmapss_archive_factory: ArchiveFactory
) -> None:
    archive = cmapss_archive_factory()
    data_dir = tmp_path / "data"

    exit_code = main(["--data-dir", str(data_dir), "--url", archive.as_uri()])

    assert exit_code == 0
    assert (data_dir / "manifests" / "cmapss_fd001.json").exists()


def test_cli_rerun_exits_zero(tmp_path: Path, cmapss_archive_factory: ArchiveFactory) -> None:
    archive = cmapss_archive_factory()
    args = ["--data-dir", str(tmp_path / "data"), "--url", archive.as_uri()]

    assert main(args) == 0
    assert main(args) == 0


def test_cli_reports_failure_with_exit_code_one(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.zip"

    exit_code = main(["--data-dir", str(tmp_path / "data"), "--url", missing.as_uri()])

    assert exit_code == 1

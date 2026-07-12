"""Tests for the processing command-line interface."""

from pathlib import Path

import pytest

from turbine_guard.data import validation
from turbine_guard.data.process_cli import main
from turbine_guard.data.validation import CanonicalProfile


@pytest.fixture
def fixture_sized_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink the FD001 canonical profile to the test fixture's dimensions."""
    monkeypatch.setitem(
        validation.CANONICAL_PROFILES,
        "FD001",
        CanonicalProfile(
            subset="FD001",
            train_rows=7,
            train_assets=2,
            test_rows=5,
            test_assets=2,
            rul_values=2,
        ),
    )


@pytest.mark.usefixtures("fixture_sized_profile")
def test_cli_processes_and_exits_zero(acquired_data_dir: Path) -> None:
    exit_code = main(["--data-dir", str(acquired_data_dir)])

    assert exit_code == 0
    processed = acquired_data_dir / "processed" / "cmapss" / "FD001"
    assert (processed / "train_FD001.parquet").exists()
    assert (processed / "test_FD001.parquet").exists()
    assert (processed / "rul_FD001.parquet").exists()
    assert (processed / "processing_report.json").exists()


@pytest.mark.usefixtures("fixture_sized_profile")
def test_cli_rerun_exits_zero(acquired_data_dir: Path) -> None:
    args = ["--data-dir", str(acquired_data_dir)]

    assert main(args) == 0
    assert main(args) == 0


def test_cli_fails_without_acquisition(tmp_path: Path) -> None:
    exit_code = main(["--data-dir", str(tmp_path / "data")])

    assert exit_code == 1


def test_cli_fails_on_non_canonical_data(acquired_data_dir: Path) -> None:
    # The tiny fixture cannot satisfy the real FD001 canonical counts, so the
    # CLI must fail loudly instead of publishing outputs.
    exit_code = main(["--data-dir", str(acquired_data_dir)])

    assert exit_code == 1
    assert not (acquired_data_dir / "processed" / "cmapss" / "FD001").exists()

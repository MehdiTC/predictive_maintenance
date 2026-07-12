"""Tests for the feature-build command-line interface."""

from pathlib import Path

from turbine_guard.features.build_cli import main


def test_cli_builds_and_exits_zero(processed_data_dir: Path) -> None:
    exit_code = main(["--data-dir", str(processed_data_dir)])

    assert exit_code == 0
    features_dir = processed_data_dir / "features" / "cmapss" / "FD001"
    assert (features_dir / "train.parquet").exists()
    assert (features_dir / "feature_manifest.json").exists()
    assert (features_dir / "split_manifest.json").exists()


def test_cli_rerun_exits_zero(processed_data_dir: Path) -> None:
    args = ["--data-dir", str(processed_data_dir)]
    assert main(args) == 0
    assert main(args) == 0


def test_cli_accepts_seed_and_rul_cap(processed_data_dir: Path) -> None:
    exit_code = main(["--data-dir", str(processed_data_dir), "--seed", "5", "--rul-cap", "15"])
    assert exit_code == 0


def test_cli_fails_without_processed_inputs(tmp_path: Path) -> None:
    exit_code = main(["--data-dir", str(tmp_path / "data")])
    assert exit_code == 1


def test_cli_fails_on_invalid_configuration(processed_data_dir: Path) -> None:
    # A negative RUL cap is rejected by RulConfig; the CLI must exit nonzero
    # instead of raising or writing outputs.
    exit_code = main(["--data-dir", str(processed_data_dir), "--rul-cap", "-4"])

    assert exit_code == 1
    assert not (processed_data_dir / "features").exists()


def test_cli_force_rebuild_exits_zero(processed_data_dir: Path) -> None:
    assert main(["--data-dir", str(processed_data_dir)]) == 0
    assert main(["--data-dir", str(processed_data_dir), "--force"]) == 0

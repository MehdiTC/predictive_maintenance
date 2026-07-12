"""Tests for the structural and semantic validation layer."""

import json
from pathlib import Path

import pandas as pd
from tests.conftest import make_trajectory_line, make_trajectory_text

from turbine_guard.data.parsing import parse_rul_file, parse_trajectory_file
from turbine_guard.data.validation import (
    DatasetValidation,
    validate_rul_frame,
    validate_trajectory_frame,
)


def parse_text(tmp_path: Path, text: str) -> pd.DataFrame:
    path = tmp_path / "trajectories.txt"
    path.write_text(text, encoding="utf-8")
    return parse_trajectory_file(path)


def failed_names(validation: DatasetValidation) -> set[str]:
    return {check.name for check in validation.failed_checks}


def test_valid_frame_passes_with_correct_stats(tmp_path: Path) -> None:
    frame = parse_text(tmp_path, make_trajectory_text({1: 4, 2: 3, 3: 5}))

    result = validate_trajectory_frame(frame, dataset="train")

    assert result.passed
    assert not failed_names(result)
    stats = result.trajectory_stats
    assert stats is not None
    assert stats.row_count == 12
    assert stats.asset_count == 3
    assert stats.trajectory_length_min == 3
    assert stats.trajectory_length_max == 5
    assert stats.trajectory_length_median == 4.0
    assert stats.duplicate_pair_count == 0
    assert stats.missing_values_per_column == {}
    assert stats.non_finite_values_per_column == {}


def test_constant_columns_reported_as_warning_not_failure(tmp_path: Path) -> None:
    frame = parse_text(tmp_path, make_trajectory_text({1: 6, 2: 6}))

    result = validate_trajectory_frame(frame, dataset="train")

    assert result.passed
    stats = result.trajectory_stats
    assert stats is not None
    assert "sensor_01" in stats.constant_columns
    assert "sensor_05" in stats.constant_columns
    assert "operating_setting_3" in stats.constant_columns
    assert any("Constant columns" in warning for warning in result.warnings)


def test_near_constant_columns_reported(tmp_path: Path) -> None:
    frame = parse_text(tmp_path, make_trajectory_text({1: 6, 2: 6}))

    result = validate_trajectory_frame(frame, dataset="train")

    stats = result.trajectory_stats
    assert stats is not None
    assert "sensor_06" in stats.near_constant_columns
    assert "sensor_06" not in stats.constant_columns


def test_duplicate_asset_cycle_pairs_fail(tmp_path: Path) -> None:
    line = make_trajectory_line(1, 1)
    frame = parse_text(tmp_path, line + "\n" + line + "\n")

    result = validate_trajectory_frame(frame, dataset="train")

    assert not result.passed
    assert "unique_asset_cycle_pairs" in failed_names(result)
    stats = result.trajectory_stats
    assert stats is not None
    assert stats.duplicate_pair_count == 1


def test_zero_asset_id_fails(tmp_path: Path) -> None:
    frame = parse_text(tmp_path, make_trajectory_line(0, 1) + "\n")

    result = validate_trajectory_frame(frame, dataset="train")

    assert "asset_ids_positive" in failed_names(result)


def test_zero_cycle_fails(tmp_path: Path) -> None:
    text = make_trajectory_line(1, 0) + "\n" + make_trajectory_line(1, 1) + "\n"
    frame = parse_text(tmp_path, text)

    result = validate_trajectory_frame(frame, dataset="train")

    assert "cycles_positive" in failed_names(result)


def test_missing_cycle_in_sequence_fails(tmp_path: Path) -> None:
    lines = [make_trajectory_line(1, cycle) for cycle in (1, 2, 4)]
    frame = parse_text(tmp_path, "\n".join(lines) + "\n")

    result = validate_trajectory_frame(frame, dataset="train")

    assert "cycles_contiguous_from_one" in failed_names(result)


def test_asset_not_starting_at_cycle_one_fails(tmp_path: Path) -> None:
    lines = [make_trajectory_line(1, cycle) for cycle in (2, 3, 4)]
    frame = parse_text(tmp_path, "\n".join(lines) + "\n")

    result = validate_trajectory_frame(frame, dataset="train")

    assert "cycles_contiguous_from_one" in failed_names(result)


def test_out_of_order_rows_still_pass(tmp_path: Path) -> None:
    lines = [make_trajectory_line(2, 1), make_trajectory_line(1, 2), make_trajectory_line(1, 1)]
    frame = parse_text(tmp_path, "\n".join(lines) + "\n")

    result = validate_trajectory_frame(frame, dataset="train")

    assert result.passed


def test_missing_values_detected(tmp_path: Path) -> None:
    fields = make_trajectory_line(1, 1).split()
    fields[10] = "NaN"
    text = " ".join(fields) + "\n" + make_trajectory_line(1, 2) + "\n"
    frame = parse_text(tmp_path, text)

    result = validate_trajectory_frame(frame, dataset="train")

    assert "no_missing_values" in failed_names(result)
    stats = result.trajectory_stats
    assert stats is not None
    assert stats.missing_values_per_column == {"sensor_06": 1}


def test_non_finite_values_detected(tmp_path: Path) -> None:
    fields = make_trajectory_line(1, 1).split()
    fields[2] = "inf"
    text = " ".join(fields) + "\n" + make_trajectory_line(1, 2) + "\n"
    frame = parse_text(tmp_path, text)

    result = validate_trajectory_frame(frame, dataset="train")

    assert "all_values_finite" in failed_names(result)
    stats = result.trajectory_stats
    assert stats is not None
    assert stats.non_finite_values_per_column == {"operating_setting_1": 1}


def test_wrong_dtype_fails(tmp_path: Path) -> None:
    frame = parse_text(tmp_path, make_trajectory_text({1: 3}))
    frame["cycle"] = frame["cycle"].astype("float64")

    result = validate_trajectory_frame(frame, dataset="train")

    assert "dtypes_match_schema" in failed_names(result)


def test_unexpected_column_fails(tmp_path: Path) -> None:
    frame = parse_text(tmp_path, make_trajectory_text({1: 3}))
    frame["surprise"] = 1.0

    result = validate_trajectory_frame(frame, dataset="train")

    assert "no_unexpected_columns" in failed_names(result)


def test_missing_column_fails(tmp_path: Path) -> None:
    frame = parse_text(tmp_path, make_trajectory_text({1: 3})).drop(columns=["sensor_21"])

    result = validate_trajectory_frame(frame, dataset="train")

    assert "required_columns_present" in failed_names(result)


def test_empty_frame_fails(tmp_path: Path) -> None:
    frame = parse_text(tmp_path, make_trajectory_text({1: 3})).iloc[0:0]

    result = validate_trajectory_frame(frame, dataset="train")

    assert "not_empty" in failed_names(result)


def test_canonical_profile_mismatch_fails(tmp_path: Path) -> None:
    frame = parse_text(tmp_path, make_trajectory_text({1: 4, 2: 3}))

    result = validate_trajectory_frame(
        frame, dataset="train", expected_rows=20_631, expected_assets=100
    )

    assert {"canonical_row_count", "canonical_asset_count"} <= failed_names(result)


def test_canonical_profile_match_passes(tmp_path: Path) -> None:
    frame = parse_text(tmp_path, make_trajectory_text({1: 4, 2: 3}))

    result = validate_trajectory_frame(frame, dataset="train", expected_rows=7, expected_assets=2)

    assert result.passed


def test_valid_rul_frame_passes(tmp_path: Path) -> None:
    path = tmp_path / "RUL_FD001.txt"
    path.write_text("112\n98\n69\n", encoding="utf-8")

    result = validate_rul_frame(parse_rul_file(path), expected_values=3)

    assert result.passed
    assert result.rul_stats is not None
    assert result.rul_stats.row_count == 3
    assert result.rul_stats.minimum == 69
    assert result.rul_stats.maximum == 112


def test_negative_rul_fails(tmp_path: Path) -> None:
    path = tmp_path / "RUL_FD001.txt"
    path.write_text("112\n-5\n", encoding="utf-8")

    result = validate_rul_frame(parse_rul_file(path))

    assert "rul_values_non_negative" in failed_names(result)


def test_rul_count_mismatch_fails(tmp_path: Path) -> None:
    path = tmp_path / "RUL_FD001.txt"
    path.write_text("112\n98\n", encoding="utf-8")

    result = validate_rul_frame(parse_rul_file(path), expected_values=100)

    assert "canonical_rul_count" in failed_names(result)


def test_validation_result_serializes_to_json(tmp_path: Path) -> None:
    frame = parse_text(tmp_path, make_trajectory_text({1: 4, 2: 3}))

    result = validate_trajectory_frame(frame, dataset="train")
    payload = json.loads(result.model_dump_json())

    assert payload["dataset"] == "train"
    assert {check["name"] for check in payload["checks"]}
    assert payload["trajectory_stats"]["row_count"] == 7

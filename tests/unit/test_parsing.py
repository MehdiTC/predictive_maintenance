"""Tests for the C-MAPSS raw-file parser."""

import hashlib
from pathlib import Path

import pandas as pd
import pytest
from tests.conftest import make_trajectory_line, make_trajectory_text

from turbine_guard.data.parsing import ParseError, parse_rul_file, parse_trajectory_file
from turbine_guard.data.schema import (
    RUL_COLUMN,
    TRAJECTORY_COLUMNS,
    TRAJECTORY_DTYPES,
)


@pytest.fixture
def trajectory_path(tmp_path: Path) -> Path:
    path = tmp_path / "train_FD001.txt"
    path.write_text(make_trajectory_text({1: 4, 2: 3}), encoding="utf-8")
    return path


def test_parses_valid_trajectory_file(trajectory_path: Path) -> None:
    frame = parse_trajectory_file(trajectory_path)

    assert list(frame.columns) == list(TRAJECTORY_COLUMNS)
    assert len(frame) == 7
    assert frame["asset_id"].tolist() == [1, 1, 1, 1, 2, 2, 2]
    assert frame["cycle"].tolist() == [1, 2, 3, 4, 1, 2, 3]


def test_trajectory_dtypes_match_schema(trajectory_path: Path) -> None:
    frame = parse_trajectory_file(trajectory_path)

    for column, dtype in TRAJECTORY_DTYPES.items():
        assert str(frame[column].dtype) == dtype, column


def test_trajectory_values_preserved(trajectory_path: Path) -> None:
    frame = parse_trajectory_file(trajectory_path)
    first_line_fields = make_trajectory_line(1, 1).split()

    assert frame.loc[0, "operating_setting_1"] == float(first_line_fields[2])
    assert frame.loc[0, "sensor_01"] == float(first_line_fields[5])
    assert frame.loc[0, "sensor_21"] == float(first_line_fields[25])


def test_trailing_whitespace_and_blank_lines_handled(tmp_path: Path) -> None:
    path = tmp_path / "train_FD001.txt"
    body = make_trajectory_line(1, 1) + "   \n\n" + make_trajectory_line(1, 2) + "\n   \n"
    path.write_text(body, encoding="utf-8")

    frame = parse_trajectory_file(path)

    assert len(frame) == 2
    assert frame.shape[1] == len(TRAJECTORY_COLUMNS)


def test_row_order_preserved_for_out_of_order_input(tmp_path: Path) -> None:
    path = tmp_path / "train_FD001.txt"
    lines = [make_trajectory_line(1, 2), make_trajectory_line(1, 1)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    frame = parse_trajectory_file(path)

    assert frame["cycle"].tolist() == [2, 1]


def test_parsing_is_deterministic(trajectory_path: Path) -> None:
    first = parse_trajectory_file(trajectory_path)
    second = parse_trajectory_file(trajectory_path)

    pd.testing.assert_frame_equal(first, second)


def test_raw_file_bytes_unchanged_by_parsing(trajectory_path: Path) -> None:
    before = hashlib.sha256(trajectory_path.read_bytes()).hexdigest()

    parse_trajectory_file(trajectory_path)

    assert hashlib.sha256(trajectory_path.read_bytes()).hexdigest() == before


def test_missing_file_rejected(tmp_path: Path) -> None:
    with pytest.raises(ParseError, match="does not exist"):
        parse_trajectory_file(tmp_path / "absent.txt")


def test_empty_file_rejected(tmp_path: Path) -> None:
    path = tmp_path / "train_FD001.txt"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ParseError, match="no data rows"):
        parse_trajectory_file(path)


def test_whitespace_only_file_rejected(tmp_path: Path) -> None:
    path = tmp_path / "train_FD001.txt"
    path.write_text("   \n \n", encoding="utf-8")

    with pytest.raises(ParseError, match="no data rows"):
        parse_trajectory_file(path)


def test_wrong_column_count_rejected_with_line_number(tmp_path: Path) -> None:
    path = tmp_path / "train_FD001.txt"
    short_line = " ".join(make_trajectory_line(1, 2).split()[:-1])
    path.write_text(make_trajectory_line(1, 1) + "\n" + short_line + "\n", encoding="utf-8")

    with pytest.raises(ParseError, match=r"line 2 has 25"):
        parse_trajectory_file(path)


def test_non_numeric_value_rejected_with_column_name(tmp_path: Path) -> None:
    path = tmp_path / "train_FD001.txt"
    fields = make_trajectory_line(1, 1).split()
    fields[5] = "not-a-number"
    path.write_text(" ".join(fields) + "\n", encoding="utf-8")

    with pytest.raises(ParseError, match=r"sensor_01.*not-a-number"):
        parse_trajectory_file(path)


def test_non_integer_asset_id_rejected(tmp_path: Path) -> None:
    path = tmp_path / "train_FD001.txt"
    fields = make_trajectory_line(1, 1).split()
    fields[0] = "1.5"
    path.write_text(" ".join(fields) + "\n", encoding="utf-8")

    with pytest.raises(ParseError, match="asset_id"):
        parse_trajectory_file(path)


def test_non_integer_cycle_rejected(tmp_path: Path) -> None:
    path = tmp_path / "train_FD001.txt"
    fields = make_trajectory_line(1, 1).split()
    fields[1] = "2.25"
    path.write_text(" ".join(fields) + "\n", encoding="utf-8")

    with pytest.raises(ParseError, match="cycle"):
        parse_trajectory_file(path)


def test_parses_valid_rul_file(tmp_path: Path) -> None:
    path = tmp_path / "RUL_FD001.txt"
    path.write_text("112 \n98 \n69\n", encoding="utf-8")

    frame = parse_rul_file(path)

    assert list(frame.columns) == [RUL_COLUMN]
    assert str(frame[RUL_COLUMN].dtype) == "int64"
    assert frame[RUL_COLUMN].tolist() == [112, 98, 69]


def test_rul_file_with_extra_fields_rejected(tmp_path: Path) -> None:
    path = tmp_path / "RUL_FD001.txt"
    path.write_text("112 3\n", encoding="utf-8")

    with pytest.raises(ParseError, match=r"line 1 has 2"):
        parse_rul_file(path)


def test_rul_file_with_non_integer_rejected(tmp_path: Path) -> None:
    path = tmp_path / "RUL_FD001.txt"
    path.write_text("112\nabc\n", encoding="utf-8")

    with pytest.raises(ParseError, match="rul"):
        parse_rul_file(path)


def test_rul_empty_file_rejected(tmp_path: Path) -> None:
    path = tmp_path / "RUL_FD001.txt"
    path.write_text("\n", encoding="utf-8")

    with pytest.raises(ParseError, match="no data rows"):
        parse_rul_file(path)

"""Parsing of the whitespace-delimited C-MAPSS raw files into typed frames.

The parser reads raw files without modifying them, assigns the canonical
column names from :mod:`turbine_guard.data.schema`, applies explicit numeric
dtypes, and raises :class:`ParseError` with the offending line number for any
malformed input instead of silently dropping rows. Non-finite values such as
``NaN`` or ``inf`` are numeric and therefore parse successfully; detecting and
rejecting them is the validation layer's responsibility, so their presence is
reported rather than hidden at parse time.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from turbine_guard.data.schema import RUL_COLUMN, TRAJECTORY_COLUMNS, TRAJECTORY_DTYPES

logger = logging.getLogger(__name__)

_MAX_REPORTED_ERRORS = 5


class ParseError(RuntimeError):
    """Raised when a raw file cannot be parsed into the canonical schema."""


def parse_trajectory_file(path: Path) -> pd.DataFrame:
    """Parse a C-MAPSS trajectory file into a canonical typed DataFrame.

    Returns a frame with the columns of ``TRAJECTORY_COLUMNS`` in order:
    ``asset_id`` and ``cycle`` as ``int64``, operating settings and sensors as
    ``float64``. Rows keep the order they appear in the file. Raises
    :class:`ParseError` for missing/empty files, rows with a wrong field
    count, or values that cannot be converted to the schema dtype.
    """
    rows = _read_rows(path, expected_fields=len(TRAJECTORY_COLUMNS))
    frame = pd.DataFrame(rows, columns=list(TRAJECTORY_COLUMNS), dtype="object")
    typed = _apply_dtypes(frame, TRAJECTORY_DTYPES, path)
    logger.info(
        "trajectory_file_parsed",
        extra={"path": str(path), "row_count": len(typed)},
    )
    return typed


def parse_rul_file(path: Path) -> pd.DataFrame:
    """Parse the official RUL file into a single-column ``int64`` DataFrame.

    Each non-empty line must contain exactly one integer: the remaining useful
    life of the corresponding test unit, in file order.
    """
    rows = _read_rows(path, expected_fields=1)
    frame = pd.DataFrame(rows, columns=[RUL_COLUMN], dtype="object")
    typed = _apply_dtypes(frame, {RUL_COLUMN: "int64"}, path)
    logger.info("rul_file_parsed", extra={"path": str(path), "row_count": len(typed)})
    return typed


def _read_rows(path: Path, expected_fields: int) -> list[list[str]]:
    """Split a raw file into whitespace-delimited rows, enforcing field count.

    Whitespace-only lines (including the trailing newline at end of file) are
    not data rows and are skipped; every other line must have exactly
    ``expected_fields`` fields, otherwise the line numbers of the offenders
    are reported.
    """
    if not path.exists():
        raise ParseError(f"Raw file {path} does not exist.")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ParseError(f"Raw file {path} could not be read: {exc}") from exc

    rows: list[list[str]] = []
    bad_lines: list[tuple[int, int]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != expected_fields:
            bad_lines.append((line_number, len(fields)))
        else:
            rows.append(fields)
    if bad_lines:
        reported = bad_lines[:_MAX_REPORTED_ERRORS]
        shown = ", ".join(f"line {number} has {count}" for number, count in reported)
        remaining = len(bad_lines) - len(reported)
        suffix = f" (+{remaining} more)" if remaining else ""
        raise ParseError(
            f"{path} has {len(bad_lines)} malformed row(s): expected "
            f"{expected_fields} whitespace-delimited fields but {shown}{suffix}."
        )
    if not rows:
        raise ParseError(f"{path} contains no data rows.")
    return rows


def _apply_dtypes(frame: pd.DataFrame, dtypes: dict[str, str], path: Path) -> pd.DataFrame:
    """Convert string columns to their schema dtypes with clear errors."""
    converted: dict[str, pd.Series[Any]] = {}
    for column, dtype in dtypes.items():
        try:
            converted[column] = frame[column].astype(np.dtype(dtype))
        except (ValueError, TypeError) as exc:
            offender = _first_unconvertible(frame[column], dtype)
            location = f" (first offending value {offender!r})" if offender is not None else ""
            raise ParseError(
                f"{path}: column '{column}' contains values that are not valid {dtype}{location}."
            ) from exc
    return pd.DataFrame(converted, columns=list(dtypes))


def _first_unconvertible(series: "pd.Series[str]", dtype: str) -> str | None:
    """Find the first value that fails conversion, for error messages."""
    caster = int if dtype == "int64" else float
    for value in series:
        try:
            caster(value)
        except (ValueError, TypeError):
            return str(value)
    return None

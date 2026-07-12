"""Tests for the structured JSON logging foundation."""

import json
import logging
import sys
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest

from turbine_guard.logging_config import JsonLogFormatter, configure_logging


@pytest.fixture(autouse=True)
def _restore_root_logger() -> Iterator[None]:
    """Put the root logger back the way it was so tests stay isolated."""
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    yield
    root.handlers[:] = original_handlers
    root.setLevel(original_level)


def _make_record(**extra: object) -> logging.LogRecord:
    record = logging.LogRecord(
        name="turbine_guard.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="something %s happened",
        args=("interesting",),
        exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_formatter_emits_json_with_core_fields() -> None:
    payload = json.loads(JsonLogFormatter().format(_make_record()))

    assert payload["level"] == "INFO"
    assert payload["logger"] == "turbine_guard.test"
    assert payload["message"] == "something interesting happened"
    assert payload["timestamp"].endswith("+00:00")


def test_formatter_includes_extra_fields() -> None:
    payload = json.loads(JsonLogFormatter().format(_make_record(asset_id=7, cycle=42)))

    assert payload["asset_id"] == 7
    assert payload["cycle"] == 42


def test_formatter_stringifies_values_json_cannot_encode() -> None:
    timestamp = datetime(2026, 7, 12, tzinfo=UTC)
    payload = json.loads(JsonLogFormatter().format(_make_record(observed_at=timestamp)))

    assert payload["observed_at"] == str(timestamp)


def test_formatter_includes_exception_details() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        record = _make_record()
        record.exc_info = sys.exc_info()

    payload = json.loads(JsonLogFormatter().format(record))

    assert "ValueError: boom" in payload["exception"]


def test_configure_logging_sets_level_and_installs_one_handler() -> None:
    configure_logging("DEBUG")

    root = logging.getLogger()
    assert root.level == logging.DEBUG
    assert len(root.handlers) == 1
    assert isinstance(root.handlers[0].formatter, JsonLogFormatter)


def test_configure_logging_is_idempotent() -> None:
    configure_logging("INFO")
    configure_logging("INFO")

    assert len(logging.getLogger().handlers) == 1


def test_configured_logger_writes_json_lines_to_stdout(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging("INFO")
    logging.getLogger("turbine_guard.test").info("hello", extra={"asset_id": 3})

    payload = json.loads(capsys.readouterr().out.strip())

    assert payload["message"] == "hello"
    assert payload["asset_id"] == 3

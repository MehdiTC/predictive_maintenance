"""Strict validation coverage for versioned online API schemas."""

import math
from datetime import UTC

import pytest
from pydantic import ValidationError

from turbine_guard.api.schemas.online import SensorReadingRequest


def valid_payload() -> dict[str, object]:
    return {
        "external_asset_id": "asset-1",
        "cycle": 1,
        "observed_at": "2026-07-12T12:00:00+04:00",
        "operating_setting_1": 1.0,
        "operating_setting_2": 2.0,
        "operating_setting_3": 3.0,
        **{f"sensor_{index:02d}": float(index) for index in range(1, 22)},
        "source": "unit-test",
        "schema_version": "1",
    }


def test_valid_reading_normalizes_timestamp_and_preserves_anonymous_names() -> None:
    reading = SensorReadingRequest.model_validate(valid_payload())
    assert reading.observed_at is not None
    assert reading.observed_at.tzinfo is UTC
    assert reading.sensor_values == tuple(float(value) for value in range(1, 22))


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_sensor_is_rejected(value: float) -> None:
    payload = valid_payload()
    payload["sensor_01"] = value
    with pytest.raises(ValidationError):
        SensorReadingRequest.model_validate(payload)


def test_non_positive_cycle_is_rejected() -> None:
    payload = valid_payload()
    payload["cycle"] = 0
    with pytest.raises(ValidationError):
        SensorReadingRequest.model_validate(payload)


def test_naive_timestamp_is_rejected() -> None:
    payload = valid_payload()
    payload["observed_at"] = "2026-07-12T12:00:00"
    with pytest.raises(ValidationError, match="timezone"):
        SensorReadingRequest.model_validate(payload)


def test_missing_and_extra_fields_are_rejected() -> None:
    missing = valid_payload()
    missing.pop("sensor_21")
    with pytest.raises(ValidationError):
        SensorReadingRequest.model_validate(missing)
    extra = valid_payload()
    extra["temperature"] = 42
    with pytest.raises(ValidationError, match="Extra inputs"):
        SensorReadingRequest.model_validate(extra)

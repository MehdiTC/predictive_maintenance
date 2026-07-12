"""Tests for the health-check service logic."""

from turbine_guard.services.health import ReadinessResult, check_readiness


def test_readiness_with_no_dependencies_is_ready() -> None:
    result = check_readiness()

    assert result.ready is True
    assert result.checks == {}


def test_readiness_passes_when_all_checks_pass() -> None:
    assert ReadinessResult(checks={"database": True, "model": True}).ready is True


def test_readiness_fails_when_any_check_fails() -> None:
    assert ReadinessResult(checks={"database": True, "model": False}).ready is False

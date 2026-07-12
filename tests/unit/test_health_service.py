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


def test_injected_readiness_checks_report_success_and_failure() -> None:
    result = check_readiness({"database": lambda: True, "model": lambda: False})
    assert result.checks == {"database": True, "model": False}
    assert result.ready is False


def test_readiness_converts_connection_error_to_failure() -> None:
    def unavailable() -> bool:
        raise TimeoutError("connection timed out")

    assert check_readiness({"database": unavailable}).checks == {"database": False}

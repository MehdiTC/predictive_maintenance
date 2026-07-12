"""Health-check logic, kept separate from API route definitions."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ReadinessResult:
    """Outcome of evaluating the service's external dependencies.

    ``checks`` maps a dependency name to whether it is currently available.
    Loop 0 has no external dependencies, so the map is empty and the service
    is always ready; later loops add entries such as the database and the
    champion model.
    """

    checks: dict[str, bool] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        """Whether every dependency check passed."""
        return all(self.checks.values())


def check_readiness(
    dependency_checks: Mapping[str, Callable[[], bool]] | None = None,
) -> ReadinessResult:
    """Evaluate injected dependencies, treating check exceptions as unavailable."""
    results: dict[str, bool] = {}
    for name, check in (dependency_checks or {}).items():
        try:
            results[name] = check()
        except Exception:
            results[name] = False
    return ReadinessResult(checks=results)

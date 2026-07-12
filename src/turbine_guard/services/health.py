"""Health-check logic, kept separate from API route definitions."""

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


def check_readiness() -> ReadinessResult:
    """Evaluate whether the service can currently handle requests."""
    return ReadinessResult(checks={})

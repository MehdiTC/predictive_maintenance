"""Typed replay failures with stable, operator-actionable meanings."""


class ReplayError(RuntimeError):
    """Base class for replay subsystem failures."""


class ReplaySourceError(ReplayError):
    """Replay input is missing, tampered, from the wrong split, or malformed."""


class ReplayStateError(ReplayError):
    """A replay run is absent or in a state that forbids the requested action."""


class ReplayConcurrencyError(ReplayError):
    """Another worker currently holds the advance lease for this run."""


class ReplayIngestionError(ReplayError):
    """The ingestion API rejected a cycle in a way that retries cannot fix."""


class ReplayTransientError(ReplayError):
    """Bounded retries against the ingestion API were exhausted."""


class ReplayOutcomeError(ReplayError):
    """A realized outcome is conflicting or physically impossible."""

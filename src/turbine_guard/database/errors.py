"""Stable persistence errors exposed above the repository boundary."""


class PersistenceError(RuntimeError):
    """Base class for operational persistence failures."""


class ConflictError(PersistenceError):
    """A uniqueness key already exists with different immutable data."""


class DuplicateExternalIdError(ConflictError):
    """An asset or external event identifier already exists."""


class SensorReadingConflictError(ConflictError):
    """An asset cycle already exists with a different sensor payload."""


class PredictionConflictError(ConflictError):
    """A model already predicted for a reading with different values."""


class NotFoundError(PersistenceError):
    """A required referenced operational record does not exist."""

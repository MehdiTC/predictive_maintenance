"""Constrained values persisted by the operational schema."""

from enum import StrEnum


class AssetStatus(StrEnum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    MAINTENANCE = "maintenance"
    RETIRED = "retired"


class RiskLevel(StrEnum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"


class MaintenanceEventType(StrEnum):
    FAILURE = "failure"
    PLANNED_MAINTENANCE = "planned_maintenance"
    INSPECTION = "inspection"
    REPAIR = "repair"


class EvaluationScope(StrEnum):
    REPLAY = "replay"
    ONLINE = "online"
    VALIDATION = "validation"
    BENCHMARK = "benchmark"


class DriftStatus(StrEnum):
    NOT_DETECTED = "not_detected"
    WARNING = "warning"
    DETECTED = "detected"
    INSUFFICIENT_DATA = "insufficient_data"


class PipelineRunType(StrEnum):
    INGESTION = "ingestion"
    MONITORING = "monitoring"
    RETRAINING = "retraining"
    BACKFILL = "backfill"
    PROMOTION = "promotion"


class PipelineRunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ReplayRunStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ReplayMode(StrEnum):
    STEP = "step"
    CONTINUOUS = "continuous"
    ACCELERATED = "accelerated"

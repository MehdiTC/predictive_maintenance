"""SQLAlchemy 2.x typed mappings for PostgreSQL operational state."""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Double,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from turbine_guard.data.schema import OPERATING_SETTING_COLUMNS, SENSOR_COLUMNS
from turbine_guard.database.base import Base, CreatedAtMixin
from turbine_guard.database.enums import (
    AssetStatus,
    DataQualityStatus,
    DriftStatus,
    EvaluationScope,
    LifecycleAssetRole,
    MaintenanceEventType,
    PipelineRunStatus,
    PipelineRunType,
    ReplayMode,
    ReplayRunStatus,
    RiskLevel,
)


def _enum(enum_type: type[Any], name: str) -> Enum:
    """Store string enums with deterministic CHECK constraints, not PostgreSQL enum types."""
    return Enum(
        enum_type,
        name=name,
        native_enum=False,
        create_constraint=True,
        validate_strings=True,
        values_callable=lambda values: [item.value for item in values],
    )


class Asset(CreatedAtMixin, Base):
    """Physical or simulated asset; deliberately not engine-specific."""

    __tablename__ = "assets"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    external_id: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    dataset_name: Mapped[str | None] = mapped_column(String(100))
    dataset_subset: Mapped[str | None] = mapped_column(String(100))
    source_asset_id: Mapped[str | None] = mapped_column(String(255))
    status: Mapped[AssetStatus] = mapped_column(
        _enum(AssetStatus, "asset_status"), nullable=False, default=AssetStatus.ACTIVE
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    sensor_readings: Mapped[list["SensorReading"]] = relationship(back_populates="asset")
    predictions: Mapped[list["Prediction"]] = relationship(back_populates="asset")
    maintenance_events: Mapped[list["MaintenanceEvent"]] = relationship(back_populates="asset")


_sensor_finite_expression = " AND ".join(
    f"{column} NOT IN ('NaN'::double precision, 'Infinity'::double precision, "
    "'-Infinity'::double precision)"
    for column in (*OPERATING_SETTING_COLUMNS, *SENSOR_COLUMNS)
)


class SensorReading(Base):
    """One immutable observed cycle and its anonymous C-MAPSS-shaped measurements."""

    __tablename__ = "sensor_readings"
    __table_args__ = (
        UniqueConstraint("asset_id", "cycle", name="uq_sensor_readings_asset_cycle"),
        CheckConstraint("cycle > 0", name="positive_cycle"),
        CheckConstraint(_sensor_finite_expression, name="finite_measurements"),
        Index("ix_sensor_readings_asset_cycle", "asset_id", "cycle"),
        Index("ix_sensor_readings_ingested_at", "ingested_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    asset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("assets.id", ondelete="RESTRICT"), nullable=False
    )
    cycle: Mapped[int] = mapped_column(Integer, nullable=False)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    operating_setting_1: Mapped[float] = mapped_column(Double, nullable=False)
    operating_setting_2: Mapped[float] = mapped_column(Double, nullable=False)
    operating_setting_3: Mapped[float] = mapped_column(Double, nullable=False)
    sensor_01: Mapped[float] = mapped_column(Double, nullable=False)
    sensor_02: Mapped[float] = mapped_column(Double, nullable=False)
    sensor_03: Mapped[float] = mapped_column(Double, nullable=False)
    sensor_04: Mapped[float] = mapped_column(Double, nullable=False)
    sensor_05: Mapped[float] = mapped_column(Double, nullable=False)
    sensor_06: Mapped[float] = mapped_column(Double, nullable=False)
    sensor_07: Mapped[float] = mapped_column(Double, nullable=False)
    sensor_08: Mapped[float] = mapped_column(Double, nullable=False)
    sensor_09: Mapped[float] = mapped_column(Double, nullable=False)
    sensor_10: Mapped[float] = mapped_column(Double, nullable=False)
    sensor_11: Mapped[float] = mapped_column(Double, nullable=False)
    sensor_12: Mapped[float] = mapped_column(Double, nullable=False)
    sensor_13: Mapped[float] = mapped_column(Double, nullable=False)
    sensor_14: Mapped[float] = mapped_column(Double, nullable=False)
    sensor_15: Mapped[float] = mapped_column(Double, nullable=False)
    sensor_16: Mapped[float] = mapped_column(Double, nullable=False)
    sensor_17: Mapped[float] = mapped_column(Double, nullable=False)
    sensor_18: Mapped[float] = mapped_column(Double, nullable=False)
    sensor_19: Mapped[float] = mapped_column(Double, nullable=False)
    sensor_20: Mapped[float] = mapped_column(Double, nullable=False)
    sensor_21: Mapped[float] = mapped_column(Double, nullable=False)
    schema_version: Mapped[str] = mapped_column(String(50), nullable=False)
    source: Mapped[str] = mapped_column(String(100), nullable=False)
    ingestion_id: Mapped[str | None] = mapped_column(String(255), unique=True)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    asset: Mapped[Asset] = relationship(back_populates="sensor_readings")
    predictions: Mapped[list["Prediction"]] = relationship(back_populates="sensor_reading")


class Prediction(CreatedAtMixin, Base):
    """Version-pinned model output for one persisted sensor reading."""

    __tablename__ = "predictions"
    __table_args__ = (
        UniqueConstraint(
            "sensor_reading_id",
            "model_name",
            "model_version",
            name="uq_predictions_reading_model_version",
        ),
        CheckConstraint("cycle > 0", name="positive_cycle"),
        CheckConstraint("predicted_rul >= 0", name="non_negative_predicted_rul"),
        CheckConstraint("lower_rul IS NULL OR lower_rul >= 0", name="non_negative_lower_rul"),
        CheckConstraint("upper_rul IS NULL OR upper_rul >= 0", name="non_negative_upper_rul"),
        CheckConstraint(
            "(lower_rul IS NULL AND upper_rul IS NULL) OR "
            "(lower_rul <= predicted_rul AND predicted_rul <= upper_rul)",
            name="ordered_interval",
        ),
        CheckConstraint("latency_ms IS NULL OR latency_ms >= 0", name="non_negative_latency"),
        CheckConstraint(
            "predicted_rul NOT IN ('NaN'::double precision, 'Infinity'::double precision, "
            "'-Infinity'::double precision) AND "
            "(lower_rul IS NULL OR lower_rul NOT IN ('NaN'::double precision, "
            "'Infinity'::double precision, '-Infinity'::double precision)) AND "
            "(upper_rul IS NULL OR upper_rul NOT IN ('NaN'::double precision, "
            "'Infinity'::double precision, '-Infinity'::double precision)) AND "
            "(latency_ms IS NULL OR latency_ms NOT IN ('NaN'::double precision, "
            "'Infinity'::double precision, '-Infinity'::double precision))",
            name="finite_values",
        ),
        Index("ix_predictions_asset_timestamp", "asset_id", "prediction_timestamp"),
        Index("ix_predictions_timestamp", "prediction_timestamp"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    asset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("assets.id", ondelete="RESTRICT"), nullable=False
    )
    sensor_reading_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("sensor_readings.id", ondelete="RESTRICT"),
        nullable=False,
    )
    cycle: Mapped[int] = mapped_column(Integer, nullable=False)
    predicted_rul: Mapped[float] = mapped_column(Double, nullable=False)
    lower_rul: Mapped[float | None] = mapped_column(Double)
    upper_rul: Mapped[float | None] = mapped_column(Double)
    risk_level: Mapped[RiskLevel] = mapped_column(_enum(RiskLevel, "risk_level"), nullable=False)
    failure_within_30: Mapped[bool | None] = mapped_column(Boolean)
    failure_within_50: Mapped[bool | None] = mapped_column(Boolean)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    model_version: Mapped[str] = mapped_column(String(100), nullable=False)
    model_alias: Mapped[str | None] = mapped_column(String(100))
    model_run_id: Mapped[str | None] = mapped_column(String(255))
    feature_version: Mapped[str] = mapped_column(String(100), nullable=False)
    prediction_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    latency_ms: Mapped[float | None] = mapped_column(Double)
    request_id: Mapped[str | None] = mapped_column(String(255), unique=True)

    asset: Mapped[Asset] = relationship(back_populates="predictions")
    sensor_reading: Mapped[SensorReading] = relationship(back_populates="predictions")


class MaintenanceEvent(CreatedAtMixin, Base):
    """Delayed failure, inspection, repair, or planned-maintenance outcome."""

    __tablename__ = "maintenance_events"
    __table_args__ = (
        CheckConstraint("event_cycle IS NULL OR event_cycle > 0", name="positive_event_cycle"),
        Index("ix_maintenance_events_asset_occurred_at", "asset_id", "occurred_at"),
        Index("ix_maintenance_events_occurred_at", "occurred_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    asset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("assets.id", ondelete="RESTRICT"), nullable=False
    )
    event_type: Mapped[MaintenanceEventType] = mapped_column(
        _enum(MaintenanceEventType, "maintenance_event_type"), nullable=False
    )
    event_cycle: Mapped[int | None] = mapped_column(Integer)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    source: Mapped[str] = mapped_column(String(100), nullable=False)
    external_event_id: Mapped[str | None] = mapped_column(String(255), unique=True)
    description: Mapped[str | None] = mapped_column(Text)
    event_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict, server_default="{}"
    )

    asset: Mapped[Asset] = relationship(back_populates="maintenance_events")


class ModelEvaluation(CreatedAtMixin, Base):
    """Delayed or batch model-quality summary, not an MLflow experiment copy."""

    __tablename__ = "model_evaluations"
    __table_args__ = (
        CheckConstraint("window_start <= window_end", name="ordered_window"),
        CheckConstraint("sample_count >= 0", name="non_negative_sample_count"),
        Index(
            "ix_model_evaluations_model_version_created",
            "model_name",
            "model_version",
            "created_at",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    model_version: Mapped[str] = mapped_column(String(100), nullable=False)
    evaluation_scope: Mapped[EvaluationScope] = mapped_column(
        _enum(EvaluationScope, "evaluation_scope"), nullable=False
    )
    dataset_subset: Mapped[str | None] = mapped_column(String(100))
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    window_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    sample_count: Mapped[int] = mapped_column(BigInteger, nullable=False)
    mae: Mapped[float | None] = mapped_column(Double)
    rmse: Mapped[float | None] = mapped_column(Double)
    nasa_score: Mapped[float | None] = mapped_column(Double)
    critical_precision: Mapped[float | None] = mapped_column(Double)
    critical_recall: Mapped[float | None] = mapped_column(Double)
    interval_coverage: Mapped[float | None] = mapped_column(Double)
    metrics: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default="{}"
    )


class DriftReport(CreatedAtMixin, Base):
    """Persisted drift summary produced by future Loop 9 calculations."""

    __tablename__ = "drift_reports"
    __table_args__ = (
        CheckConstraint("window_start <= window_end", name="ordered_window"),
        CheckConstraint("max_psi IS NULL OR max_psi >= 0", name="non_negative_max_psi"),
        CheckConstraint(
            "max_wasserstein IS NULL OR max_wasserstein >= 0",
            name="non_negative_max_wasserstein",
        ),
        CheckConstraint("drifted_feature_count >= 0", name="non_negative_feature_count"),
        Index("ix_drift_reports_model_window", "model_name", "model_version", "window_end"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    model_version: Mapped[str] = mapped_column(String(100), nullable=False)
    feature_version: Mapped[str] = mapped_column(String(100), nullable=False)
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    window_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    status: Mapped[DriftStatus] = mapped_column(_enum(DriftStatus, "drift_status"), nullable=False)
    max_psi: Mapped[float | None] = mapped_column(Double)
    max_wasserstein: Mapped[float | None] = mapped_column(Double)
    drifted_feature_count: Mapped[int] = mapped_column(Integer, nullable=False)
    details: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default="{}"
    )


class DataQualityReport(CreatedAtMixin, Base):
    """Persisted quality summary for one immutable monitoring window."""

    __tablename__ = "data_quality_reports"
    __table_args__ = (
        CheckConstraint("window_start <= window_end", name="ordered_window"),
        CheckConstraint("record_count >= 0", name="non_negative_record_count"),
        CheckConstraint("asset_count >= 0", name="non_negative_asset_count"),
        CheckConstraint("failure_count >= 0", name="non_negative_failure_count"),
        Index("ix_data_quality_reports_model_window", "model_name", "model_version", "window_end"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    pipeline_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("pipeline_runs.id", ondelete="RESTRICT"), nullable=False
    )
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    model_version: Mapped[str] = mapped_column(String(100), nullable=False)
    feature_version: Mapped[str] = mapped_column(String(100), nullable=False)
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    window_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    status: Mapped[DataQualityStatus] = mapped_column(
        _enum(DataQualityStatus, "data_quality_status"), nullable=False
    )
    record_count: Mapped[int] = mapped_column(BigInteger, nullable=False)
    asset_count: Mapped[int] = mapped_column(Integer, nullable=False)
    failure_count: Mapped[int] = mapped_column(Integer, nullable=False)
    details: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default="{}"
    )


class ReplayRun(CreatedAtMixin, Base):
    """Durable progress and phase state for one held-out trajectory replay.

    The final source cycle is replay-internal ground truth. It lives only in
    this table, which no prediction endpoint reads, so the online inference
    path structurally cannot observe an asset's future.
    """

    __tablename__ = "replay_runs"
    __table_args__ = (
        UniqueConstraint(
            "dataset_name",
            "dataset_subset",
            "source_asset_id",
            "attempt",
            name="uq_replay_runs_source_attempt",
        ),
        CheckConstraint("source_asset_id > 0", name="positive_source_asset"),
        CheckConstraint("attempt >= 1", name="positive_attempt"),
        CheckConstraint("final_cycle > 0", name="positive_final_cycle"),
        CheckConstraint(
            "last_confirmed_cycle >= 0 AND last_confirmed_cycle <= final_cycle",
            name="confirmed_cycle_in_range",
        ),
        CheckConstraint("cycle_delay_seconds >= 0", name="non_negative_delay"),
        CheckConstraint("simulated_cycle_duration_seconds > 0", name="positive_cycle_duration"),
        CheckConstraint(
            "status != 'completed' OR completed_at IS NOT NULL",
            name="completed_run_has_finish",
        ),
        CheckConstraint(
            "status != 'failed' OR error_message IS NOT NULL",
            name="failed_run_has_error",
        ),
        CheckConstraint(
            "(lease_token IS NULL) = (lease_expires_at IS NULL)",
            name="lease_fields_paired",
        ),
        Index("ix_replay_runs_status", "status"),
        Index("ix_replay_runs_source_asset", "dataset_subset", "source_asset_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_name: Mapped[str] = mapped_column(String(100), nullable=False)
    dataset_subset: Mapped[str] = mapped_column(String(100), nullable=False)
    source_asset_id: Mapped[int] = mapped_column(Integer, nullable=False)
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    external_asset_id: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    asset_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("assets.id", ondelete="RESTRICT")
    )
    final_cycle: Mapped[int] = mapped_column(Integer, nullable=False)
    last_confirmed_cycle: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[ReplayRunStatus] = mapped_column(
        _enum(ReplayRunStatus, "replay_run_status"), nullable=False
    )
    mode: Mapped[ReplayMode] = mapped_column(_enum(ReplayMode, "replay_mode"), nullable=False)
    cycle_delay_seconds: Mapped[float] = mapped_column(Double, nullable=False)
    simulated_cycle_duration_seconds: Mapped[float] = mapped_column(Double, nullable=False)
    replay_started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_advanced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    lease_token: Mapped[str | None] = mapped_column(String(64))
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    failure_event_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("maintenance_events.id", ondelete="RESTRICT")
    )
    labels_backfilled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    evaluation_completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_message: Mapped[str | None] = mapped_column(Text)
    run_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict, server_default="{}"
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    asset: Mapped[Asset | None] = relationship()
    failure_event: Mapped[MaintenanceEvent | None] = relationship()


class PredictionOutcome(CreatedAtMixin, Base):
    """Realized delayed label linking one immutable prediction to one outcome event.

    Predictions are never mutated; the realized RUL lives in this separate
    table so one prediction can be evaluated against multiple outcomes or
    re-evaluations while the original model output stays intact.
    """

    __tablename__ = "prediction_outcomes"
    __table_args__ = (
        UniqueConstraint(
            "prediction_id",
            "maintenance_event_id",
            name="uq_prediction_outcomes_prediction_event",
        ),
        CheckConstraint("cycle > 0", name="positive_cycle"),
        CheckConstraint("realized_rul >= 0", name="non_negative_realized_rul"),
        Index("ix_prediction_outcomes_asset", "asset_id"),
        Index("ix_prediction_outcomes_event", "maintenance_event_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    prediction_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("predictions.id", ondelete="RESTRICT"), nullable=False
    )
    maintenance_event_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("maintenance_events.id", ondelete="RESTRICT"),
        nullable=False,
    )
    asset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("assets.id", ondelete="RESTRICT"), nullable=False
    )
    cycle: Mapped[int] = mapped_column(Integer, nullable=False)
    realized_rul: Mapped[int] = mapped_column(Integer, nullable=False)
    labeled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    prediction: Mapped[Prediction] = relationship()
    maintenance_event: Mapped[MaintenanceEvent] = relationship()


class PipelineRun(CreatedAtMixin, Base):
    """Auditable operational workflow lifecycle without an orchestration dependency."""

    __tablename__ = "pipeline_runs"
    __table_args__ = (
        CheckConstraint(
            "finished_at IS NULL OR finished_at >= started_at", name="valid_finish_time"
        ),
        CheckConstraint(
            "status NOT IN ('succeeded', 'failed', 'cancelled') OR finished_at IS NOT NULL",
            name="terminal_run_has_finish",
        ),
        CheckConstraint(
            "status != 'failed' OR error_message IS NOT NULL", name="failed_run_has_error"
        ),
        Index("ix_pipeline_runs_status_started", "status", "started_at"),
        Index("ix_pipeline_runs_type_started", "run_type", "started_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_type: Mapped[PipelineRunType] = mapped_column(
        _enum(PipelineRunType, "pipeline_run_type"), nullable=False
    )
    status: Mapped[PipelineRunStatus] = mapped_column(
        _enum(PipelineRunStatus, "pipeline_run_status"), nullable=False
    )
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    trigger: Mapped[str] = mapped_column(String(100), nullable=False)
    idempotency_key: Mapped[str | None] = mapped_column(String(64), unique=True)
    phase: Mapped[str | None] = mapped_column(String(100))
    git_sha: Mapped[str | None] = mapped_column(String(64))
    model_version: Mapped[str | None] = mapped_column(String(100))
    input_manifest_checksum: Mapped[str | None] = mapped_column(String(64))
    output_manifest_checksum: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)
    run_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict, server_default="{}"
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class LifecycleAssetAssignment(CreatedAtMixin, Base):
    """Leakage-auditable asset role within one retraining lifecycle."""

    __tablename__ = "lifecycle_asset_assignments"
    __table_args__ = (
        UniqueConstraint("pipeline_run_id", "asset_id", name="uq_lifecycle_run_asset"),
        Index("ix_lifecycle_assets_run_role", "pipeline_run_id", "role"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    pipeline_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("pipeline_runs.id", ondelete="RESTRICT"), nullable=False
    )
    asset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("assets.id", ondelete="RESTRICT"), nullable=False
    )
    role: Mapped[LifecycleAssetRole] = mapped_column(
        _enum(LifecycleAssetRole, "lifecycle_asset_role"), nullable=False
    )
    source_asset_id: Mapped[str | None] = mapped_column(String(255))
    row_count: Mapped[int] = mapped_column(Integer, nullable=False)


class LifecycleEvent(CreatedAtMixin, Base):
    """Append-only audit event for candidate, alias, approval, and rollback changes."""

    __tablename__ = "lifecycle_events"
    __table_args__ = (
        Index("ix_lifecycle_events_run_created", "pipeline_run_id", "created_at"),
        Index("ix_lifecycle_events_model_created", "model_name", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    pipeline_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("pipeline_runs.id", ondelete="RESTRICT")
    )
    event_key: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    event_type: Mapped[str] = mapped_column(String(100), nullable=False)
    phase: Mapped[str] = mapped_column(String(100), nullable=False)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    from_version: Mapped[str | None] = mapped_column(String(100))
    to_version: Mapped[str | None] = mapped_column(String(100))
    actor: Mapped[str] = mapped_column(String(255), nullable=False)
    details: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default="{}"
    )

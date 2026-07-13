"""Validated persistence commands kept separate from ORM and future API schemas."""

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from turbine_guard.database.enums import (
    AssetStatus,
    DriftStatus,
    EvaluationScope,
    MaintenanceEventType,
    PipelineRunStatus,
    PipelineRunType,
    ReplayMode,
    ReplayRunStatus,
    RiskLevel,
)


def _required(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty.")


def _aware(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware.")


def _finite(values: tuple[float, ...], field_name: str) -> None:
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{field_name} must contain only finite numbers.")


def _enum_member(value: object, enum_type: type[object], field_name: str) -> None:
    if not isinstance(value, enum_type):
        raise ValueError(f"{field_name} must be a valid constrained value.")


@dataclass(frozen=True)
class NewAsset:
    external_id: str
    dataset_name: str | None = None
    dataset_subset: str | None = None
    source_asset_id: str | None = None
    status: AssetStatus = AssetStatus.ACTIVE

    def __post_init__(self) -> None:
        _required(self.external_id, "external_id")
        _enum_member(self.status, AssetStatus, "status")


@dataclass(frozen=True)
class NewSensorReading:
    asset_id: uuid.UUID
    cycle: int
    observed_at: datetime
    operating_settings: tuple[float, float, float]
    sensor_values: tuple[
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
    ]
    schema_version: str
    source: str
    ingestion_id: str | None = None

    def __post_init__(self) -> None:
        if self.cycle <= 0:
            raise ValueError("cycle must be positive.")
        _aware(self.observed_at, "observed_at")
        _finite(self.operating_settings, "operating_settings")
        _finite(self.sensor_values, "sensor_values")
        _required(self.schema_version, "schema_version")
        _required(self.source, "source")


@dataclass(frozen=True)
class NewPrediction:
    asset_id: uuid.UUID
    sensor_reading_id: uuid.UUID
    cycle: int
    predicted_rul: float
    risk_level: RiskLevel
    model_name: str
    model_version: str
    feature_version: str
    prediction_timestamp: datetime
    lower_rul: float | None = None
    upper_rul: float | None = None
    failure_within_30: bool | None = None
    failure_within_50: bool | None = None
    model_alias: str | None = None
    model_run_id: str | None = None
    latency_ms: float | None = None
    request_id: str | None = None

    def __post_init__(self) -> None:
        if self.cycle <= 0:
            raise ValueError("cycle must be positive.")
        numeric = tuple(
            value
            for value in (self.predicted_rul, self.lower_rul, self.upper_rul, self.latency_ms)
            if value is not None
        )
        _finite(numeric, "prediction values")
        if self.predicted_rul < 0 or (self.latency_ms is not None and self.latency_ms < 0):
            raise ValueError("predicted_rul and latency_ms must be non-negative.")
        if (self.lower_rul is None) != (self.upper_rul is None):
            raise ValueError("Prediction interval bounds must both be present or absent.")
        if self.lower_rul is not None and not (
            0 <= self.lower_rul <= self.predicted_rul <= self.upper_rul  # type: ignore[operator]
        ):
            raise ValueError("Prediction interval must satisfy 0 <= lower <= prediction <= upper.")
        for value, name in (
            (self.model_name, "model_name"),
            (self.model_version, "model_version"),
            (self.feature_version, "feature_version"),
        ):
            _required(value, name)
        _aware(self.prediction_timestamp, "prediction_timestamp")
        _enum_member(self.risk_level, RiskLevel, "risk_level")


@dataclass(frozen=True)
class NewMaintenanceEvent:
    asset_id: uuid.UUID
    event_type: MaintenanceEventType
    occurred_at: datetime
    source: str
    event_cycle: int | None = None
    external_event_id: str | None = None
    description: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.event_cycle is not None and self.event_cycle <= 0:
            raise ValueError("event_cycle must be positive when provided.")
        _aware(self.occurred_at, "occurred_at")
        _required(self.source, "source")
        _enum_member(self.event_type, MaintenanceEventType, "event_type")


@dataclass(frozen=True)
class NewModelEvaluation:
    model_name: str
    model_version: str
    evaluation_scope: EvaluationScope
    window_start: datetime
    window_end: datetime
    sample_count: int
    dataset_subset: str | None = None
    mae: float | None = None
    rmse: float | None = None
    nasa_score: float | None = None
    critical_precision: float | None = None
    critical_recall: float | None = None
    interval_coverage: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _required(self.model_name, "model_name")
        _required(self.model_version, "model_version")
        _aware(self.window_start, "window_start")
        _aware(self.window_end, "window_end")
        if self.window_start > self.window_end or self.sample_count < 0:
            raise ValueError("Evaluation window must be ordered and sample_count non-negative.")
        _finite(
            tuple(
                value
                for value in (
                    self.mae,
                    self.rmse,
                    self.nasa_score,
                    self.critical_precision,
                    self.critical_recall,
                    self.interval_coverage,
                )
                if value is not None
            ),
            "evaluation metrics",
        )
        _enum_member(self.evaluation_scope, EvaluationScope, "evaluation_scope")


@dataclass(frozen=True)
class NewDriftReport:
    model_name: str
    model_version: str
    feature_version: str
    window_start: datetime
    window_end: datetime
    status: DriftStatus
    drifted_feature_count: int
    max_psi: float | None = None
    max_wasserstein: float | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for value, name in (
            (self.model_name, "model_name"),
            (self.model_version, "model_version"),
            (self.feature_version, "feature_version"),
        ):
            _required(value, name)
        _aware(self.window_start, "window_start")
        _aware(self.window_end, "window_end")
        if self.window_start > self.window_end or self.drifted_feature_count < 0:
            raise ValueError("Drift window must be ordered and feature count non-negative.")
        values = tuple(value for value in (self.max_psi, self.max_wasserstein) if value is not None)
        _finite(values, "drift metrics")
        if any(value < 0 for value in values):
            raise ValueError("Drift distances must be non-negative.")
        _enum_member(self.status, DriftStatus, "status")


@dataclass(frozen=True)
class NewReplayRun:
    dataset_name: str
    dataset_subset: str
    source_asset_id: int
    external_asset_id: str
    final_cycle: int
    mode: ReplayMode
    cycle_delay_seconds: float
    simulated_cycle_duration_seconds: float
    replay_started_at: datetime
    attempt: int = 1
    status: ReplayRunStatus = ReplayRunStatus.CREATED
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _required(self.dataset_name, "dataset_name")
        _required(self.dataset_subset, "dataset_subset")
        _required(self.external_asset_id, "external_asset_id")
        if self.source_asset_id <= 0 or self.final_cycle <= 0 or self.attempt < 1:
            raise ValueError("source_asset_id, final_cycle, and attempt must be positive.")
        _finite(
            (self.cycle_delay_seconds, self.simulated_cycle_duration_seconds),
            "replay timing values",
        )
        if self.cycle_delay_seconds < 0 or self.simulated_cycle_duration_seconds <= 0:
            raise ValueError(
                "cycle_delay_seconds must be non-negative and "
                "simulated_cycle_duration_seconds positive."
            )
        _aware(self.replay_started_at, "replay_started_at")
        _enum_member(self.mode, ReplayMode, "mode")
        _enum_member(self.status, ReplayRunStatus, "status")


@dataclass(frozen=True)
class NewPredictionOutcome:
    prediction_id: uuid.UUID
    maintenance_event_id: uuid.UUID
    asset_id: uuid.UUID
    cycle: int
    realized_rul: int
    labeled_at: datetime

    def __post_init__(self) -> None:
        if self.cycle <= 0:
            raise ValueError("cycle must be positive.")
        if self.realized_rul < 0:
            raise ValueError("realized_rul must be non-negative.")
        _aware(self.labeled_at, "labeled_at")


@dataclass(frozen=True)
class NewPipelineRun:
    run_type: PipelineRunType
    status: PipelineRunStatus
    started_at: datetime
    trigger: str
    finished_at: datetime | None = None
    git_sha: str | None = None
    model_version: str | None = None
    input_manifest_checksum: str | None = None
    output_manifest_checksum: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _aware(self.started_at, "started_at")
        _required(self.trigger, "trigger")
        terminal = {
            PipelineRunStatus.SUCCEEDED,
            PipelineRunStatus.FAILED,
            PipelineRunStatus.CANCELLED,
        }
        if self.finished_at is not None:
            _aware(self.finished_at, "finished_at")
            if self.finished_at < self.started_at:
                raise ValueError("finished_at must not precede started_at.")
        if self.status in terminal and self.finished_at is None:
            raise ValueError("Terminal pipeline runs require finished_at.")
        if self.status is PipelineRunStatus.FAILED and not self.error_message:
            raise ValueError("Failed pipeline runs require error_message.")
        _enum_member(self.run_type, PipelineRunType, "run_type")
        _enum_member(self.status, PipelineRunStatus, "status")

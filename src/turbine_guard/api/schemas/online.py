"""Versioned HTTP contracts for Loop 7 online inference and asset health."""

import uuid
from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_serializer, field_validator

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]


class ApiModel(BaseModel):
    """Strict stable API base; unknown client fields are rejected."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    @field_serializer("*", check_fields=False)
    def _serialize_datetimes_as_utc(self, value: Any) -> Any:
        """Keep every external timestamp in the documented UTC representation."""
        if isinstance(value, datetime):
            return value.astimezone(UTC)
        return value


class SensorReadingRequest(ApiModel):
    """One anonymous 21-channel operating-cycle observation."""

    external_asset_id: str = Field(min_length=1, max_length=255)
    cycle: PositiveInt
    observed_at: datetime | None = Field(
        default=None, description="Timezone-aware observation time; defaults to receipt time."
    )
    operating_setting_1: FiniteFloat
    operating_setting_2: FiniteFloat
    operating_setting_3: FiniteFloat
    sensor_01: FiniteFloat
    sensor_02: FiniteFloat
    sensor_03: FiniteFloat
    sensor_04: FiniteFloat
    sensor_05: FiniteFloat
    sensor_06: FiniteFloat
    sensor_07: FiniteFloat
    sensor_08: FiniteFloat
    sensor_09: FiniteFloat
    sensor_10: FiniteFloat
    sensor_11: FiniteFloat
    sensor_12: FiniteFloat
    sensor_13: FiniteFloat
    sensor_14: FiniteFloat
    sensor_15: FiniteFloat
    sensor_16: FiniteFloat
    sensor_17: FiniteFloat
    sensor_18: FiniteFloat
    sensor_19: FiniteFloat
    sensor_20: FiniteFloat
    sensor_21: FiniteFloat
    source: str = Field(min_length=1, max_length=100)
    ingestion_id: str | None = Field(default=None, min_length=1, max_length=255)
    schema_version: str = Field(default="1", min_length=1, max_length=50)

    @field_validator("observed_at")
    @classmethod
    def _utc_timestamp(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("observed_at must include a timezone offset.")
        return value.astimezone(UTC)

    @field_validator("external_asset_id", "source", "schema_version", "ingestion_id")
    @classmethod
    def _non_blank_string(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("String identifiers must not be blank.")
        return normalized

    @property
    def operating_settings(self) -> tuple[float, float, float]:
        return (
            self.operating_setting_1,
            self.operating_setting_2,
            self.operating_setting_3,
        )

    @property
    def sensor_values(self) -> tuple[float, ...]:
        return tuple(getattr(self, f"sensor_{index:02d}") for index in range(1, 22))


class PredictionResponse(ApiModel):
    predicted_rul: float
    lower_rul: float
    upper_rul: float
    risk_level: Literal["healthy", "warning", "critical"]
    failure_within_30: bool
    failure_within_50: bool
    model_name: str
    model_version: str
    model_alias: str
    model_run_id: str | None
    feature_version: str
    prediction_timestamp: datetime
    latency_ms: float | None


class SensorIngestionResponse(ApiModel):
    request_id: str
    asset_id: uuid.UUID
    external_asset_id: str
    cycle: int
    reading_id: uuid.UUID
    prediction: PredictionResponse
    idempotent: bool
    reading_idempotent: bool
    prediction_idempotent: bool
    data_quality_warnings: tuple[str, ...] = ()


class AssetSummaryResponse(ApiModel):
    asset_id: uuid.UUID
    external_asset_id: str
    status: str
    latest_cycle: int | None
    latest_risk_level: str | None
    latest_predicted_rul: float | None
    last_observed_at: datetime | None


class AssetListResponse(ApiModel):
    items: list[AssetSummaryResponse]
    limit: int
    offset: int


class ReadingSummaryResponse(ApiModel):
    reading_id: uuid.UUID
    cycle: int
    observed_at: datetime
    source: str
    schema_version: str


class MaintenanceEventSummaryResponse(ApiModel):
    event_id: uuid.UUID
    event_type: str
    event_cycle: int | None
    occurred_at: datetime
    source: str
    description: str | None


class AssetDetailResponse(ApiModel):
    asset_id: uuid.UUID
    external_asset_id: str
    dataset_name: str | None
    dataset_subset: str | None
    source_asset_id: str | None
    status: str
    created_at: datetime
    updated_at: datetime
    latest_reading: ReadingSummaryResponse | None
    latest_prediction: PredictionResponse | None
    recent_maintenance_events: list[MaintenanceEventSummaryResponse]


class PredictionTrendPoint(ApiModel):
    cycle: int
    predicted_rul: float
    risk_level: str
    prediction_timestamp: datetime


class AssetHealthResponse(ApiModel):
    asset_id: uuid.UUID
    external_asset_id: str
    latest_cycle: int | None
    predicted_rul: float | None
    lower_rul: float | None
    upper_rul: float | None
    risk_level: str | None
    failure_within_30: bool | None
    failure_within_50: bool | None
    prediction_trend: list[PredictionTrendPoint]
    latest_observation_at: datetime | None
    model_version: str | None
    stale: bool
    data_quality_status: Literal["valid", "no_data"]


class RecentPredictionItem(ApiModel):
    prediction_id: uuid.UUID
    asset_id: uuid.UUID
    external_asset_id: str
    reading_id: uuid.UUID
    cycle: int
    prediction: PredictionResponse


class RecentPredictionsResponse(ApiModel):
    items: list[RecentPredictionItem]
    limit: int


class CurrentModelResponse(ApiModel):
    model_name: str
    registry_version: str
    alias: str
    source_run_id: str
    target_definition: str
    rul_cap: int | None
    feature_count: int
    feature_version: str
    validation_rmse: float | None
    replay_rmse: float | None
    official_test_rmse: float | None
    conformal_coverage_target: float | None
    model_load_timestamp: datetime
    model_checksum: str | None
    lineage_id: str | None
    model_family: str | None = None
    git_sha: str | None = None
    dataset_checksum: str | None = None
    feature_manifest_checksum: str | None = None


class MonitoringSummaryResponse(ApiModel):
    request_count: int
    prediction_count: int
    validation_failures: int
    database_failures: int
    model_load_failures: int
    prediction_failures: int
    conflict_count: int
    average_prediction_latency_ms: float | None
    current_model_version: str | None
    recent_risk_distribution: dict[str, int]
    reading_count: int
    stored_prediction_count: int
    latest_ingestion_time: datetime | None


class ErrorDetail(ApiModel):
    code: str
    message: str
    request_id: str
    details: list[dict[str, Any]] | None = None


class ErrorResponse(ApiModel):
    error: ErrorDetail

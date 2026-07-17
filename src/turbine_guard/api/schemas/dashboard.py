"""Typed read and constrained-control contracts for the Loop 11 dashboard."""

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from turbine_guard.api.schemas.online import ApiModel, MaintenanceEventSummaryResponse

RiskFilter = Literal["healthy", "warning", "critical"]


class FleetAssetItem(ApiModel):
    asset_id: uuid.UUID
    external_asset_id: str
    asset_status: str
    latest_cycle: int | None
    predicted_rul: float | None
    lower_rul: float | None
    upper_rul: float | None
    risk_level: str | None
    latest_observation_at: datetime | None
    prediction_timestamp: datetime | None
    model_version: str | None
    feature_version: str | None
    stale: bool


class FleetOverviewResponse(ApiModel):
    total_assets: int
    active_assets: int
    latest_observation_at: datetime | None
    healthy_count: int
    warning_count: int
    critical_count: int
    assets_without_recent_predictions: int
    current_model_version: str | None
    drift_status: str
    performance_status: str
    replay_status: str
    items: list[FleetAssetItem]
    limit: int
    offset: int


class PredictionHistoryItem(ApiModel):
    prediction_id: uuid.UUID
    asset_id: uuid.UUID
    external_asset_id: str
    cycle: int
    predicted_rul: float
    lower_rul: float
    upper_rul: float
    risk_level: str
    model_name: str
    model_version: str
    feature_version: str
    prediction_timestamp: datetime
    latency_ms: float | None


class PredictionHistoryResponse(ApiModel):
    items: list[PredictionHistoryItem]
    limit: int
    offset: int


class AlertAssetItem(ApiModel):
    asset_id: uuid.UUID
    external_asset_id: str
    current_risk_level: Literal["warning", "critical"]
    first_warning_cycle: int | None
    first_critical_cycle: int | None
    latest_predicted_rul: float
    latest_prediction_at: datetime
    alert_age_seconds: float
    model_version: str
    outcome: str | None


class AlertSummaryResponse(ApiModel):
    warning_count: int
    critical_count: int
    items: list[AlertAssetItem]
    limit: int


class SensorHistoryPoint(ApiModel):
    cycle: int
    observed_at: datetime
    values: dict[str, float]


class ReplayRunResponse(ApiModel):
    run_id: uuid.UUID
    source_asset_id: int
    attempt: int
    external_asset_id: str
    operational_asset_id: uuid.UUID | None
    status: str
    mode: str
    last_confirmed_cycle: int
    final_cycle: int | None = Field(
        default=None,
        description="Only disclosed after completion; hidden replay ground truth otherwise.",
    )
    progress_percent: float
    started_at: datetime
    last_advanced_at: datetime | None
    completed_at: datetime | None
    error_message: str | None


class ReplayStatusResponse(ApiModel):
    enabled: bool
    writable: bool
    public_demo_mode: bool
    allowed_source_asset_ids: list[int]
    restrictions: list[str]
    runs: list[ReplayRunResponse]


class AssetDashboardResponse(ApiModel):
    asset_id: uuid.UUID
    external_asset_id: str
    asset_status: str
    dataset_name: str | None
    dataset_subset: str | None
    source_asset_id: str | None
    latest_cycle: int | None
    predicted_rul: float | None
    lower_rul: float | None
    upper_rul: float | None
    risk_level: str | None
    failure_within_30: bool | None
    failure_within_50: bool | None
    model_version: str | None
    feature_version: str | None
    latest_observation_at: datetime | None
    stale: bool
    data_quality_warnings: list[str]
    predictions: list[PredictionHistoryItem]
    sensor_columns: list[str]
    available_sensor_columns: list[str]
    sensor_history: list[SensorHistoryPoint]
    maintenance_events: list[MaintenanceEventSummaryResponse]
    replay: ReplayRunResponse | None


class DriftFeatureItem(ApiModel):
    feature: str
    psi: float | None
    wasserstein: float | None
    normalized_wasserstein: float | None
    missingness_shift: float | None
    drifted: bool
    warning: bool


class DriftDetailResponse(ApiModel):
    status: str
    available: bool
    model_name: str | None
    model_version: str | None
    feature_version: str | None
    window_start: datetime | None
    window_end: datetime | None
    drifted_feature_count: int
    max_psi: float | None
    max_wasserstein: float | None
    report_timestamp: datetime | None
    top_features: list[DriftFeatureItem]
    note: str


class PerformanceDetailResponse(ApiModel):
    status: str
    available: bool
    target_label: str
    model_name: str | None
    model_version: str | None
    window_start: datetime | None
    window_end: datetime | None
    labeled_rows: int
    completed_assets: int | None
    mae: float | None
    rmse: float | None
    nasa_score: float | None
    critical_precision: float | None
    critical_recall: float | None
    critical_f1: float | None
    false_alarms_per_1000_cycles: float | None
    mean_alert_lead_time: float | None
    timely_alert_rate: float | None
    interval_coverage: float | None
    average_interval_width: float | None
    report_timestamp: datetime | None


class LifecycleItem(ApiModel):
    run_id: uuid.UUID
    run_type: str
    status: str
    phase: str | None
    model_version: str | None
    started_at: datetime
    finished_at: datetime | None
    decision: str | None
    candidate_version: str | None


class ModelOverviewResponse(ApiModel):
    available: bool
    registry_source: str | None
    registered_model_name: str | None
    registry_version: str | None
    alias: str | None
    aliases: dict[str, str]
    model_family: str | None
    target_definition: str | None
    rul_cap: int | None
    feature_count: int | None
    feature_version: str | None
    validation_rmse: float | None
    replay_rmse: float | None
    official_benchmark_rmse: float | None
    conformal_coverage_target: float | None
    source_run_id: str | None
    model_load_timestamp: datetime | None
    git_sha: str | None
    manifest_lineage: dict[str, str]
    latest_lifecycle: list[LifecycleItem]
    latest_event: dict[str, Any] | None


class DemoPredictionPoint(ApiModel):
    cycle: int
    predicted_rul: float
    lower_rul: float | None
    upper_rul: float | None
    risk_level: str


class DemoStateResponse(ApiModel):
    """Everything the guided landing-page demo needs in one bounded payload."""

    enabled: bool
    demo_source_asset_id: int
    run: ReplayRunResponse | None
    series: list[DemoPredictionPoint]
    model_version: str | None
    max_attempts: int
    max_cycles_per_request: int
    cooldown_seconds: float


class ReplayActionRequest(ApiModel):
    action: Literal["start", "step", "pause", "resume", "accelerate", "reset"]
    source_asset_id: int | None = Field(default=None, ge=1)
    run_id: uuid.UUID | None = None
    max_cycles: int | None = Field(default=None, ge=1, le=100)
    confirm_reset: bool = False
    control_token: str | None = Field(default=None, max_length=512)


class ReplayActionResponse(ApiModel):
    action: str
    run: ReplayRunResponse
    message: str

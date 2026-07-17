"""Bounded read-side projections shared by dashboard HTML and JSON routes."""

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from sqlalchemy import case, desc, func, select
from sqlalchemy.orm import Session, sessionmaker

from turbine_guard.api.schemas.dashboard import (
    AlertAssetItem,
    AlertSummaryResponse,
    AssetDashboardResponse,
    DriftDetailResponse,
    DriftFeatureItem,
    FleetAssetItem,
    FleetOverviewResponse,
    LifecycleItem,
    ModelOverviewResponse,
    PerformanceDetailResponse,
    PredictionHistoryItem,
    PredictionHistoryResponse,
    ReplayRunResponse,
    SensorHistoryPoint,
)
from turbine_guard.api.schemas.online import MaintenanceEventSummaryResponse
from turbine_guard.config.settings import Settings
from turbine_guard.data.schema import SENSOR_COLUMNS
from turbine_guard.database.enums import AssetStatus, EvaluationScope, ReplayRunStatus, RiskLevel
from turbine_guard.database.models import (
    Asset,
    DataQualityReport,
    DriftReport,
    LifecycleEvent,
    MaintenanceEvent,
    ModelEvaluation,
    PipelineRun,
    Prediction,
    ReplayRun,
    SensorReading,
)
from turbine_guard.database.session import session_scope
from turbine_guard.services.errors import AssetNotFoundError, RequestParameterError
from turbine_guard.serving.champion import ChampionLoader


class DashboardService:
    """Create stable public projections without exposing ORM rows or internal paths."""

    def __init__(
        self,
        sessions: sessionmaker[Session],
        model_loader: ChampionLoader,
        settings: Settings,
    ) -> None:
        self._sessions = sessions
        self._loader = model_loader
        self._settings = settings

    def fleet(self, *, limit: int, offset: int) -> FleetOverviewResponse:
        now = datetime.now(UTC)
        latest_reading = _latest_reading_subquery()
        latest_prediction = _latest_prediction_subquery()
        with session_scope(self._sessions) as session:
            total = _count(session, select(func.count()).select_from(Asset))
            active = _count(
                session,
                select(func.count()).select_from(Asset).where(Asset.status == AssetStatus.ACTIVE),
            )
            latest_observation = session.scalar(select(func.max(SensorReading.observed_at)))
            rows = session.execute(
                select(
                    Asset.id,
                    Asset.external_id,
                    Asset.status,
                    latest_reading.c.cycle,
                    latest_reading.c.observed_at,
                    latest_prediction.c.predicted_rul,
                    latest_prediction.c.lower_rul,
                    latest_prediction.c.upper_rul,
                    latest_prediction.c.risk_level,
                    latest_prediction.c.prediction_timestamp,
                    latest_prediction.c.model_version,
                    latest_prediction.c.feature_version,
                )
                .outerjoin(latest_reading, latest_reading.c.asset_id == Asset.id)
                .outerjoin(latest_prediction, latest_prediction.c.asset_id == Asset.id)
                .order_by(Asset.external_id, Asset.id)
                .limit(limit)
                .offset(offset)
            ).all()
            risk_counts = {
                str(risk): int(count)
                for risk, count in session.execute(
                    select(latest_prediction.c.risk_level, func.count())
                    .select_from(Asset)
                    .join(latest_prediction, latest_prediction.c.asset_id == Asset.id)
                    .group_by(latest_prediction.c.risk_level)
                )
            }
            without_prediction = _count(
                session,
                select(func.count())
                .select_from(Asset)
                .outerjoin(latest_prediction, latest_prediction.c.asset_id == Asset.id)
                .where(
                    (latest_prediction.c.asset_id.is_(None))
                    | (
                        latest_prediction.c.prediction_timestamp
                        < now - timedelta(seconds=self._settings.asset_stale_after_seconds)
                    )
                ),
            )
            drift = session.scalar(
                select(DriftReport).order_by(desc(DriftReport.window_end)).limit(1)
            )
            performance = session.scalar(
                select(ModelEvaluation)
                .where(ModelEvaluation.evaluation_scope == EvaluationScope.ONLINE)
                .order_by(desc(ModelEvaluation.window_end), desc(ModelEvaluation.created_at))
                .limit(1)
            )
            replay = session.scalar(
                select(ReplayRun).order_by(desc(ReplayRun.created_at), desc(ReplayRun.id)).limit(1)
            )

        items = [
            FleetAssetItem(
                asset_id=row.id,
                external_asset_id=row.external_id,
                asset_status=row.status.value,
                latest_cycle=row.cycle,
                predicted_rul=row.predicted_rul,
                lower_rul=row.lower_rul,
                upper_rul=row.upper_rul,
                risk_level=None if row.risk_level is None else row.risk_level.value,
                latest_observation_at=row.observed_at,
                prediction_timestamp=row.prediction_timestamp,
                model_version=row.model_version,
                feature_version=row.feature_version,
                stale=_stale(row.observed_at, now, self._settings.asset_stale_after_seconds),
            )
            for row in rows
        ]
        try:
            model_version = self._loader.get().metadata.version
        except Exception:
            model_version = None
        return FleetOverviewResponse(
            total_assets=total,
            active_assets=active,
            latest_observation_at=latest_observation,
            healthy_count=risk_counts.get(RiskLevel.HEALTHY.value, 0),
            warning_count=risk_counts.get(RiskLevel.WARNING.value, 0),
            critical_count=risk_counts.get(RiskLevel.CRITICAL.value, 0),
            assets_without_recent_predictions=without_prediction,
            current_model_version=model_version,
            drift_status="unavailable" if drift is None else drift.status.value,
            performance_status=(
                "insufficient_data"
                if performance is None
                else str(performance.metrics.get("status", "available"))
            ),
            replay_status="not_started" if replay is None else replay.status.value,
            items=items,
            limit=limit,
            offset=offset,
        )

    def prediction_history(
        self,
        *,
        limit: int,
        offset: int = 0,
        asset_id: uuid.UUID | None = None,
        risk_level: str | None = None,
        model_version: str | None = None,
        since: datetime | None = None,
    ) -> PredictionHistoryResponse:
        query = select(Prediction, Asset.external_id).join(Asset, Asset.id == Prediction.asset_id)
        if asset_id is not None:
            query = query.where(Prediction.asset_id == asset_id)
        if risk_level is not None:
            try:
                risk = RiskLevel(risk_level)
            except ValueError as exc:
                raise RequestParameterError("Unknown risk-level filter.") from exc
            query = query.where(Prediction.risk_level == risk)
        if model_version is not None:
            query = query.where(Prediction.model_version == model_version)
        if since is not None:
            query = query.where(Prediction.prediction_timestamp >= since)
        with session_scope(self._sessions) as session:
            rows = session.execute(
                query.order_by(
                    desc(Prediction.prediction_timestamp),
                    desc(Prediction.created_at),
                    Asset.external_id,
                )
                .limit(limit)
                .offset(offset)
            ).all()
        return PredictionHistoryResponse(
            items=[_history_item(prediction, external_id) for prediction, external_id in rows],
            limit=limit,
            offset=offset,
        )

    def alerts(self, *, limit: int) -> AlertSummaryResponse:
        latest_prediction = _latest_prediction_subquery()
        alert_cycles = (
            select(
                Prediction.asset_id.label("asset_id"),
                func.min(
                    case(
                        (
                            Prediction.risk_level.in_([RiskLevel.WARNING, RiskLevel.CRITICAL]),
                            Prediction.cycle,
                        )
                    )
                ).label("first_warning_cycle"),
                func.min(
                    case((Prediction.risk_level == RiskLevel.CRITICAL, Prediction.cycle))
                ).label("first_critical_cycle"),
            )
            .group_by(Prediction.asset_id)
            .subquery()
        )
        latest_event = _latest_event_subquery()
        with session_scope(self._sessions) as session:
            count_rows = session.execute(
                select(latest_prediction.c.risk_level, func.count())
                .where(latest_prediction.c.risk_level.in_([RiskLevel.WARNING, RiskLevel.CRITICAL]))
                .group_by(latest_prediction.c.risk_level)
            ).all()
            rows = session.execute(
                select(
                    Asset.id,
                    Asset.external_id,
                    latest_prediction.c.risk_level,
                    latest_prediction.c.predicted_rul,
                    latest_prediction.c.prediction_timestamp,
                    latest_prediction.c.model_version,
                    alert_cycles.c.first_warning_cycle,
                    alert_cycles.c.first_critical_cycle,
                    latest_event.c.event_type,
                )
                .join(latest_prediction, latest_prediction.c.asset_id == Asset.id)
                .outerjoin(alert_cycles, alert_cycles.c.asset_id == Asset.id)
                .outerjoin(latest_event, latest_event.c.asset_id == Asset.id)
                .where(latest_prediction.c.risk_level.in_([RiskLevel.WARNING, RiskLevel.CRITICAL]))
                .order_by(
                    case((latest_prediction.c.risk_level == RiskLevel.CRITICAL, 0), else_=1),
                    latest_prediction.c.predicted_rul,
                    Asset.external_id,
                )
                .limit(limit)
            ).all()
        now = datetime.now(UTC)
        items = [
            AlertAssetItem(
                asset_id=row.id,
                external_asset_id=row.external_id,
                current_risk_level=row.risk_level.value,
                first_warning_cycle=row.first_warning_cycle,
                first_critical_cycle=row.first_critical_cycle,
                latest_predicted_rul=row.predicted_rul,
                latest_prediction_at=row.prediction_timestamp,
                alert_age_seconds=max(
                    0.0, (now - _aware(row.prediction_timestamp)).total_seconds()
                ),
                model_version=row.model_version,
                outcome=None if row.event_type is None else row.event_type.value,
            )
            for row in rows
        ]
        return AlertSummaryResponse(
            warning_count=next(
                (int(count) for risk, count in count_rows if risk == RiskLevel.WARNING), 0
            ),
            critical_count=next(
                (int(count) for risk, count in count_rows if risk == RiskLevel.CRITICAL), 0
            ),
            items=items,
            limit=limit,
        )

    def asset(
        self, asset_id: uuid.UUID, *, sensor_columns: tuple[str, ...], limit: int
    ) -> AssetDashboardResponse:
        columns = _validate_sensor_columns(sensor_columns)
        with session_scope(self._sessions) as session:
            asset = session.get(Asset, asset_id)
            if asset is None:
                raise AssetNotFoundError("The requested asset does not exist.")
            reading = session.scalar(
                select(SensorReading)
                .where(SensorReading.asset_id == asset_id)
                .order_by(desc(SensorReading.cycle))
                .limit(1)
            )
            prediction = session.scalar(
                select(Prediction)
                .where(Prediction.asset_id == asset_id)
                .order_by(desc(Prediction.prediction_timestamp), desc(Prediction.created_at))
                .limit(1)
            )
            prediction_query = select(Prediction).where(Prediction.asset_id == asset_id)
            if prediction is not None:
                prediction_query = prediction_query.where(
                    Prediction.model_name == prediction.model_name,
                    Prediction.model_version == prediction.model_version,
                )
            prediction_rows = list(
                reversed(
                    list(
                        session.scalars(
                            prediction_query.order_by(
                                desc(Prediction.prediction_timestamp), desc(Prediction.created_at)
                            ).limit(limit)
                        )
                    )
                )
            )
            reading_rows = list(
                reversed(
                    list(
                        session.scalars(
                            select(SensorReading)
                            .where(SensorReading.asset_id == asset_id)
                            .order_by(desc(SensorReading.cycle))
                            .limit(limit)
                        )
                    )
                )
            )
            events = list(
                session.scalars(
                    select(MaintenanceEvent)
                    .where(MaintenanceEvent.asset_id == asset_id)
                    .order_by(desc(MaintenanceEvent.occurred_at))
                    .limit(50)
                )
            )
            replay = session.scalar(
                select(ReplayRun)
                .where(ReplayRun.asset_id == asset_id)
                .order_by(desc(ReplayRun.attempt))
                .limit(1)
            )
            quality = session.scalar(
                select(DataQualityReport).order_by(desc(DataQualityReport.window_end)).limit(1)
            )
        now = datetime.now(UTC)
        stale = _stale(
            None if reading is None else reading.observed_at,
            now,
            self._settings.asset_stale_after_seconds,
        )
        warnings: list[str] = []
        if reading is None:
            warnings.append("No sensor readings are available.")
        if prediction is None:
            warnings.append("No prediction is available for the latest asset state.")
        if stale:
            warnings.append("The latest observation is older than the configured freshness window.")
        if quality is not None and quality.status.value in {"warning", "fail"}:
            warnings.append(f"The latest fleet data-quality report is {quality.status.value}.")
        return AssetDashboardResponse(
            asset_id=asset.id,
            external_asset_id=asset.external_id,
            asset_status=asset.status.value,
            dataset_name=asset.dataset_name,
            dataset_subset=asset.dataset_subset,
            source_asset_id=asset.source_asset_id,
            latest_cycle=None if reading is None else reading.cycle,
            predicted_rul=None if prediction is None else prediction.predicted_rul,
            lower_rul=None if prediction is None else prediction.lower_rul,
            upper_rul=None if prediction is None else prediction.upper_rul,
            risk_level=None if prediction is None else prediction.risk_level.value,
            failure_within_30=None if prediction is None else prediction.failure_within_30,
            failure_within_50=None if prediction is None else prediction.failure_within_50,
            model_version=None if prediction is None else prediction.model_version,
            feature_version=None if prediction is None else prediction.feature_version,
            latest_observation_at=None if reading is None else reading.observed_at,
            stale=stale,
            data_quality_warnings=warnings,
            predictions=[_history_item(item, asset.external_id) for item in prediction_rows],
            sensor_columns=list(columns),
            available_sensor_columns=list(SENSOR_COLUMNS),
            sensor_history=[
                SensorHistoryPoint(
                    cycle=item.cycle,
                    observed_at=item.observed_at,
                    values={column: float(getattr(item, column)) for column in columns},
                )
                for item in reading_rows
            ],
            maintenance_events=[
                MaintenanceEventSummaryResponse(
                    event_id=event.id,
                    event_type=event.event_type.value,
                    event_cycle=event.event_cycle,
                    occurred_at=event.occurred_at,
                    source=event.source,
                    description=event.description,
                )
                for event in events
            ],
            replay=None if replay is None else replay_response(replay),
        )

    def drift(self) -> DriftDetailResponse:
        with session_scope(self._sessions) as session:
            report = session.scalar(
                select(DriftReport).order_by(desc(DriftReport.window_end)).limit(1)
            )
        note = "Drift is a change signal; it does not by itself prove that the model is wrong."
        if report is None:
            return DriftDetailResponse(
                status="insufficient_data",
                available=False,
                model_name=None,
                model_version=None,
                feature_version=None,
                window_start=None,
                window_end=None,
                drifted_feature_count=0,
                max_psi=None,
                max_wasserstein=None,
                report_timestamp=None,
                top_features=[],
                note=note,
            )
        records = cast(list[dict[str, Any]], report.details.get("features", []))
        ranked = sorted(records, key=_drift_rank, reverse=True)[
            : self._settings.dashboard_top_drift_features
        ]
        return DriftDetailResponse(
            status=report.status.value,
            available=True,
            model_name=report.model_name,
            model_version=report.model_version,
            feature_version=report.feature_version,
            window_start=report.window_start,
            window_end=report.window_end,
            drifted_feature_count=report.drifted_feature_count,
            max_psi=report.max_psi,
            max_wasserstein=report.max_wasserstein,
            report_timestamp=report.created_at,
            top_features=[
                DriftFeatureItem(
                    feature=str(record.get("feature", "unknown")),
                    psi=_float_or_none(record.get("psi")),
                    wasserstein=_float_or_none(record.get("wasserstein")),
                    normalized_wasserstein=_float_or_none(record.get("normalized_wasserstein")),
                    missingness_shift=_float_or_none(record.get("missingness_shift")),
                    drifted=bool(record.get("drifted", False)),
                    warning=bool(record.get("warning", False)),
                )
                for record in ranked
            ],
            note=note,
        )

    def performance(self) -> PerformanceDetailResponse:
        with session_scope(self._sessions) as session:
            report = session.scalar(
                select(ModelEvaluation)
                .where(ModelEvaluation.evaluation_scope == EvaluationScope.ONLINE)
                .order_by(desc(ModelEvaluation.window_end), desc(ModelEvaluation.created_at))
                .limit(1)
            )
        if report is None:
            return PerformanceDetailResponse(
                status="insufficient_data",
                available=False,
                target_label="Uncapped realized RUL (cycles)",
                model_name=None,
                model_version=None,
                window_start=None,
                window_end=None,
                labeled_rows=0,
                completed_assets=None,
                mae=None,
                rmse=None,
                nasa_score=None,
                critical_precision=None,
                critical_recall=None,
                critical_f1=None,
                false_alarms_per_1000_cycles=None,
                mean_alert_lead_time=None,
                timely_alert_rate=None,
                interval_coverage=None,
                average_interval_width=None,
                report_timestamp=None,
            )
        critical = cast(dict[str, Any], report.metrics.get("critical", {}))
        interval = cast(dict[str, Any], report.metrics.get("interval") or {})
        return PerformanceDetailResponse(
            status=str(report.metrics.get("status", "available")),
            available=True,
            target_label=(
                "Uncapped realized RUL (cycles); deployed predictions may use a capped target"
            ),
            model_name=report.model_name,
            model_version=report.model_version,
            window_start=report.window_start,
            window_end=report.window_end,
            labeled_rows=report.sample_count,
            completed_assets=_int_or_none(report.metrics.get("asset_count")),
            mae=report.mae,
            rmse=report.rmse,
            nasa_score=report.nasa_score,
            critical_precision=report.critical_precision,
            critical_recall=report.critical_recall,
            critical_f1=_float_or_none(critical.get("f1")),
            false_alarms_per_1000_cycles=_float_or_none(
                critical.get("false_alarms_per_1000_cycles")
            ),
            mean_alert_lead_time=_float_or_none(critical.get("mean_first_alert_lead_time")),
            timely_alert_rate=_float_or_none(critical.get("timely_warning_asset_percentage")),
            interval_coverage=report.interval_coverage,
            average_interval_width=_float_or_none(interval.get("average_width")),
            report_timestamp=report.created_at,
        )

    def model(self) -> ModelOverviewResponse:
        with session_scope(self._sessions) as session:
            lifecycle_rows = list(
                session.scalars(
                    select(PipelineRun).order_by(desc(PipelineRun.started_at)).limit(10)
                )
            )
            event = session.scalar(
                select(LifecycleEvent).order_by(desc(LifecycleEvent.created_at)).limit(1)
            )
        lifecycle = [
            LifecycleItem(
                run_id=run.id,
                run_type=run.run_type.value,
                status=run.status.value,
                phase=run.phase,
                model_version=run.model_version,
                started_at=run.started_at,
                finished_at=run.finished_at,
                decision=_nested_string(run.run_metadata, "trigger_decision", "action"),
                candidate_version=_string_or_none(run.run_metadata.get("candidate_version")),
            )
            for run in lifecycle_rows
        ]
        latest_event = None
        if event is not None:
            latest_event = {
                "event_type": event.event_type,
                "phase": event.phase,
                "from_version": event.from_version,
                "to_version": event.to_version,
                "actor": event.actor,
                "created_at": event.created_at.isoformat(),
            }
        try:
            metadata = self._loader.get().metadata
            alias_values = self._loader.registry_aliases()
            return ModelOverviewResponse(
                available=True,
                registry_source=metadata.registry_source,
                registered_model_name=metadata.model_name,
                registry_version=metadata.version,
                alias=metadata.alias,
                aliases=alias_values,
                model_family=metadata.model_family,
                target_definition=metadata.target_definition,
                rul_cap=metadata.rul_cap,
                feature_count=metadata.feature_count,
                feature_version=metadata.feature_version,
                validation_rmse=metadata.validation_rmse,
                replay_rmse=metadata.replay_rmse,
                official_benchmark_rmse=metadata.official_test_rmse,
                conformal_coverage_target=metadata.conformal_coverage_target,
                source_run_id=metadata.source_run_id,
                model_load_timestamp=metadata.loaded_at,
                git_sha=metadata.git_sha,
                manifest_lineage={
                    key: value
                    for key, value in {
                        "lineage_id": metadata.lineage_id,
                        "dataset_checksum": metadata.dataset_checksum,
                        "feature_manifest_checksum": metadata.feature_manifest_checksum,
                        "model_checksum": metadata.checksum,
                    }.items()
                    if value is not None
                },
                latest_lifecycle=lifecycle,
                latest_event=latest_event,
            )
        except Exception:
            return ModelOverviewResponse(
                available=False,
                registry_source=None,
                registered_model_name=self._settings.mlflow_registered_model_name,
                registry_version=None,
                alias=None,
                aliases={},
                model_family=None,
                target_definition=None,
                rul_cap=None,
                feature_count=None,
                feature_version=None,
                validation_rmse=None,
                replay_rmse=None,
                official_benchmark_rmse=None,
                conformal_coverage_target=None,
                source_run_id=None,
                model_load_timestamp=None,
                git_sha=None,
                manifest_lineage={},
                latest_lifecycle=lifecycle,
                latest_event=latest_event,
            )


def replay_response(run: ReplayRun) -> ReplayRunResponse:
    """Hide the final-cycle ground truth until the replay is complete."""
    completed = run.status == ReplayRunStatus.COMPLETED
    return ReplayRunResponse(
        run_id=run.id,
        source_asset_id=run.source_asset_id,
        attempt=run.attempt,
        external_asset_id=run.external_asset_id,
        operational_asset_id=run.asset_id,
        status=run.status.value,
        mode=run.mode.value,
        last_confirmed_cycle=run.last_confirmed_cycle,
        final_cycle=run.final_cycle if completed else None,
        progress_percent=round(100.0 * run.last_confirmed_cycle / run.final_cycle, 1),
        started_at=run.replay_started_at,
        last_advanced_at=run.last_advanced_at,
        completed_at=run.completed_at,
        error_message=run.error_message,
    )


def _latest_reading_subquery() -> Any:
    ranked = select(
        SensorReading.asset_id,
        SensorReading.cycle,
        SensorReading.observed_at,
        func.row_number()
        .over(
            partition_by=SensorReading.asset_id,
            order_by=(desc(SensorReading.cycle), desc(SensorReading.ingested_at)),
        )
        .label("position"),
    ).subquery()
    return (
        select(ranked.c.asset_id, ranked.c.cycle, ranked.c.observed_at)
        .where(ranked.c.position == 1)
        .subquery()
    )


def _latest_prediction_subquery() -> Any:
    ranked = select(
        Prediction.asset_id,
        Prediction.predicted_rul,
        Prediction.lower_rul,
        Prediction.upper_rul,
        Prediction.risk_level,
        Prediction.prediction_timestamp,
        Prediction.model_version,
        Prediction.feature_version,
        func.row_number()
        .over(
            partition_by=Prediction.asset_id,
            order_by=(desc(Prediction.prediction_timestamp), desc(Prediction.created_at)),
        )
        .label("position"),
    ).subquery()
    return (
        select(*[column for column in ranked.c if column.key != "position"])
        .where(ranked.c.position == 1)
        .subquery()
    )


def _latest_event_subquery() -> Any:
    ranked = select(
        MaintenanceEvent.asset_id,
        MaintenanceEvent.event_type,
        func.row_number()
        .over(
            partition_by=MaintenanceEvent.asset_id,
            order_by=(desc(MaintenanceEvent.occurred_at), desc(MaintenanceEvent.created_at)),
        )
        .label("position"),
    ).subquery()
    return select(ranked.c.asset_id, ranked.c.event_type).where(ranked.c.position == 1).subquery()


def _history_item(prediction: Prediction, external_id: str) -> PredictionHistoryItem:
    return PredictionHistoryItem(
        prediction_id=prediction.id,
        asset_id=prediction.asset_id,
        external_asset_id=external_id,
        cycle=prediction.cycle,
        predicted_rul=prediction.predicted_rul,
        lower_rul=(
            prediction.predicted_rul if prediction.lower_rul is None else prediction.lower_rul
        ),
        upper_rul=(
            prediction.predicted_rul if prediction.upper_rul is None else prediction.upper_rul
        ),
        risk_level=prediction.risk_level.value,
        model_name=prediction.model_name,
        model_version=prediction.model_version,
        feature_version=prediction.feature_version,
        prediction_timestamp=prediction.prediction_timestamp,
        latency_ms=prediction.latency_ms,
    )


def _validate_sensor_columns(columns: tuple[str, ...]) -> tuple[str, ...]:
    selected = columns or ("sensor_02", "sensor_04", "sensor_07", "sensor_11")
    if len(selected) > 6 or len(set(selected)) != len(selected):
        raise RequestParameterError("Select at most six distinct anonymous sensors.")
    if any(column not in SENSOR_COLUMNS for column in selected):
        raise RequestParameterError("Sensor filters must use sensor_01 through sensor_21.")
    return selected


def _stale(observed_at: datetime | None, now: datetime, threshold: int) -> bool:
    return observed_at is None or (now - _aware(observed_at)).total_seconds() > threshold


def _aware(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _count(session: Session, statement: Any) -> int:
    return int(session.scalar(statement) or 0)


def _drift_rank(record: dict[str, Any]) -> float:
    values = (
        _float_or_none(record.get("psi")),
        _float_or_none(record.get("normalized_wasserstein")),
        _float_or_none(record.get("missingness_shift")),
    )
    return max((value for value in values if value is not None), default=-1.0)


def _float_or_none(value: Any) -> float | None:
    return None if value is None else float(value)


def _int_or_none(value: Any) -> int | None:
    return None if value is None else int(value)


def _string_or_none(value: Any) -> str | None:
    return None if value is None else str(value)


def _nested_string(value: dict[str, Any], parent: str, child: str) -> str | None:
    nested = value.get(parent)
    return str(nested.get(child)) if isinstance(nested, dict) and nested.get(child) else None

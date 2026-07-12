"""Atomic online inference and read-side asset services."""

import time
import uuid
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast

import pandas as pd
from sqlalchemy.orm import Session, sessionmaker

from turbine_guard.api.schemas.online import (
    AssetDetailResponse,
    AssetHealthResponse,
    AssetListResponse,
    AssetSummaryResponse,
    CurrentModelResponse,
    MaintenanceEventSummaryResponse,
    MonitoringSummaryResponse,
    PredictionResponse,
    PredictionTrendPoint,
    ReadingSummaryResponse,
    RecentPredictionItem,
    RecentPredictionsResponse,
    SensorIngestionResponse,
)
from turbine_guard.config.settings import Settings
from turbine_guard.data.schema import OPERATING_SETTING_COLUMNS, SENSOR_COLUMNS, TRAJECTORY_COLUMNS
from turbine_guard.database.commands import NewAsset, NewPrediction, NewSensorReading
from turbine_guard.database.enums import RiskLevel
from turbine_guard.database.errors import DuplicateExternalIdError
from turbine_guard.database.models import Asset, Prediction, SensorReading
from turbine_guard.database.repositories import (
    AssetRepository,
    MaintenanceEventRepository,
    PredictionRepository,
    SensorReadingRepository,
)
from turbine_guard.database.session import session_scope
from turbine_guard.features.builder import FeatureBuilder, FeatureError
from turbine_guard.observability.metrics import OnlineMetrics
from turbine_guard.services.errors import (
    AssetNotFoundError,
    FeatureContractError,
    HistoryConflictError,
    ModelUnavailableError,
)
from turbine_guard.serving.model_loader import (
    ChampionModelLoader,
    LoadedChampion,
    ModelMetadata,
    validate_prediction_output,
)


@dataclass(frozen=True)
class SensorObservation:
    external_asset_id: str
    cycle: int
    observed_at: datetime | None
    operating_settings: tuple[float, float, float]
    sensor_values: tuple[float, ...]
    source: str
    ingestion_id: str | None
    schema_version: str


class OnlineInferenceService:
    """Application boundary above repositories, features, and the MLflow cache."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        model_loader: ChampionModelLoader,
        metrics: OnlineMetrics,
        settings: Settings,
    ) -> None:
        self._sessions = session_factory
        self._loader = model_loader
        self._metrics = metrics
        self._settings = settings

    def ingest(self, observation: SensorObservation, request_id: str) -> SensorIngestionResponse:
        """Persist reading and prediction in one all-or-nothing transaction."""
        try:
            loaded = self._loader.get()
        except Exception as exc:
            raise ModelUnavailableError("The champion model is not available.") from exc

        with session_scope(self._sessions) as session:
            assets = AssetRepository(session)
            sensors = SensorReadingRepository(session)
            predictions = PredictionRepository(session)
            asset = self._resolve_asset(assets, observation)
            existing_reading = sensors.get_by_asset_cycle(asset.id, observation.cycle)
            if existing_reading is None:
                self._validate_next_cycle(sensors.latest(asset.id), observation.cycle)
            observed_at = observation.observed_at
            if observed_at is None:
                observed_at = (
                    existing_reading.observed_at
                    if existing_reading is not None
                    else datetime.now(UTC)
                )
            command = NewSensorReading(
                asset_id=asset.id,
                cycle=observation.cycle,
                observed_at=observed_at,
                operating_settings=observation.operating_settings,
                sensor_values=_sensor_tuple(observation.sensor_values),
                schema_version=observation.schema_version,
                source=observation.source,
                ingestion_id=observation.ingestion_id,
            )
            reading = sensors.insert(command)
            existing_prediction = predictions.get_for_model(
                reading.id, loaded.metadata.model_name, loaded.metadata.version
            )
            if existing_prediction is not None:
                response = _ingestion_response(
                    asset,
                    reading,
                    existing_prediction,
                    request_id=request_id,
                    reading_idempotent=True,
                    prediction_idempotent=True,
                )
            else:
                history = sensors.history_through(asset.id, observation.cycle)
                features = self._features(history, loaded)
                started = time.perf_counter()
                try:
                    output = loaded.model.predict(features)
                    point, lower, upper, risk = validate_prediction_output(output)
                except Exception as exc:
                    raise ModelUnavailableError("The champion model could not predict.") from exc
                latency_ms = (time.perf_counter() - started) * 1000.0
                timestamp = datetime.now(UTC)
                prediction = predictions.create(
                    NewPrediction(
                        asset_id=asset.id,
                        sensor_reading_id=reading.id,
                        cycle=reading.cycle,
                        predicted_rul=point,
                        lower_rul=lower,
                        upper_rul=upper,
                        risk_level=RiskLevel(risk),
                        failure_within_30=point <= 30,
                        failure_within_50=point <= 50,
                        model_name=loaded.metadata.model_name,
                        model_version=loaded.metadata.version,
                        model_alias=loaded.metadata.alias,
                        model_run_id=loaded.metadata.source_run_id,
                        feature_version=loaded.metadata.feature_version,
                        prediction_timestamp=timestamp,
                        latency_ms=latency_ms,
                    )
                )
                response = _ingestion_response(
                    asset,
                    reading,
                    prediction,
                    request_id=request_id,
                    reading_idempotent=existing_reading is not None,
                    prediction_idempotent=False,
                )
        self._metrics.set_model(
            loaded.metadata.model_name, loaded.metadata.version, loaded.metadata.alias
        )
        assert response.prediction.latency_ms is not None
        self._metrics.record_prediction(
            response.prediction.risk_level, response.prediction.latency_ms
        )
        return response

    def list_assets(self, *, limit: int, offset: int) -> AssetListResponse:
        with session_scope(self._sessions) as session:
            assets = AssetRepository(session).list(limit=limit, offset=offset)
            sensors = SensorReadingRepository(session)
            predictions = PredictionRepository(session)
            items = [
                _asset_summary(asset, sensors.latest(asset.id), predictions.latest(asset.id))
                for asset in assets
            ]
        return AssetListResponse(items=items, limit=limit, offset=offset)

    def get_asset(self, asset_id: uuid.UUID) -> AssetDetailResponse:
        with session_scope(self._sessions) as session:
            asset = _require_asset(AssetRepository(session).get(asset_id))
            reading = SensorReadingRepository(session).latest(asset.id)
            prediction = PredictionRepository(session).latest(asset.id)
            events = MaintenanceEventRepository(session).for_asset(asset.id)[-10:]
            return AssetDetailResponse(
                asset_id=asset.id,
                external_asset_id=asset.external_id,
                dataset_name=asset.dataset_name,
                dataset_subset=asset.dataset_subset,
                source_asset_id=asset.source_asset_id,
                status=asset.status.value,
                created_at=asset.created_at,
                updated_at=asset.updated_at,
                latest_reading=_reading_response(reading),
                latest_prediction=_prediction_response(prediction),
                recent_maintenance_events=[
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
            )

    def get_asset_health(self, asset_id: uuid.UUID) -> AssetHealthResponse:
        with session_scope(self._sessions) as session:
            asset = _require_asset(AssetRepository(session).get(asset_id))
            reading = SensorReadingRepository(session).latest(asset.id)
            repository = PredictionRepository(session)
            prediction = repository.latest(asset.id)
            trend = list(
                reversed(
                    repository.recent(
                        asset_id=asset.id, limit=self._settings.api_prediction_trend_size
                    )
                )
            )
            stale = (
                reading is not None
                and (datetime.now(UTC) - reading.observed_at).total_seconds()
                > self._settings.asset_stale_after_seconds
            )
            return AssetHealthResponse(
                asset_id=asset.id,
                external_asset_id=asset.external_id,
                latest_cycle=None if reading is None else reading.cycle,
                predicted_rul=None if prediction is None else prediction.predicted_rul,
                lower_rul=None if prediction is None else prediction.lower_rul,
                upper_rul=None if prediction is None else prediction.upper_rul,
                risk_level=None if prediction is None else prediction.risk_level.value,
                failure_within_30=(None if prediction is None else prediction.failure_within_30),
                failure_within_50=(None if prediction is None else prediction.failure_within_50),
                prediction_trend=[
                    PredictionTrendPoint(
                        cycle=item.cycle,
                        predicted_rul=item.predicted_rul,
                        risk_level=item.risk_level.value,
                        prediction_timestamp=item.prediction_timestamp,
                    )
                    for item in trend
                ],
                latest_observation_at=None if reading is None else reading.observed_at,
                model_version=None if prediction is None else prediction.model_version,
                stale=stale,
                data_quality_status="no_data" if reading is None else "valid",
            )

    def recent_predictions(
        self, *, limit: int, asset_id: uuid.UUID | None = None
    ) -> RecentPredictionsResponse:
        with session_scope(self._sessions) as session:
            assets = AssetRepository(session)
            predictions = PredictionRepository(session).recent(asset_id=asset_id, limit=limit)
            items: list[RecentPredictionItem] = []
            for prediction in predictions:
                asset = _require_asset(assets.get(prediction.asset_id))
                prediction_response = _prediction_response(prediction)
                assert prediction_response is not None
                items.append(
                    RecentPredictionItem(
                        prediction_id=prediction.id,
                        asset_id=asset.id,
                        external_asset_id=asset.external_id,
                        reading_id=prediction.sensor_reading_id,
                        cycle=prediction.cycle,
                        prediction=prediction_response,
                    )
                )
        return RecentPredictionsResponse(items=items, limit=limit)

    def current_model(self) -> CurrentModelResponse:
        try:
            metadata = self._loader.get().metadata
        except Exception as exc:
            raise ModelUnavailableError("The champion model is not available.") from exc
        return _model_response(metadata)

    def monitoring_summary(self) -> MonitoringSummaryResponse:
        snapshot = self._metrics.snapshot()
        with session_scope(self._sessions) as session:
            sensors = SensorReadingRepository(session)
            predictions = PredictionRepository(session)
            try:
                model_version = self._loader.get().metadata.version
            except Exception:
                model_version = None
            return MonitoringSummaryResponse(
                request_count=snapshot.request_count,
                prediction_count=snapshot.prediction_count,
                validation_failures=snapshot.validation_failures,
                database_failures=snapshot.database_failures,
                model_load_failures=snapshot.model_load_failures,
                prediction_failures=snapshot.prediction_failures,
                conflict_count=snapshot.conflict_count,
                average_prediction_latency_ms=snapshot.average_prediction_latency_ms,
                current_model_version=model_version,
                recent_risk_distribution=predictions.risk_distribution(),
                reading_count=sensors.count(),
                stored_prediction_count=predictions.count(),
                latest_ingestion_time=sensors.latest_ingested_at(),
            )

    def _resolve_asset(self, repository: AssetRepository, observation: SensorObservation) -> Asset:
        asset = repository.get_by_external_id_for_update(observation.external_asset_id)
        if asset is not None:
            return asset
        if observation.cycle != 1:
            raise HistoryConflictError("A new asset must begin at cycle 1.")
        with suppress(DuplicateExternalIdError):
            repository.create(NewAsset(external_id=observation.external_asset_id))
        asset = repository.get_by_external_id_for_update(observation.external_asset_id)
        if asset is None:
            raise RuntimeError("Asset could not be resolved after creation.")
        return asset

    @staticmethod
    def _validate_next_cycle(latest: SensorReading | None, cycle: int) -> None:
        expected = 1 if latest is None else latest.cycle + 1
        if cycle != expected:
            raise HistoryConflictError(
                f"New readings must be contiguous; expected cycle {expected}."
            )

    @staticmethod
    def _features(history: list[SensorReading], loaded: LoadedChampion) -> pd.DataFrame:
        cycles = [reading.cycle for reading in history]
        if cycles != list(range(1, cycles[-1] + 1)):
            raise HistoryConflictError("Stored asset history is not contiguous from cycle 1.")
        frame = _history_frame(history)
        builder = FeatureBuilder(loaded.feature_config)
        if builder.feature_columns() != loaded.feature_columns:
            raise FeatureContractError("Feature builder and champion contracts do not match.")
        try:
            generated = builder.transform_asset(frame)
        except FeatureError as exc:
            raise FeatureContractError("Features could not be reconstructed.") from exc
        current = generated.loc[generated["cycle"] == history[-1].cycle]
        if len(current) != 1:
            raise FeatureContractError("Current-cycle feature row is ambiguous.")
        features = current.loc[:, list(loaded.feature_columns)]
        if tuple(features.columns) != loaded.feature_columns:
            raise FeatureContractError("Generated feature order is incompatible with champion.")
        return features


def _history_frame(history: list[SensorReading]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for reading in history:
        row: dict[str, object] = {"asset_id": reading.asset_id, "cycle": reading.cycle}
        row.update({name: getattr(reading, name) for name in OPERATING_SETTING_COLUMNS})
        row.update({name: getattr(reading, name) for name in SENSOR_COLUMNS})
        rows.append(row)
    return pd.DataFrame(rows, columns=list(TRAJECTORY_COLUMNS))


def _sensor_tuple(
    values: tuple[float, ...],
) -> tuple[
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
]:
    if len(values) != 21:
        raise ValueError("Exactly 21 sensor values are required.")
    return cast(
        tuple[
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
        ],
        values,
    )


def _require_asset(asset: Asset | None) -> Asset:
    if asset is None:
        raise AssetNotFoundError("The requested asset does not exist.")
    return asset


def _prediction_response(prediction: Prediction | None) -> PredictionResponse | None:
    if prediction is None:
        return None
    return PredictionResponse(
        predicted_rul=prediction.predicted_rul,
        lower_rul=prediction.lower_rul
        if prediction.lower_rul is not None
        else prediction.predicted_rul,
        upper_rul=prediction.upper_rul
        if prediction.upper_rul is not None
        else prediction.predicted_rul,
        risk_level=prediction.risk_level.value,
        failure_within_30=bool(prediction.failure_within_30),
        failure_within_50=bool(prediction.failure_within_50),
        model_name=prediction.model_name,
        model_version=prediction.model_version,
        model_alias=prediction.model_alias or "",
        model_run_id=prediction.model_run_id,
        feature_version=prediction.feature_version,
        prediction_timestamp=prediction.prediction_timestamp,
        latency_ms=prediction.latency_ms,
    )


def _ingestion_response(
    asset: Asset,
    reading: SensorReading,
    prediction: Prediction,
    *,
    request_id: str,
    reading_idempotent: bool,
    prediction_idempotent: bool,
) -> SensorIngestionResponse:
    response = _prediction_response(prediction)
    assert response is not None
    return SensorIngestionResponse(
        request_id=request_id,
        asset_id=asset.id,
        external_asset_id=asset.external_id,
        cycle=reading.cycle,
        reading_id=reading.id,
        prediction=response,
        idempotent=reading_idempotent and prediction_idempotent,
        reading_idempotent=reading_idempotent,
        prediction_idempotent=prediction_idempotent,
    )


def _asset_summary(
    asset: Asset, reading: SensorReading | None, prediction: Prediction | None
) -> AssetSummaryResponse:
    return AssetSummaryResponse(
        asset_id=asset.id,
        external_asset_id=asset.external_id,
        status=asset.status.value,
        latest_cycle=None if reading is None else reading.cycle,
        latest_risk_level=None if prediction is None else prediction.risk_level.value,
        latest_predicted_rul=None if prediction is None else prediction.predicted_rul,
        last_observed_at=None if reading is None else reading.observed_at,
    )


def _reading_response(reading: SensorReading | None) -> ReadingSummaryResponse | None:
    if reading is None:
        return None
    return ReadingSummaryResponse(
        reading_id=reading.id,
        cycle=reading.cycle,
        observed_at=reading.observed_at,
        source=reading.source,
        schema_version=reading.schema_version,
    )


def _model_response(metadata: ModelMetadata) -> CurrentModelResponse:
    return CurrentModelResponse(
        model_name=metadata.model_name,
        registry_version=metadata.version,
        alias=metadata.alias,
        source_run_id=metadata.source_run_id,
        target_definition=metadata.target_definition,
        rul_cap=metadata.rul_cap,
        feature_count=metadata.feature_count,
        feature_version=metadata.feature_version,
        validation_rmse=metadata.validation_rmse,
        replay_rmse=metadata.replay_rmse,
        official_test_rmse=metadata.official_test_rmse,
        conformal_coverage_target=metadata.conformal_coverage_target,
        model_load_timestamp=metadata.loaded_at,
        model_checksum=metadata.checksum,
        lineage_id=metadata.lineage_id,
    )

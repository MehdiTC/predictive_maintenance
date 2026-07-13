"""Focused repositories participating in caller-owned SQLAlchemy transactions."""

import uuid
from dataclasses import asdict
from datetime import datetime

from sqlalchemy import Select, desc, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from turbine_guard.data.schema import OPERATING_SETTING_COLUMNS, SENSOR_COLUMNS
from turbine_guard.database.commands import (
    NewAsset,
    NewDriftReport,
    NewMaintenanceEvent,
    NewModelEvaluation,
    NewPipelineRun,
    NewPrediction,
    NewPredictionOutcome,
    NewReplayRun,
    NewSensorReading,
)
from turbine_guard.database.enums import (
    AssetStatus,
    MaintenanceEventType,
    PipelineRunStatus,
)
from turbine_guard.database.errors import (
    DuplicateExternalIdError,
    PredictionConflictError,
    PredictionOutcomeConflictError,
    ReplayRunConflictError,
    SensorReadingConflictError,
)
from turbine_guard.database.models import (
    Asset,
    DriftReport,
    MaintenanceEvent,
    ModelEvaluation,
    PipelineRun,
    Prediction,
    PredictionOutcome,
    ReplayRun,
    SensorReading,
)


class AssetRepository:
    """Create, retrieve, list, and update assets without owning commits."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, command: NewAsset) -> Asset:
        asset = Asset(**asdict(command))
        try:
            with self.session.begin_nested():
                self.session.add(asset)
                self.session.flush()
        except IntegrityError as exc:
            raise DuplicateExternalIdError(
                f"Asset external_id {command.external_id!r} already exists."
            ) from exc
        return asset

    def get(self, asset_id: uuid.UUID) -> Asset | None:
        return self.session.get(Asset, asset_id)

    def get_by_external_id(self, external_id: str) -> Asset | None:
        return self.session.scalar(select(Asset).where(Asset.external_id == external_id))

    def get_by_external_id_for_update(self, external_id: str) -> Asset | None:
        """Lock an asset so ingestion order is serialized per asset."""
        return self.session.scalar(
            select(Asset).where(Asset.external_id == external_id).with_for_update()
        )

    def list(self, *, limit: int = 100, offset: int = 0) -> list[Asset]:
        if limit <= 0 or offset < 0:
            raise ValueError("limit must be positive and offset non-negative.")
        return list(
            self.session.scalars(
                select(Asset).order_by(Asset.external_id).limit(limit).offset(offset)
            )
        )

    def update_status(self, asset: Asset, status: AssetStatus) -> Asset:
        asset.status = status
        self.session.flush()
        return asset


class SensorReadingRepository:
    """Immutable cycle storage with exact-replay idempotency."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def insert(self, command: NewSensorReading) -> SensorReading:
        values = _sensor_values(command)
        statement = (
            insert(SensorReading)
            .values(**values)
            .on_conflict_do_nothing()
            .returning(SensorReading.id)
        )
        inserted_id = self.session.scalar(statement)
        if inserted_id is not None:
            reading = self.session.get(SensorReading, inserted_id)
            assert reading is not None
            return reading
        existing = self.get_by_asset_cycle(command.asset_id, command.cycle)
        if existing is None or not _same_sensor(existing, command):
            raise SensorReadingConflictError(
                f"Asset {command.asset_id} cycle {command.cycle} already has different data."
            )
        return existing

    def insert_many(self, commands: list[NewSensorReading]) -> list[SensorReading]:
        """Insert a whole batch atomically; caller rollback prevents partial publication."""
        seen: dict[tuple[uuid.UUID, int], NewSensorReading] = {}
        for command in commands:
            key = (command.asset_id, command.cycle)
            prior = seen.get(key)
            if prior is not None and prior != command:
                raise SensorReadingConflictError(
                    f"Conflicting reading duplicated within batch: {key}."
                )
            seen[key] = command
        with self.session.begin_nested():
            return [self.insert(command) for command in seen.values()]

    def get_by_asset_cycle(self, asset_id: uuid.UUID, cycle: int) -> SensorReading | None:
        return self.session.scalar(
            select(SensorReading).where(
                SensorReading.asset_id == asset_id, SensorReading.cycle == cycle
            )
        )

    def history_through(self, asset_id: uuid.UUID, cycle: int) -> list[SensorReading]:
        return list(
            self.session.scalars(
                select(SensorReading)
                .where(SensorReading.asset_id == asset_id, SensorReading.cycle <= cycle)
                .order_by(SensorReading.cycle)
            )
        )

    def latest(self, asset_id: uuid.UUID) -> SensorReading | None:
        return self.session.scalar(
            select(SensorReading)
            .where(SensorReading.asset_id == asset_id)
            .order_by(desc(SensorReading.cycle))
            .limit(1)
        )

    def count(self) -> int:
        return int(self.session.scalar(select(func.count()).select_from(SensorReading)) or 0)

    def latest_ingested_at(self) -> datetime | None:
        return self.session.scalar(select(func.max(SensorReading.ingested_at)))


class PredictionRepository:
    """Model-version-pinned prediction history with exact-replay idempotency."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, command: NewPrediction) -> Prediction:
        reading = self.session.get(SensorReading, command.sensor_reading_id)
        if (
            reading is None
            or reading.asset_id != command.asset_id
            or reading.cycle != command.cycle
        ):
            raise ValueError("Prediction asset and cycle must match its sensor reading.")
        values = asdict(command)
        statement = (
            insert(Prediction).values(**values).on_conflict_do_nothing().returning(Prediction.id)
        )
        inserted_id = self.session.scalar(statement)
        if inserted_id is not None:
            prediction = self.session.get(Prediction, inserted_id)
            assert prediction is not None
            return prediction
        existing = self.session.scalar(
            select(Prediction).where(
                Prediction.sensor_reading_id == command.sensor_reading_id,
                Prediction.model_name == command.model_name,
                Prediction.model_version == command.model_version,
            )
        )
        if existing is None or not _same_prediction(existing, command):
            raise PredictionConflictError(
                "The model version already has a different prediction for this reading."
            )
        return existing

    def for_asset(self, asset_id: uuid.UUID, *, limit: int = 1000) -> list[Prediction]:
        return list(
            self.session.scalars(
                select(Prediction)
                .where(Prediction.asset_id == asset_id)
                .order_by(Prediction.prediction_timestamp, Prediction.created_at)
                .limit(limit)
            )
        )

    def latest(self, asset_id: uuid.UUID) -> Prediction | None:
        return self.session.scalar(
            select(Prediction)
            .where(Prediction.asset_id == asset_id)
            .order_by(desc(Prediction.prediction_timestamp), desc(Prediction.created_at))
            .limit(1)
        )

    def get_for_model(
        self, sensor_reading_id: uuid.UUID, model_name: str, model_version: str
    ) -> Prediction | None:
        return self.session.scalar(
            select(Prediction).where(
                Prediction.sensor_reading_id == sensor_reading_id,
                Prediction.model_name == model_name,
                Prediction.model_version == model_version,
            )
        )

    def recent(
        self,
        *,
        since: datetime | None = None,
        asset_id: uuid.UUID | None = None,
        limit: int = 100,
    ) -> list[Prediction]:
        if limit <= 0:
            raise ValueError("limit must be positive.")
        query: Select[tuple[Prediction]] = select(Prediction)
        if since is not None:
            query = query.where(Prediction.prediction_timestamp >= since)
        if asset_id is not None:
            query = query.where(Prediction.asset_id == asset_id)
        return list(
            self.session.scalars(query.order_by(desc(Prediction.prediction_timestamp)).limit(limit))
        )

    def count(self) -> int:
        return int(self.session.scalar(select(func.count()).select_from(Prediction)) or 0)

    def risk_distribution(self, *, limit: int = 1000) -> dict[str, int]:
        recent = (
            select(Prediction.risk_level)
            .order_by(desc(Prediction.prediction_timestamp))
            .limit(limit)
            .subquery()
        )
        rows = self.session.execute(
            select(recent.c.risk_level, func.count()).group_by(recent.c.risk_level)
        )
        return {str(risk): int(count) for risk, count in rows}


class MaintenanceEventRepository:
    """Append operational events, deduplicating only explicit external identifiers."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, command: NewMaintenanceEvent) -> MaintenanceEvent:
        values = asdict(command)
        values["event_metadata"] = values.pop("metadata")
        if command.external_event_id is not None:
            inserted_id = self.session.scalar(
                insert(MaintenanceEvent)
                .values(**values)
                .on_conflict_do_nothing()
                .returning(MaintenanceEvent.id)
            )
            if inserted_id is not None:
                inserted = self.session.get(MaintenanceEvent, inserted_id)
                assert inserted is not None
                return inserted
            existing = self.session.scalar(
                select(MaintenanceEvent).where(
                    MaintenanceEvent.external_event_id == command.external_event_id
                )
            )
            if existing is not None and _same_event(existing, command):
                return existing
            raise DuplicateExternalIdError(
                f"External event ID {command.external_event_id!r} has conflicting data."
            )
        event = MaintenanceEvent(**values)
        self.session.add(event)
        self.session.flush()
        return event

    def for_asset(self, asset_id: uuid.UUID) -> list[MaintenanceEvent]:
        return list(
            self.session.scalars(
                select(MaintenanceEvent)
                .where(MaintenanceEvent.asset_id == asset_id)
                .order_by(MaintenanceEvent.occurred_at, MaintenanceEvent.created_at)
            )
        )

    def latest_failure_or_maintenance(self, asset_id: uuid.UUID) -> MaintenanceEvent | None:
        return self.session.scalar(
            select(MaintenanceEvent)
            .where(
                MaintenanceEvent.asset_id == asset_id,
                MaintenanceEvent.event_type.in_(
                    [
                        MaintenanceEventType.FAILURE,
                        MaintenanceEventType.PLANNED_MAINTENANCE,
                    ]
                ),
            )
            .order_by(desc(MaintenanceEvent.occurred_at), desc(MaintenanceEvent.created_at))
            .limit(1)
        )


class ModelEvaluationRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, command: NewModelEvaluation) -> ModelEvaluation:
        evaluation = ModelEvaluation(**asdict(command))
        self.session.add(evaluation)
        self.session.flush()
        return evaluation

    def for_model(self, model_name: str, model_version: str) -> list[ModelEvaluation]:
        return list(
            self.session.scalars(
                select(ModelEvaluation)
                .where(
                    ModelEvaluation.model_name == model_name,
                    ModelEvaluation.model_version == model_version,
                )
                .order_by(desc(ModelEvaluation.created_at))
            )
        )

    def for_replay_run(self, replay_run_id: uuid.UUID) -> list[ModelEvaluation]:
        """Evaluations whose JSONB metrics reference one replay run."""
        return list(
            self.session.scalars(
                select(ModelEvaluation)
                .where(ModelEvaluation.metrics.contains({"replay_run_id": str(replay_run_id)}))
                .order_by(desc(ModelEvaluation.created_at))
            )
        )


class DriftReportRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, command: NewDriftReport) -> DriftReport:
        report = DriftReport(**asdict(command))
        self.session.add(report)
        self.session.flush()
        return report

    def for_model(self, model_name: str, model_version: str) -> list[DriftReport]:
        return list(
            self.session.scalars(
                select(DriftReport)
                .where(
                    DriftReport.model_name == model_name,
                    DriftReport.model_version == model_version,
                )
                .order_by(desc(DriftReport.window_end))
            )
        )


class ReplayRunRepository:
    """Durable replay progress rows with row-level locking for advancement."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, command: NewReplayRun) -> ReplayRun:
        values = asdict(command)
        values["run_metadata"] = values.pop("metadata")
        run = ReplayRun(**values)
        try:
            with self.session.begin_nested():
                self.session.add(run)
                self.session.flush()
        except IntegrityError as exc:
            raise ReplayRunConflictError(
                f"A replay run for {command.dataset_subset} asset "
                f"{command.source_asset_id} attempt {command.attempt} or external asset "
                f"{command.external_asset_id!r} already exists."
            ) from exc
        return run

    def get(self, run_id: uuid.UUID) -> ReplayRun | None:
        return self.session.get(ReplayRun, run_id)

    def get_for_update(self, run_id: uuid.UUID) -> ReplayRun | None:
        """Lock one run row so competing workers serialize on it briefly."""
        return self.session.scalar(
            select(ReplayRun).where(ReplayRun.id == run_id).with_for_update()
        )

    def latest_for_source(
        self, dataset_name: str, dataset_subset: str, source_asset_id: int
    ) -> ReplayRun | None:
        return self.session.scalar(
            select(ReplayRun)
            .where(
                ReplayRun.dataset_name == dataset_name,
                ReplayRun.dataset_subset == dataset_subset,
                ReplayRun.source_asset_id == source_asset_id,
            )
            .order_by(desc(ReplayRun.attempt))
            .limit(1)
        )

    def list_runs(self, *, limit: int = 100) -> list[ReplayRun]:
        if limit <= 0:
            raise ValueError("limit must be positive.")
        return list(
            self.session.scalars(
                select(ReplayRun).order_by(desc(ReplayRun.created_at)).limit(limit)
            )
        )


class PredictionOutcomeRepository:
    """Idempotent realized-label storage keyed by prediction and outcome event."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, command: NewPredictionOutcome) -> PredictionOutcome:
        values = asdict(command)
        statement = (
            insert(PredictionOutcome)
            .values(**values)
            .on_conflict_do_nothing()
            .returning(PredictionOutcome.id)
        )
        inserted_id = self.session.scalar(statement)
        if inserted_id is not None:
            outcome = self.session.get(PredictionOutcome, inserted_id)
            assert outcome is not None
            return outcome
        existing = self.session.scalar(
            select(PredictionOutcome).where(
                PredictionOutcome.prediction_id == command.prediction_id,
                PredictionOutcome.maintenance_event_id == command.maintenance_event_id,
            )
        )
        if existing is None or not _same_outcome(existing, command):
            raise PredictionOutcomeConflictError(
                f"Prediction {command.prediction_id} already has a different realized "
                "label for this outcome event."
            )
        return existing

    def for_event(self, maintenance_event_id: uuid.UUID) -> list[PredictionOutcome]:
        return list(
            self.session.scalars(
                select(PredictionOutcome)
                .where(PredictionOutcome.maintenance_event_id == maintenance_event_id)
                .order_by(PredictionOutcome.cycle)
            )
        )

    def for_asset(self, asset_id: uuid.UUID) -> list[PredictionOutcome]:
        return list(
            self.session.scalars(
                select(PredictionOutcome)
                .where(PredictionOutcome.asset_id == asset_id)
                .order_by(PredictionOutcome.cycle)
            )
        )


class PipelineRunRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, command: NewPipelineRun) -> PipelineRun:
        values = asdict(command)
        values["run_metadata"] = values.pop("metadata")
        run = PipelineRun(**values)
        self.session.add(run)
        self.session.flush()
        return run

    def get(self, run_id: uuid.UUID) -> PipelineRun | None:
        return self.session.get(PipelineRun, run_id)

    def finish(
        self,
        run: PipelineRun,
        *,
        status: PipelineRunStatus,
        finished_at: datetime,
        error_message: str | None = None,
        output_manifest_checksum: str | None = None,
    ) -> PipelineRun:
        terminal = {
            PipelineRunStatus.SUCCEEDED,
            PipelineRunStatus.FAILED,
            PipelineRunStatus.CANCELLED,
        }
        if status not in terminal or finished_at < run.started_at:
            raise ValueError("A finished pipeline run requires a terminal status and valid time.")
        if status is PipelineRunStatus.FAILED and not error_message:
            raise ValueError("A failed pipeline run requires an actionable error message.")
        run.status = status
        run.finished_at = finished_at
        run.error_message = error_message
        run.output_manifest_checksum = output_manifest_checksum
        self.session.flush()
        return run

    def recent(self, *, limit: int = 100) -> list[PipelineRun]:
        return list(
            self.session.scalars(
                select(PipelineRun).order_by(desc(PipelineRun.started_at)).limit(limit)
            )
        )


def _sensor_values(command: NewSensorReading) -> dict[str, object]:
    values: dict[str, object] = {
        "asset_id": command.asset_id,
        "cycle": command.cycle,
        "observed_at": command.observed_at,
        "schema_version": command.schema_version,
        "source": command.source,
        "ingestion_id": command.ingestion_id,
    }
    values.update(dict(zip(OPERATING_SETTING_COLUMNS, command.operating_settings, strict=True)))
    values.update(dict(zip(SENSOR_COLUMNS, command.sensor_values, strict=True)))
    return values


def _same_sensor(reading: SensorReading, command: NewSensorReading) -> bool:
    return (
        reading.observed_at == command.observed_at
        and tuple(getattr(reading, name) for name in OPERATING_SETTING_COLUMNS)
        == command.operating_settings
        and tuple(getattr(reading, name) for name in SENSOR_COLUMNS) == command.sensor_values
        and reading.schema_version == command.schema_version
        and reading.source == command.source
        and reading.ingestion_id == command.ingestion_id
    )


def _same_prediction(prediction: Prediction, command: NewPrediction) -> bool:
    fields = asdict(command)
    return all(getattr(prediction, name) == value for name, value in fields.items())


def _same_event(event: MaintenanceEvent, command: NewMaintenanceEvent) -> bool:
    values = asdict(command)
    values["event_metadata"] = values.pop("metadata")
    return all(getattr(event, name) == value for name, value in values.items())


def _same_outcome(outcome: PredictionOutcome, command: NewPredictionOutcome) -> bool:
    """Label content equality; ``labeled_at`` is retry bookkeeping, not content."""
    return (
        outcome.prediction_id == command.prediction_id
        and outcome.maintenance_event_id == command.maintenance_event_id
        and outcome.asset_id == command.asset_id
        and outcome.cycle == command.cycle
        and outcome.realized_rul == command.realized_rul
    )

"""Operational feature and delayed-label assembly using the shared Loop 3 builder."""

import uuid
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from turbine_guard.data.schema import (
    ASSET_ID_COLUMN,
    CYCLE_COLUMN,
    OPERATING_SETTING_COLUMNS,
    SENSOR_COLUMNS,
)
from turbine_guard.database.enums import LifecycleAssetRole, PipelineRunStatus, ReplayRunStatus
from turbine_guard.database.models import (
    LifecycleAssetAssignment,
    PipelineRun,
    Prediction,
    PredictionOutcome,
    ReplayRun,
    SensorReading,
)
from turbine_guard.features.builder import FeatureBuilder
from turbine_guard.features.config import FeatureConfig

_SOURCE_COLUMNS = (*OPERATING_SETTING_COLUMNS, *SENSOR_COLUMNS)


@dataclass(frozen=True)
class LabeledAssetData:
    asset_id: uuid.UUID
    source_asset_id: int
    external_asset_id: str
    labeled_at: datetime
    frame: pd.DataFrame

    @property
    def row_count(self) -> int:
        return len(self.frame)


def sensor_window(session: Session, start: datetime, end: datetime) -> pd.DataFrame:
    """Load accepted raw readings in ingestion order for one half-open UTC window."""
    readings = list(
        session.scalars(
            select(SensorReading)
            .where(SensorReading.ingested_at >= start, SensorReading.ingested_at < end)
            .order_by(SensorReading.ingested_at, SensorReading.asset_id, SensorReading.cycle)
        )
    )
    return _sensor_frame(readings)


def feature_window(
    session: Session,
    start: datetime,
    end: datetime,
    feature_config: FeatureConfig,
) -> pd.DataFrame:
    """Reconstruct window features with each selected asset's complete prior history."""
    window = list(
        session.scalars(
            select(SensorReading).where(
                SensorReading.ingested_at >= start, SensorReading.ingested_at < end
            )
        )
    )
    builder = FeatureBuilder(feature_config)
    if not window:
        return pd.DataFrame(columns=list(builder.feature_columns()), dtype="float64")
    keys = {(reading.asset_id, reading.cycle) for reading in window}
    asset_ids = sorted({reading.asset_id for reading in window}, key=str)
    history = list(
        session.scalars(
            select(SensorReading)
            .where(SensorReading.asset_id.in_(asset_ids))
            .order_by(SensorReading.asset_id, SensorReading.cycle)
        )
    )
    generated = builder.transform(_sensor_frame(history))
    selected = generated[
        [
            (asset_id, int(cycle)) in keys
            for asset_id, cycle in zip(
                generated[ASSET_ID_COLUMN], generated[CYCLE_COLUMN], strict=True
            )
        ]
    ]
    return selected.loc[:, list(builder.feature_columns())].reset_index(drop=True)


def completed_labeled_assets(
    session: Session,
    feature_config: FeatureConfig,
    *,
    labeled_after: datetime | None = None,
) -> list[LabeledAssetData]:
    """Return completed assets whose every accepted cycle has one consistent realized label."""
    runs = list(
        session.scalars(
            select(ReplayRun)
            .where(ReplayRun.status == ReplayRunStatus.COMPLETED, ReplayRun.asset_id.is_not(None))
            .order_by(ReplayRun.source_asset_id, ReplayRun.attempt)
        )
    )
    builder = FeatureBuilder(feature_config)
    assets: list[LabeledAssetData] = []
    for run in runs:
        assert run.asset_id is not None
        readings = list(
            session.scalars(
                select(SensorReading)
                .where(SensorReading.asset_id == run.asset_id)
                .order_by(SensorReading.cycle)
            )
        )
        outcome_rows = session.execute(
            select(
                Prediction.cycle,
                PredictionOutcome.realized_rul,
                PredictionOutcome.labeled_at,
            )
            .join(Prediction, Prediction.id == PredictionOutcome.prediction_id)
            .where(PredictionOutcome.asset_id == run.asset_id)
            .order_by(Prediction.cycle)
        ).all()
        labels: dict[int, int] = {}
        labeled_times: list[datetime] = []
        for cycle, realized_rul, labeled_at in outcome_rows:
            previous = labels.setdefault(int(cycle), int(realized_rul))
            if previous != int(realized_rul):
                raise ValueError(f"Asset {run.asset_id} has conflicting delayed labels.")
            labeled_times.append(labeled_at)
        if not readings or set(labels) != {reading.cycle for reading in readings}:
            continue
        latest_label = max(labeled_times)
        if labeled_after is not None and latest_label <= labeled_after:
            continue
        raw = _sensor_frame(readings)
        raw[ASSET_ID_COLUMN] = run.source_asset_id
        generated = builder.transform(raw)
        generated["rul"] = generated[CYCLE_COLUMN].map(labels).astype("float64")
        assets.append(
            LabeledAssetData(
                asset_id=run.asset_id,
                source_asset_id=run.source_asset_id,
                external_asset_id=run.external_asset_id,
                labeled_at=latest_label,
                frame=generated,
            )
        )
    return assets


def delayed_model_frame(
    session: Session,
    *,
    start: datetime,
    end: datetime,
    model_name: str,
    model_version: str,
) -> pd.DataFrame:
    """Join predictions to labels that became available in the monitoring window."""
    rows = session.execute(
        select(Prediction, PredictionOutcome)
        .join(PredictionOutcome, PredictionOutcome.prediction_id == Prediction.id)
        .where(
            Prediction.model_name == model_name,
            Prediction.model_version == model_version,
            PredictionOutcome.labeled_at >= start,
            PredictionOutcome.labeled_at < end,
        )
        .order_by(Prediction.asset_id, Prediction.cycle)
    ).all()
    asset_ids = sorted({prediction.asset_id for prediction, _ in rows}, key=str)
    evaluation_ids = {asset_id: index for index, asset_id in enumerate(asset_ids, start=1)}
    records = [
        {
            "asset_id": evaluation_ids[prediction.asset_id],
            "cycle": prediction.cycle,
            "y_true": float(outcome.realized_rul),
            "y_true_uncapped": float(outcome.realized_rul),
            "y_pred": prediction.predicted_rul,
            "lower": prediction.lower_rul,
            "upper": prediction.upper_rul,
            "model_name": prediction.model_name,
            "model_version": prediction.model_version,
            "model_run_id": prediction.model_run_id,
            "prediction_timestamp": prediction.prediction_timestamp,
        }
        for prediction, outcome in rows
    ]
    return pd.DataFrame(records)


def previously_trained_asset_ids(session: Session) -> set[uuid.UUID]:
    """Assets incorporated by a successfully promoted prior lifecycle."""
    return set(
        session.scalars(
            select(LifecycleAssetAssignment.asset_id)
            .join(PipelineRun, PipelineRun.id == LifecycleAssetAssignment.pipeline_run_id)
            .where(
                LifecycleAssetAssignment.role == LifecycleAssetRole.RETRAINING_ADDITION,
                PipelineRun.status == PipelineRunStatus.SUCCEEDED,
            )
        )
    )


def _sensor_frame(readings: list[SensorReading]) -> pd.DataFrame:
    columns = [ASSET_ID_COLUMN, CYCLE_COLUMN, *_SOURCE_COLUMNS, "ingested_at"]
    rows = [
        {
            ASSET_ID_COLUMN: reading.asset_id,
            CYCLE_COLUMN: reading.cycle,
            **{column: float(getattr(reading, column)) for column in _SOURCE_COLUMNS},
            "ingested_at": reading.ingested_at,
        }
        for reading in readings
    ]
    return pd.DataFrame(rows, columns=columns)

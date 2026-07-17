"""Shared champion-serving contracts usable without importing MLflow.

The online API, dashboard, and inference services depend only on these
types, so a deployment can serve either the live MLflow registry champion
or a checksum-verified exported deployment bundle through one interface.
"""

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

import pandas as pd

from turbine_guard.features.config import FeatureConfig

LIVE_REGISTRY_SOURCE = "live_registry"
EXPORTED_SNAPSHOT_SOURCE = "exported_snapshot"


class PyFuncModel(Protocol):
    """Minimal model behavior required by the inference service."""

    metadata: Any

    def predict(self, model_input: pd.DataFrame) -> pd.DataFrame: ...


@dataclass(frozen=True)
class ModelMetadata:
    """Registry and evaluation identity exposed by the online API."""

    model_name: str
    version: str
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
    loaded_at: datetime
    checksum: str | None
    lineage_id: str | None
    model_family: str | None = None
    git_sha: str | None = None
    dataset_checksum: str | None = None
    feature_manifest_checksum: str | None = None
    registry_source: str = LIVE_REGISTRY_SOURCE


@dataclass(frozen=True)
class LoadedChampion:
    """Cached model plus the exact shared feature configuration it accepts."""

    model: PyFuncModel
    metadata: ModelMetadata
    feature_config: FeatureConfig
    feature_columns: tuple[str, ...]


class ChampionLoader(Protocol):
    """One champion per process, loaded from a registry or a verified bundle."""

    def get(self) -> LoadedChampion: ...

    def refresh(self) -> LoadedChampion: ...

    def check_model(self) -> bool: ...

    def check_feature_contract(self) -> bool: ...

    def registry_aliases(self) -> dict[str, str]: ...


def validate_prediction_output(frame: pd.DataFrame) -> tuple[float, float, float, str]:
    """Validate one rich pyfunc output without trusting registry artifacts blindly."""
    required = ("predicted_rul", "lower_rul", "upper_rul", "risk_level")
    if len(frame) != 1 or tuple(frame.columns) != required:
        raise ValueError("Champion returned an invalid output schema.")
    point = float(frame.iloc[0]["predicted_rul"])
    lower = float(frame.iloc[0]["lower_rul"])
    upper = float(frame.iloc[0]["upper_rul"])
    risk = str(frame.iloc[0]["risk_level"])
    if not all(math.isfinite(value) for value in (point, lower, upper)):
        raise ValueError("Champion returned non-finite RUL output.")
    if not 0 <= lower <= point <= upper:
        raise ValueError("Champion returned invalid RUL interval ordering.")
    if risk not in {"healthy", "warning", "critical"}:
        raise ValueError("Champion returned an invalid risk level.")
    return point, lower, upper, risk

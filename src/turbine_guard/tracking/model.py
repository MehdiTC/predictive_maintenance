"""MLflow pyfunc packaging for the existing Loop 4 champion bundle."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from mlflow.models import ModelSignature
from mlflow.pyfunc.model import PythonModel, PythonModelContext
from mlflow.types import ColSpec, DataType, Schema

from turbine_guard.modeling.artifacts import load_joblib, sha256_path
from turbine_guard.modeling.pipeline import ModelBundle

POINT_COLUMN = "predicted_rul"
LOWER_COLUMN = "lower_rul"
UPPER_COLUMN = "upper_rul"
RISK_COLUMN = "risk_level"


class ChampionPyFuncModel(PythonModel):
    """Expose point, interval, and risk outputs by delegating to ``ModelBundle``."""

    def __init__(self, feature_columns: tuple[str, ...], bundle_sha256: str) -> None:
        self.feature_columns = feature_columns
        self.bundle_sha256 = bundle_sha256
        self._bundle: ModelBundle | None = None

    def load_context(self, context: PythonModelContext) -> None:
        """Checksum and load the copied trusted champion artifact."""
        path = Path(context.artifacts["champion_bundle"])
        actual = sha256_path(path)
        if actual != self.bundle_sha256:
            raise ValueError(
                f"Packaged champion checksum mismatch: expected {self.bundle_sha256}, "
                f"found {actual}."
            )
        value = load_joblib(path)
        if not isinstance(value, ModelBundle):
            raise TypeError("Packaged champion is not a TurbineGuard ModelBundle.")
        if value.feature_columns != self.feature_columns:
            raise ValueError("Packaged champion feature contract changed.")
        self._bundle = value

    def predict(
        self,
        context: PythonModelContext,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Return rich RUL output; MLflow applies the declared input schema first."""
        del context, params
        if self._bundle is None:
            raise ValueError("Champion model has not been loaded by MLflow.")
        if tuple(model_input.columns) != self.feature_columns:
            raise ValueError("Prediction columns do not match the ordered Loop 3 feature manifest.")
        point = self._bundle.predict(model_input)
        lower, upper = self._bundle.predict_interval(model_input)
        risk = np.where(
            point <= self._bundle.critical_horizon,
            "critical",
            np.where(point <= self._bundle.warning_horizon, "warning", "healthy"),
        )
        return pd.DataFrame(
            {
                POINT_COLUMN: point,
                LOWER_COLUMN: lower,
                UPPER_COLUMN: upper,
                RISK_COLUMN: risk,
            },
            index=model_input.index,
        )


def champion_signature(feature_columns: tuple[str, ...]) -> ModelSignature:
    """Explicit named schema excluding all identifiers, metadata, and targets."""
    return ModelSignature(
        inputs=Schema(
            [ColSpec(type=DataType.double, name=name, required=True) for name in feature_columns]
        ),
        outputs=Schema(
            [
                ColSpec(type=DataType.double, name=POINT_COLUMN),
                ColSpec(type=DataType.double, name=LOWER_COLUMN),
                ColSpec(type=DataType.double, name=UPPER_COLUMN),
                ColSpec(type=DataType.string, name=RISK_COLUMN),
            ]
        ),
    )

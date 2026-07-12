"""Leakage-safe point-model pipelines for the bounded Loop 4 candidate set."""

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from turbine_guard.modeling.config import CandidateConfig, ModelKind, TrainingConfig


class MedianRulRegressor(RegressorMixin, BaseEstimator):
    """Constant baseline fitted to the training target median only."""

    def __init__(self) -> None:
        self.constant_: float | None = None

    def fit(self, features: Any, target: Any) -> "MedianRulRegressor":
        """Store the training-target median; feature values are intentionally unused."""
        del features
        values = np.asarray(target, dtype="float64")
        if values.size == 0 or not bool(np.isfinite(values).all()):
            raise ValueError("Median baseline requires a non-empty finite target.")
        self.constant_ = float(np.median(values))
        return self

    def predict(self, features: Any) -> np.ndarray:
        """Return the learned constant for each input row."""
        if self.constant_ is None:
            raise ValueError("MedianRulRegressor has not been fitted.")
        return np.full(len(features), self.constant_, dtype="float64")


def build_pipeline(candidate: CandidateConfig, config: TrainingConfig) -> Pipeline:
    """Construct an unfitted candidate pipeline with explicit preprocessing policy."""
    params = candidate.params
    if candidate.kind is ModelKind.CONSTANT:
        return Pipeline((("model", MedianRulRegressor()),))
    if candidate.kind is ModelKind.RIDGE:
        steps: list[tuple[str, Any]] = [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            )
        ]
        if config.scale_ridge:
            steps.append(("scaler", StandardScaler()))
        steps.append(("model", Ridge(alpha=float(params.get("alpha", 1.0)))))
        return Pipeline(steps)
    if candidate.kind is ModelKind.HIST_GRADIENT_BOOSTING:
        hist_model = HistGradientBoostingRegressor(
            learning_rate=float(params.get("learning_rate", 0.1)),
            max_iter=int(params.get("max_iter", 100)),
            max_leaf_nodes=int(params.get("max_leaf_nodes", 31)),
            random_state=config.random_seed,
            early_stopping=False,
        )
        return Pipeline((("model", hist_model),))
    if candidate.kind is ModelKind.XGBOOST:
        xgb_model = XGBRegressor(
            learning_rate=float(params.get("learning_rate", 0.1)),
            max_depth=int(params.get("max_depth", 3)),
            n_estimators=int(params.get("n_estimators", 100)),
            subsample=float(params.get("subsample", 1.0)),
            colsample_bytree=float(params.get("colsample_bytree", 1.0)),
            objective="reg:squarederror",
            random_state=config.random_seed,
            n_jobs=1,
            tree_method="hist",
            missing=np.nan,
            verbosity=0,
        )
        return Pipeline((("model", xgb_model),))
    raise ValueError(f"Unsupported model kind: {candidate.kind}.")


def preprocessing_policy(candidate: CandidateConfig, config: TrainingConfig) -> str:
    """Human- and machine-readable missing-value/scaling policy for a candidate."""
    if candidate.kind is ModelKind.RIDGE:
        scaling = "standard_scaling" if config.scale_ridge else "no_scaling"
        return f"training_median_imputation_with_indicators+{scaling}"
    if candidate.kind in (ModelKind.HIST_GRADIENT_BOOSTING, ModelKind.XGBOOST):
        return "native_missing_value_support_no_scaling"
    return "no_feature_preprocessing"

"""Champion-family retraining and identical-holdout candidate comparison."""

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from turbine_guard.modeling.alerts import alert_metrics
from turbine_guard.modeling.artifacts import serialize_joblib
from turbine_guard.modeling.config import CandidateConfig, ModelKind, TargetConfig, TrainingConfig
from turbine_guard.modeling.conformal import interval_metrics
from turbine_guard.modeling.metrics import regression_metrics
from turbine_guard.modeling.pipeline import (
    ModelBundle,
    fit_candidate_bundle,
    prediction_latency_ms,
)
from turbine_guard.monitoring.config import PromotionThresholds


@dataclass(frozen=True)
class TrainedCandidate:
    bundle: ModelBundle
    candidate_config: CandidateConfig
    target_config: TargetConfig
    artifact_bytes: bytes


@dataclass(frozen=True)
class CandidateComparison:
    holdout_sha256: str
    row_count: int
    asset_count: int
    candidate: dict[str, Any]
    champion: dict[str, Any]
    naive: dict[str, Any]

    def record(self) -> dict[str, Any]:
        return {
            "holdout_sha256": self.holdout_sha256,
            "row_count": self.row_count,
            "asset_count": self.asset_count,
            "candidate": self.candidate,
            "champion": self.champion,
            "naive": self.naive,
        }


@dataclass(frozen=True)
class PromotionGateResult:
    passed: bool
    gates: dict[str, dict[str, Any]]

    @property
    def blocking_failures(self) -> tuple[str, ...]:
        return tuple(name for name, result in self.gates.items() if not bool(result["passed"]))

    def record(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "blocking_failures": list(self.blocking_failures),
            "gates": self.gates,
        }


def champion_candidate_config(
    *, model_family: str, parameters: dict[str, str], candidate_name: str
) -> CandidateConfig:
    """Recover one existing Loop 4 family/configuration from MLflow run parameters."""
    kind = ModelKind(model_family)
    allowed = {
        ModelKind.CONSTANT: set(),
        ModelKind.RIDGE: {"alpha"},
        ModelKind.HIST_GRADIENT_BOOSTING: {
            "learning_rate",
            "max_iter",
            "max_leaf_nodes",
        },
        ModelKind.XGBOOST: {
            "learning_rate",
            "max_depth",
            "n_estimators",
            "subsample",
            "colsample_bytree",
        },
    }[kind]
    parsed = tuple(sorted((name, _parameter_value(parameters[name])) for name in allowed))
    return CandidateConfig(candidate_name, kind, parsed, complexity_rank=0)


def train_candidate(
    *,
    training_frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    champion_bundle: ModelBundle,
    candidate_config: CandidateConfig,
    training_config: TrainingConfig,
    metadata: dict[str, Any],
) -> TrainedCandidate:
    """Retrain the champion family/target through Loop 4's established fit path."""
    target = TargetConfig(champion_bundle.target_name, champion_bundle.target_cap)
    bundle = fit_candidate_bundle(
        training_frame=training_frame,
        feature_columns=feature_columns,
        candidate=candidate_config,
        target=target,
        config=training_config,
        conformal=champion_bundle.conformal,
        metadata=metadata,
    )
    return TrainedCandidate(bundle, candidate_config, target, serialize_joblib(bundle))


def compare_candidate(
    *,
    candidate: TrainedCandidate,
    champion: ModelBundle,
    training_frame: pd.DataFrame,
    holdout_frame: pd.DataFrame,
    training_config: TrainingConfig,
) -> CandidateComparison:
    """Evaluate candidate, frozen champion, and naive baseline on exactly one holdout."""
    columns = candidate.bundle.feature_columns
    features = holdout_frame.loc[:, list(columns)]
    truth_uncapped = holdout_frame["rul"].to_numpy(dtype="float64")
    truth = (
        truth_uncapped
        if candidate.target_config.cap is None
        else np.minimum(truth_uncapped, candidate.target_config.cap)
    )
    candidate_metrics = _bundle_metrics(
        candidate.bundle,
        features,
        holdout_frame,
        truth,
        truth_uncapped,
        artifact_size=len(candidate.artifact_bytes),
        repeats=training_config.latency_repeats,
    )
    champion_metrics = _bundle_metrics(
        champion,
        features,
        holdout_frame,
        truth,
        truth_uncapped,
        artifact_size=len(serialize_joblib(champion)),
        repeats=training_config.latency_repeats,
    )
    naive_config = CandidateConfig("promotion_naive", ModelKind.CONSTANT)
    naive = fit_candidate_bundle(
        training_frame=training_frame,
        feature_columns=columns,
        candidate=naive_config,
        target=candidate.target_config,
        config=training_config,
        conformal=champion.conformal,
        metadata={"purpose": "promotion_naive_baseline"},
    )
    naive_prediction = naive.predict(features)
    naive_metrics = {"regression": regression_metrics(truth, naive_prediction)}
    return CandidateComparison(
        holdout_sha256=_holdout_sha256(holdout_frame, columns),
        row_count=len(holdout_frame),
        asset_count=int(holdout_frame["asset_id"].nunique()),
        candidate=candidate_metrics,
        champion=champion_metrics,
        naive=naive_metrics,
    )


def promotion_gates(
    comparison: CandidateComparison,
    *,
    thresholds: PromotionThresholds,
    data_quality_passes: bool,
    enough_labeled_data: bool,
    artifact_valid: bool,
    reload_equivalence_difference: float | None,
) -> PromotionGateResult:
    """Evaluate every blocking gate explicitly; no weighted score can mask a failure."""
    candidate = comparison.candidate
    champion = comparison.champion
    naive = comparison.naive
    gates = {
        "data_quality": _gate(data_quality_passes, data_quality_passes, True),
        "enough_labeled_data": _gate(enough_labeled_data, enough_labeled_data, True),
        "candidate_artifact_valid": _gate(artifact_valid, artifact_valid, True),
        "beats_naive_baseline": _maximum_gate(
            candidate["regression"]["rmse"], naive["regression"]["rmse"], strict=True
        ),
        "rmse_not_materially_worse": _relative_gate(
            candidate["regression"]["rmse"],
            champion["regression"]["rmse"],
            thresholds.rmse_relative_tolerance,
        ),
        "nasa_not_materially_worse": _relative_gate(
            candidate["regression"]["nasa_score"],
            champion["regression"]["nasa_score"],
            thresholds.nasa_relative_tolerance,
        ),
        "critical_recall": _minimum_gate(
            candidate["critical"]["recall"], thresholds.minimum_critical_recall
        ),
        "false_alarms": _maximum_gate(
            candidate["critical"]["false_alarms_per_1000_cycles"],
            thresholds.maximum_false_alarms_per_1000,
        ),
        "conformal_coverage": _minimum_gate(
            candidate["interval"]["empirical_coverage"], thresholds.minimum_coverage
        ),
        "inference_latency": _maximum_gate(
            candidate["inference_latency_ms"], thresholds.maximum_latency_ms
        ),
        "artifact_size": _maximum_gate(
            candidate["artifact_size_bytes"], thresholds.maximum_artifact_size_bytes
        ),
        "mlflow_reload_equivalence": _maximum_gate(
            reload_equivalence_difference,
            thresholds.equivalence_tolerance,
        ),
    }
    return PromotionGateResult(all(bool(result["passed"]) for result in gates.values()), gates)


def _bundle_metrics(
    bundle: ModelBundle,
    features: pd.DataFrame,
    holdout: pd.DataFrame,
    truth: np.ndarray,
    truth_uncapped: np.ndarray,
    *,
    artifact_size: int,
    repeats: int,
) -> dict[str, Any]:
    point = bundle.predict(features)
    lower, upper = bundle.predict_interval(features)
    evaluation = pd.DataFrame(
        {
            "asset_id": holdout["asset_id"].to_numpy(dtype="int64"),
            "cycle": holdout["cycle"].to_numpy(dtype="int64"),
            "y_true": truth,
            "y_true_uncapped": truth_uncapped,
            "y_pred": point,
        }
    )
    critical = alert_metrics(
        evaluation,
        horizon=bundle.critical_horizon,
        minimum_lead_cycles=training_config_minimum_lead(),
    )
    warning = alert_metrics(
        evaluation,
        horizon=bundle.warning_horizon,
        minimum_lead_cycles=training_config_minimum_lead(),
    )
    return {
        "regression": regression_metrics(truth, point),
        "critical": _trim_alert(critical),
        "warning": _trim_alert(warning),
        "interval": interval_metrics(truth, lower, upper),
        "inference_latency_ms": prediction_latency_ms(bundle, features, repeats=repeats),
        "artifact_size_bytes": artifact_size,
    }


def training_config_minimum_lead() -> int:
    """Loop 4's established default; bundle stores horizons but not this policy scalar."""
    return 1


def _trim_alert(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key != "per_asset"}


def _holdout_sha256(frame: pd.DataFrame, columns: tuple[str, ...]) -> str:
    selected = frame.loc[:, ["asset_id", "cycle", "rul", *columns]]
    values = pd.util.hash_pandas_object(selected, index=False).to_numpy(dtype="uint64")
    return hashlib.sha256(values.tobytes()).hexdigest()


def _parameter_value(value: str) -> int | float | str | bool:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        integer = int(value)
        return integer
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _gate(passed: bool, actual: object, threshold: object) -> dict[str, Any]:
    return {"passed": passed, "actual": actual, "threshold": threshold}


def _minimum_gate(actual: float | None, threshold: float) -> dict[str, Any]:
    return _gate(actual is not None and actual >= threshold, actual, {"minimum": threshold})


def _maximum_gate(
    actual: float | int | None, threshold: float | int, *, strict: bool = False
) -> dict[str, Any]:
    passed = actual is not None and (actual < threshold if strict else actual <= threshold)
    return _gate(passed, actual, {"maximum": threshold, "strict": strict})


def _relative_gate(actual: float, champion: float, tolerance: float) -> dict[str, Any]:
    maximum = champion * (1.0 + tolerance)
    return _gate(
        actual <= maximum,
        actual,
        {"champion": champion, "relative_tolerance": tolerance, "maximum": maximum},
    )

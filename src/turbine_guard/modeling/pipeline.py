"""End-to-end offline Loop 4 training, selection, calibration, and evaluation."""

import copy
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from turbine_guard.data.acquisition import current_git_commit
from turbine_guard.data.schema import ASSET_ID_COLUMN, CYCLE_COLUMN
from turbine_guard.modeling.alerts import alert_metrics
from turbine_guard.modeling.artifacts import (
    ArtifactError,
    TrainingManifest,
    configuration_sha256,
    record_file,
    serialize_joblib,
    sha256_bytes,
    sha256_path,
    verify_existing_run,
    write_bytes,
    write_json,
    write_training_manifest,
)
from turbine_guard.modeling.config import (
    CandidateConfig,
    ModelKind,
    TargetConfig,
    TrainingConfig,
    config_record,
)
from turbine_guard.modeling.conformal import (
    SplitConformalCalibrator,
    interval_metrics,
    lifecycle_stages,
)
from turbine_guard.modeling.data import (
    DatasetRole,
    ModelDataError,
    VerifiedModelData,
    load_verified_model_data,
    model_matrix,
    official_final_rows,
    target_values,
)
from turbine_guard.modeling.estimators import build_pipeline, preprocessing_policy
from turbine_guard.modeling.metrics import (
    common_domain_metrics,
    evaluate_regression_frame,
)
from turbine_guard.modeling.reporting import (
    write_candidate_comparison,
    write_human_report,
    write_report_set,
)
from turbine_guard.modeling.selection import ChampionSelectionError, select_champion
from turbine_guard.modeling.simulation import simulate_maintenance_policies

logger = logging.getLogger(__name__)


class TrainingStatus(StrEnum):
    """Outcome of one training command."""

    TRAINED = "trained"
    ALREADY_TRAINED = "already_trained"


class TrainingError(RuntimeError):
    """Raised when Loop 4 cannot produce a complete trustworthy run."""


@dataclass(frozen=True)
class ModelBundle:
    """Reloadable champion point model, preprocessing, conformal state, and contract."""

    pipeline: Pipeline
    feature_columns: tuple[str, ...]
    target_name: str
    target_cap: int | None
    critical_horizon: int
    warning_horizon: int
    conformal: SplitConformalCalibrator
    metadata: dict[str, Any]

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict non-negative RUL in the selected target's domain."""
        if tuple(features.columns) != self.feature_columns:
            raise ValueError("Prediction feature order does not match the saved model contract.")
        raw_prediction = self.pipeline.predict(features)
        if isinstance(raw_prediction, tuple):
            raise ValueError("Regressor returned an unexpected tuple prediction.")
        prediction = cast(
            np.ndarray[Any, np.dtype[np.float64]],
            np.maximum(0.0, np.asarray(raw_prediction, dtype="float64")),
        )
        if self.target_cap is not None:
            prediction = cast(
                np.ndarray[Any, np.dtype[np.float64]],
                np.minimum(prediction, self.target_cap),
            )
        return prediction

    def predict_interval(self, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Predict conformal lower/upper bounds in the selected target's domain."""
        prediction = self.predict(features)
        lower, upper = self.conformal.intervals(prediction)
        if self.target_cap is not None:
            upper = np.minimum(upper, self.target_cap)
        return lower, upper


@dataclass(frozen=True)
class TrainingResult:
    """Outcome and key paths from :func:`train_models`."""

    status: TrainingStatus
    selected_candidate_id: str
    artifacts_dir: Path
    champion_path: Path
    champion_selection_path: Path
    summary: dict[str, Any]


@dataclass
class _CandidateRun:
    candidate: CandidateConfig
    target: TargetConfig
    candidate_id: str
    pipeline: Pipeline
    model_bytes: bytes
    validation_frame: pd.DataFrame
    report: dict[str, Any]


def fit_candidate_bundle(
    *,
    training_frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    candidate: CandidateConfig,
    target: TargetConfig,
    config: TrainingConfig,
    conformal: SplitConformalCalibrator,
    metadata: dict[str, Any],
) -> ModelBundle:
    """Fit one established Loop 4 candidate on an explicitly supplied safe training frame.

    Loop 9 owns asset eligibility and role isolation; this function remains the
    single point-model/preprocessing/bundle construction path.
    """
    features = model_matrix(training_frame, feature_columns)
    truth = target_values(training_frame, target.cap)
    pipeline = build_pipeline(candidate, config)
    pipeline.fit(features, truth)
    return ModelBundle(
        pipeline=pipeline,
        feature_columns=feature_columns,
        target_name=target.name,
        target_cap=target.cap,
        critical_horizon=config.alerts.critical_horizon,
        warning_horizon=config.alerts.warning_horizon,
        conformal=copy.deepcopy(conformal),
        metadata=metadata,
    )


def prediction_latency_ms(
    bundle: ModelBundle, features: pd.DataFrame, *, repeats: int = 5
) -> float:
    """Measure median per-row bundle inference latency using Loop 4 semantics."""
    if repeats < 1 or features.empty:
        raise ValueError("Latency measurement requires data and at least one repeat.")
    batch = features.iloc[: min(512, len(features))]
    bundle.predict(batch)
    timings: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        bundle.predict(batch)
        timings.append(1000.0 * (time.perf_counter() - started) / len(batch))
    return float(np.median(timings))


def train_models(config: TrainingConfig) -> TrainingResult:
    """Run the complete Loop 4 boundary without any Loop 5 tracking/registry work."""
    try:
        data = load_verified_model_data(config)
        existing = (
            None
            if config.force
            else verify_existing_run(
                config,
                dataset_manifest_sha256=data.feature_manifest.source_report_sha256,
                feature_manifest_sha256=data.feature_manifest_sha256,
                split_manifest_sha256=data.split_manifest_sha256,
                input_checksums=data.input_checksums,
            )
        )
        if existing is not None:
            selected = existing.manifest.selected_candidate_id
            return TrainingResult(
                status=TrainingStatus.ALREADY_TRAINED,
                selected_candidate_id=selected,
                artifacts_dir=config.artifacts_dir,
                champion_path=config.artifacts_dir / "models" / "champion.joblib",
                champion_selection_path=(
                    config.artifacts_dir / "reports" / "champion_selection.json"
                ),
                summary={"selected_candidate_id": selected, "idempotent": True},
            )
        return _train_fresh(config, data)
    except (
        ArtifactError,
        ChampionSelectionError,
        ModelDataError,
        OSError,
        ValueError,
    ) as exc:
        raise TrainingError(str(exc)) from exc


def _train_fresh(config: TrainingConfig, data: VerifiedModelData) -> TrainingResult:
    created_at = datetime.now(UTC)
    output_root = config.artifacts_dir
    reports_dir = output_root / "reports"
    models_dir = output_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    if config.force:
        (output_root / "training_manifest.json").unlink(missing_ok=True)

    train_frame = data.frame(DatasetRole.TRAIN)
    validation_frame = data.frame(DatasetRole.VALIDATION)
    features = data.feature_columns
    x_train = model_matrix(train_frame, features)
    x_validation = model_matrix(validation_frame, features)
    comparison_cap = min(target.cap for target in config.targets if target.cap is not None)

    runs: list[_CandidateRun] = []
    model_paths: list[Path] = []
    for target in config.targets:
        y_train = target_values(train_frame, target.cap)
        for candidate in config.candidates:
            candidate_id = f"{target.name}--{candidate.name}"
            pipeline = build_pipeline(candidate, config)
            started = time.perf_counter()
            pipeline.fit(x_train, y_train)
            training_seconds = time.perf_counter() - started
            prediction = _bounded_prediction(pipeline, x_validation, target.cap)
            predicted_frame = _prediction_frame(validation_frame, target, prediction)
            model_bytes = serialize_joblib(pipeline)
            latency_ms = _prediction_latency_ms(
                pipeline,
                x_validation,
                target.cap,
                repeats=config.latency_repeats,
            )
            critical = alert_metrics(
                predicted_frame,
                horizon=config.alerts.critical_horizon,
                minimum_lead_cycles=config.alerts.minimum_lead_cycles,
            )
            warning = alert_metrics(
                predicted_frame,
                horizon=config.alerts.warning_horizon,
                minimum_lead_cycles=config.alerts.minimum_lead_cycles,
            )
            report: dict[str, Any] = {
                "candidate_id": candidate_id,
                "candidate_name": candidate.name,
                "model_kind": candidate.kind.value,
                "configuration": candidate.params,
                "target_definition": {"name": target.name, "cap": target.cap},
                "evaluation_target": target.name,
                "common_comparison_domain": f"uncapped_rul <= {comparison_cap}",
                "preprocessing_policy": preprocessing_policy(candidate, config),
                "complexity_rank": candidate.complexity_rank,
                "random_seed": config.random_seed,
                "regression_metrics": evaluate_regression_frame(predicted_frame),
                "common_domain_metrics": common_domain_metrics(predicted_frame, comparison_cap),
                "critical_alert_metrics": critical,
                "warning_alert_metrics": warning,
                "alert_thresholds": {
                    "critical_horizon": config.alerts.critical_horizon,
                    "warning_horizon": config.alerts.warning_horizon,
                    "score_threshold_tuned": False,
                },
                "training_seconds": training_seconds,
                "prediction_latency_ms": latency_ms,
                "latency_unit": "median milliseconds per row in a fixed validation batch",
                "model_size_bytes": len(model_bytes),
            }
            model_path = models_dir / f"{candidate_id}.joblib"
            write_bytes(model_path, model_bytes)
            model_paths.append(model_path)
            runs.append(
                _CandidateRun(
                    candidate=candidate,
                    target=target,
                    candidate_id=candidate_id,
                    pipeline=pipeline,
                    model_bytes=model_bytes,
                    validation_frame=predicted_frame,
                    report=report,
                )
            )
            logger.info(
                "candidate_evaluated",
                extra={
                    "candidate_id": candidate_id,
                    "validation_rmse": report["common_domain_metrics"]["rmse"],
                    "critical_recall": critical["recall"],
                },
            )

    selection = select_champion([run.report for run in runs], config.selection)
    selected = next(run for run in runs if run.candidate_id == selection.selected_candidate_id)
    selection_artifact = {
        **selection.artifact,
        "evaluation_version": config.evaluation_version,
        "created_at": created_at.isoformat(),
        "git_commit_sha": current_git_commit(),
        "dataset_manifest_sha256": data.feature_manifest.source_report_sha256,
        "feature_manifest_sha256": data.feature_manifest_sha256,
        "split_manifest_sha256": data.split_manifest_sha256,
    }

    calibration_report, conformal = _calibrate(config, data, selected)
    replay_report, replay_predictions, replay_intervals = _evaluate_replay(
        config, data, selected, conformal, comparison_cap
    )
    official_report, official_intervals = _evaluate_official(
        data, selected, conformal, comparison_cap
    )
    conformal_report = {
        "calibration": calibration_report,
        "replay": replay_intervals,
        "official_test": official_intervals,
    }
    simulation_report = simulate_maintenance_policies(
        replay_predictions,
        config.alerts,
        config.maintenance_scenarios,
    )

    bundle_metadata = {
        "candidate_id": selected.candidate_id,
        "model_kind": selected.candidate.kind.value,
        "model_configuration": selected.candidate.params,
        "preprocessing_policy": preprocessing_policy(selected.candidate, config),
        "target_definition": {"name": selected.target.name, "cap": selected.target.cap},
        "feature_count": len(features),
        "random_seed": config.random_seed,
        "evaluation_version": config.evaluation_version,
        "feature_manifest_sha256": data.feature_manifest_sha256,
        "split_manifest_sha256": data.split_manifest_sha256,
        "created_at": created_at.isoformat(),
    }
    bundle = ModelBundle(
        pipeline=selected.pipeline,
        feature_columns=features,
        target_name=selected.target.name,
        target_cap=selected.target.cap,
        critical_horizon=config.alerts.critical_horizon,
        warning_horizon=config.alerts.warning_horizon,
        conformal=conformal,
        metadata=bundle_metadata,
    )
    champion_bytes = serialize_joblib(bundle)
    champion_path = models_dir / "champion.joblib"
    write_bytes(champion_path, champion_bytes)

    metadata = {
        **bundle_metadata,
        "ordered_feature_list": list(features),
        "alert_thresholds": selection_artifact["selected_alert_thresholds"],
        "conformal": conformal.record(),
        "champion_model_checksum": sha256_bytes(champion_bytes),
        "training_configuration": config_record(config),
        "evaluation_summaries": {
            "validation": selected.report["regression_metrics"]["row_weighted"],
            "replay": replay_report["regression_metrics"]["row_weighted"],
            "official_test": official_report["regression_metrics"]["row_weighted"],
        },
        "serialization_warning": (
            "Joblib/pickle files can execute arbitrary code when loaded; load only trusted, "
            "checksum-verified artifacts."
        ),
    }
    write_json(models_dir / "champion_metadata.json", metadata)

    interpretation = _interpret_models(runs, features, data)
    report_payloads = {
        "candidate_comparison": {"candidates": [run.report for run in runs]},
        "validation_report": {
            "dataset_role": "validation",
            "selected_candidate": selected.candidate_id,
            "selected_candidate_metrics": selected.report,
        },
        "champion_selection": selection_artifact,
        "replay_evaluation": replay_report,
        "official_test_benchmark": official_report,
        "conformal_metrics": conformal_report,
        "maintenance_simulation": simulation_report,
        "model_interpretation": interpretation,
        "model_latency_size": {
            "candidates": [
                {
                    "candidate_id": run.candidate_id,
                    "prediction_latency_ms": run.report["prediction_latency_ms"],
                    "model_size_bytes": run.report["model_size_bytes"],
                }
                for run in runs
            ]
        },
    }
    report_paths = write_report_set(reports_dir, report_payloads)
    comparison_csv = reports_dir / "candidate_comparison.csv"
    write_candidate_comparison(comparison_csv, [run.report for run in runs])
    slice_csv = reports_dir / "slice_metrics.csv"
    _write_slice_csv(slice_csv, runs)
    interpretation_csv = reports_dir / "feature_importance_coefficients.csv"
    _write_records_csv(interpretation_csv, interpretation["feature_records"])
    summary_path = reports_dir / "evaluation_summary.md"
    write_human_report(
        summary_path,
        selection=selection_artifact,
        replay=replay_report,
        official=official_report,
        conformal=conformal_report,
        simulation=simulation_report,
    )
    all_paths = [
        *model_paths,
        champion_path,
        models_dir / "champion_metadata.json",
        *report_paths,
        comparison_csv,
        slice_csv,
        interpretation_csv,
        summary_path,
    ]

    _verify_inputs_unchanged(data, config)
    records = tuple(record_file(path, output_root) for path in sorted(all_paths))
    manifest = TrainingManifest(
        evaluation_version=config.evaluation_version,
        created_at=created_at,
        git_commit=current_git_commit(),
        dataset_manifest_sha256=data.feature_manifest.source_report_sha256,
        feature_manifest_sha256=data.feature_manifest_sha256,
        split_manifest_sha256=data.split_manifest_sha256,
        configuration_sha256=configuration_sha256(config),
        input_checksums=data.input_checksums,
        selected_candidate_id=selected.candidate_id,
        artifacts=records,
    )
    write_training_manifest(manifest, output_root / "training_manifest.json")
    summary = {
        "selected_candidate_id": selected.candidate_id,
        "target": selected.target.name,
        "validation_common_rmse": selected.report["common_domain_metrics"]["rmse"],
        "replay_rmse": replay_report["regression_metrics"]["row_weighted"]["rmse"],
        "official_test_rmse": official_report["regression_metrics"]["row_weighted"]["rmse"],
        "replay_interval_coverage": replay_intervals["empirical_coverage"],
        "artifacts_dir": str(output_root),
    }
    return TrainingResult(
        status=TrainingStatus.TRAINED,
        selected_candidate_id=selected.candidate_id,
        artifacts_dir=output_root,
        champion_path=champion_path,
        champion_selection_path=reports_dir / "champion_selection.json",
        summary=summary,
    )


def _calibrate(
    config: TrainingConfig,
    data: VerifiedModelData,
    selected: _CandidateRun,
) -> tuple[dict[str, Any], SplitConformalCalibrator]:
    frame = data.frame(DatasetRole.CALIBRATION)
    features = model_matrix(frame, data.feature_columns)
    truth = target_values(frame, selected.target.cap)
    prediction = _bounded_prediction(selected.pipeline, features, selected.target.cap)
    calibrator = SplitConformalCalibrator(config.conformal_coverage).fit(truth, prediction)
    lower, upper = _bounded_intervals(calibrator, prediction, selected.target.cap)
    metrics = interval_metrics(
        truth,
        lower,
        upper,
        lifecycle_stages(frame[CYCLE_COLUMN], frame["rul"]),
    )
    return {**calibrator.record(), "empirical_metrics": metrics}, calibrator


def _evaluate_replay(
    config: TrainingConfig,
    data: VerifiedModelData,
    selected: _CandidateRun,
    conformal: SplitConformalCalibrator,
    comparison_cap: int,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    source = data.frame(DatasetRole.REPLAY)
    features = model_matrix(source, data.feature_columns)
    prediction = _bounded_prediction(selected.pipeline, features, selected.target.cap)
    frame = _prediction_frame(source, selected.target, prediction)
    lower, upper = _bounded_intervals(conformal, prediction, selected.target.cap)
    intervals = interval_metrics(
        frame["y_true"],
        lower,
        upper,
        lifecycle_stages(frame[CYCLE_COLUMN], frame["y_true_uncapped"]),
    )
    report = {
        "dataset_role": "replay_final_held_out",
        "target_definition": {"name": selected.target.name, "cap": selected.target.cap},
        "regression_metrics": evaluate_regression_frame(frame),
        "common_domain_metrics": common_domain_metrics(frame, comparison_cap),
        "critical_alert_metrics": alert_metrics(
            frame,
            horizon=config.alerts.critical_horizon,
            minimum_lead_cycles=config.alerts.minimum_lead_cycles,
        ),
        "warning_alert_metrics": alert_metrics(
            frame,
            horizon=config.alerts.warning_horizon,
            minimum_lead_cycles=config.alerts.minimum_lead_cycles,
        ),
        "conformal_metrics": intervals,
    }
    return report, frame, intervals


def _evaluate_official(
    data: VerifiedModelData,
    selected: _CandidateRun,
    conformal: SplitConformalCalibrator,
    comparison_cap: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = official_final_rows(data)
    features = model_matrix(source, data.feature_columns)
    prediction = _bounded_prediction(selected.pipeline, features, selected.target.cap)
    frame = _prediction_frame(source, selected.target, prediction)
    lower, upper = _bounded_intervals(conformal, prediction, selected.target.cap)
    intervals = interval_metrics(frame["y_true"], lower, upper)
    report = {
        "dataset_role": "official_nasa_test_final_rows_only",
        "target_definition": {"name": selected.target.name, "cap": selected.target.cap},
        "benchmark_note": (
            "One prediction per asset at its final observed test cycle, joined to the official "
            "NASA final-row RUL value. No official labels influenced design or selection."
        ),
        "regression_metrics": evaluate_regression_frame(frame),
        "common_domain_metrics": common_domain_metrics(frame, comparison_cap),
        "conformal_metrics": intervals,
    }
    return report, intervals


def _prediction_frame(
    source: pd.DataFrame, target: TargetConfig, prediction: np.ndarray
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            ASSET_ID_COLUMN: source[ASSET_ID_COLUMN].to_numpy(dtype="int64"),
            CYCLE_COLUMN: source[CYCLE_COLUMN].to_numpy(dtype="int64"),
            "y_true_uncapped": source["rul"].to_numpy(dtype="float64"),
            "y_true": target_values(source, target.cap).to_numpy(dtype="float64"),
            "y_pred": prediction,
        }
    )


def _bounded_prediction(pipeline: Pipeline, features: pd.DataFrame, cap: int | None) -> np.ndarray:
    raw_prediction = pipeline.predict(features)
    if isinstance(raw_prediction, tuple):
        raise ValueError("Regressor returned an unexpected tuple prediction.")
    prediction = cast(
        np.ndarray[Any, np.dtype[np.float64]],
        np.maximum(0.0, np.asarray(raw_prediction, dtype="float64")),
    )
    if not bool(np.isfinite(prediction).all()):
        raise ValueError("Model produced non-finite predictions.")
    return (
        prediction
        if cap is None
        else cast(
            np.ndarray[Any, np.dtype[np.float64]],
            np.minimum(prediction, cap),
        )
    )


def _bounded_intervals(
    calibrator: SplitConformalCalibrator,
    prediction: np.ndarray,
    cap: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = calibrator.intervals(prediction)
    return (lower, upper if cap is None else np.minimum(upper, cap))


def _prediction_latency_ms(
    pipeline: Pipeline,
    features: pd.DataFrame,
    cap: int | None,
    *,
    repeats: int,
) -> float:
    batch = features.iloc[: min(512, len(features))]
    _bounded_prediction(pipeline, batch, cap)
    timings: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        _bounded_prediction(pipeline, batch, cap)
        timings.append(1000.0 * (time.perf_counter() - started) / len(batch))
    return float(np.median(timings))


def _interpret_models(
    runs: list[_CandidateRun],
    features: tuple[str, ...],
    data: VerifiedModelData,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for kind in (ModelKind.RIDGE, ModelKind.XGBOOST):
        matching = [run for run in runs if run.candidate.kind is kind]
        if not matching:
            continue
        best = min(matching, key=lambda run: float(run.report["common_domain_metrics"]["rmse"]))
        model = best.pipeline.named_steps["model"]
        if kind is ModelKind.RIDGE:
            imputer = best.pipeline.named_steps["imputer"]
            names = [str(name) for name in imputer.get_feature_names_out(list(features))]
            values = np.asarray(model.coef_, dtype="float64")
            measure = "standardized_ridge_coefficient"
        else:
            names = list(features)
            values = np.asarray(model.feature_importances_, dtype="float64")
            measure = "xgboost_builtin_feature_importance"
        for name, value in zip(names, values, strict=True):
            records.append(
                {
                    "candidate_id": best.candidate_id,
                    "model_kind": kind.value,
                    "feature": name,
                    "measure": measure,
                    "value": float(value),
                    "absolute_value": abs(float(value)),
                }
            )

    source_columns = data.feature_manifest.feature_config.source_columns
    grouped: dict[str, float] = {}
    xgb_records = [record for record in records if record["model_kind"] == ModelKind.XGBOOST.value]
    for record in xgb_records:
        feature = str(record["feature"])
        source = next(
            (column for column in source_columns if feature.startswith(column + "_")), "other"
        )
        grouped[source] = grouped.get(source, 0.0) + float(record["absolute_value"])
    total = sum(grouped.values())
    families = [
        {"source_column": source, "importance_share": value / total if total else 0.0}
        for source, value in sorted(grouped.items(), key=lambda item: (-item[1], item[0]))
    ]
    sorted_importance = sorted(
        (float(record["absolute_value"]) for record in xgb_records), reverse=True
    )
    top20_share = (
        sum(sorted_importance[:20]) / sum(sorted_importance) if sorted_importance else None
    )
    return {
        "feature_records": records,
        "source_family_summary": families,
        "top_20_xgboost_importance_share": top20_share,
        "simpler_subset_observation": (
            "A concentrated top-20 share suggests a smaller feature subset may be competitive, "
            "but Loop 3 is not redesigned in this loop."
            if top20_share is not None and top20_share >= 0.70
            else (
                "Importance is not concentrated enough to justify a smaller subset without a "
                "separate study."
            )
        ),
        "limitations": (
            "Ridge coefficients depend on scaling and collinearity. Built-in XGBoost importance "
            "can favor features used frequently in splits. Neither supports causal interpretation."
        ),
    }


def _write_slice_csv(path: Path, runs: list[_CandidateRun]) -> None:
    records: list[dict[str, Any]] = []
    for run in runs:
        for record in run.report["regression_metrics"]["slices"]:
            records.append({"candidate_id": run.candidate_id, **record})
    _write_records_csv(path, records)


def _write_records_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    tmp_path = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def _verify_inputs_unchanged(data: VerifiedModelData, config: TrainingConfig) -> None:
    for filename, expected in data.input_checksums.items():
        path = config.features_dir / filename
        actual = sha256_path(path)
        if actual != expected:
            raise ModelDataError(f"Loop 3 input changed during training: {filename}.")

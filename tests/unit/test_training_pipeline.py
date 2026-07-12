"""Small-fixture Loop 4 pipeline, artifact, idempotency, and CLI tests."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from turbine_guard.modeling.artifacts import load_joblib
from turbine_guard.modeling.cli import main
from turbine_guard.modeling.config import (
    CandidateConfig,
    ModelKind,
    SelectionConfig,
    TargetConfig,
    TrainingConfig,
)
from turbine_guard.modeling.data import DatasetRole, load_verified_model_data, model_matrix
from turbine_guard.modeling.pipeline import (
    TrainingError,
    TrainingResult,
    TrainingStatus,
    train_models,
)


def tiny_candidates() -> tuple[CandidateConfig, ...]:
    return (
        CandidateConfig("constant", ModelKind.CONSTANT),
        CandidateConfig("ridge", ModelKind.RIDGE, (("alpha", 1.0),), 1),
        CandidateConfig(
            "tree",
            ModelKind.HIST_GRADIENT_BOOSTING,
            (("max_iter", 5), ("max_leaf_nodes", 7), ("learning_rate", 0.1)),
            2,
        ),
        CandidateConfig(
            "xgb",
            ModelKind.XGBOOST,
            (("n_estimators", 5), ("max_depth", 2), ("learning_rate", 0.1)),
            3,
        ),
    )


def training_config(data_dir: Path, output_dir: Path, *, force: bool = False) -> TrainingConfig:
    return TrainingConfig(
        data_dir=data_dir,
        output_dir=output_dir,
        targets=(TargetConfig("uncapped"), TargetConfig("capped_125", 125)),
        candidates=tiny_candidates(),
        selection=SelectionConfig(
            minimum_critical_recall=0.0,
            maximum_false_alarms_per_1000_cycles=1000.0,
            relative_rmse_tolerance=0.0,
        ),
        conformal_coverage=0.8,
        latency_repeats=1,
        force=force,
    )


def test_successful_training_pipeline_and_reload_equality(
    feature_data_dir: Path, tmp_path: Path
) -> None:
    config = training_config(feature_data_dir, tmp_path / "models")
    result = train_models(config)

    assert result.status is TrainingStatus.TRAINED
    assert result.champion_path.exists()
    assert result.champion_selection_path.exists()
    for name in (
        "candidate_comparison.csv",
        "validation_report.json",
        "replay_evaluation.json",
        "official_test_benchmark.json",
        "conformal_metrics.json",
        "maintenance_simulation.json",
        "evaluation_summary.md",
    ):
        assert (config.artifacts_dir / "reports" / name).exists()

    bundle = load_joblib(result.champion_path)
    data = load_verified_model_data(config)
    replay = model_matrix(data.frame(DatasetRole.REPLAY), data.feature_columns)
    first = bundle.predict(replay)
    restored = load_joblib(result.champion_path)
    np.testing.assert_array_equal(restored.predict(replay), first)

    conformal = json.loads(
        (config.artifacts_dir / "reports" / "conformal_metrics.json").read_text()
    )
    assert conformal["calibration"]["calibration_rows"] == len(data.frame(DatasetRole.CALIBRATION))


def test_idempotent_rerun_and_force_rebuild(feature_data_dir: Path, tmp_path: Path) -> None:
    output = tmp_path / "models"
    config = training_config(feature_data_dir, output)
    first = train_models(config)
    manifest_mtime = (output / "training_manifest.json").stat().st_mtime_ns
    second = train_models(config)
    assert first.status is TrainingStatus.TRAINED
    assert second.status is TrainingStatus.ALREADY_TRAINED
    assert (output / "training_manifest.json").stat().st_mtime_ns == manifest_mtime

    forced = train_models(training_config(feature_data_dir, output, force=True))
    assert forced.status is TrainingStatus.TRAINED


def test_training_artifact_tamper_detection(feature_data_dir: Path, tmp_path: Path) -> None:
    config = training_config(feature_data_dir, tmp_path / "models")
    result = train_models(config)
    result.champion_path.write_bytes(result.champion_path.read_bytes() + b"tamper")

    with pytest.raises(TrainingError, match="checksum mismatch"):
        train_models(config)


def test_missing_input_failure(tmp_path: Path) -> None:
    with pytest.raises(TrainingError, match="manifests are missing"):
        train_models(training_config(tmp_path / "missing", tmp_path / "models"))


def test_invalid_configuration_failure() -> None:
    with pytest.raises(ValueError, match="coverage"):
        TrainingConfig(conformal_coverage=1.0)


def test_loop3_feature_artifacts_remain_unchanged(feature_data_dir: Path, tmp_path: Path) -> None:
    features_dir = feature_data_dir / "features" / "cmapss" / "FD001"
    before = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in features_dir.iterdir()
        if path.is_file()
    }
    train_models(training_config(feature_data_dir, tmp_path / "models"))
    after = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in features_dir.iterdir()
        if path.is_file()
    }
    assert after == before


def test_cli_logs_concise_result(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    result = TrainingResult(
        status=TrainingStatus.TRAINED,
        selected_candidate_id="uncapped--ridge",
        artifacts_dir=Path("data/models"),
        champion_path=Path("data/models/champion.joblib"),
        champion_selection_path=Path("data/models/champion_selection.json"),
        summary={"selected_candidate_id": "uncapped--ridge"},
    )
    monkeypatch.setattr("turbine_guard.modeling.cli.train_models", lambda _: result)
    assert main([]) == 0
    output = capsys.readouterr().out
    assert '"message": "model_training_result"' in output
    assert '"status": "trained"' in output

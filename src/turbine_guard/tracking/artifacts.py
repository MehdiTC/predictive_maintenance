"""Verified bridge from Loop 4 local artifacts to MLflow logging."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from turbine_guard.modeling.artifacts import (
    ArtifactError,
    TrainingManifest,
    sha256_path,
)
from turbine_guard.modeling.config import TrainingConfig


@dataclass(frozen=True)
class TrackingArtifacts:
    """Complete, checksum-verified local inputs for one MLflow execution."""

    root: Path
    training_manifest_path: Path
    training_manifest: TrainingManifest
    training_manifest_sha256: str
    acquisition_manifest_path: Path
    acquisition_manifest_sha256: str
    validation_report_path: Path
    validation_report_sha256: str
    feature_manifest_path: Path
    split_manifest_path: Path
    champion_path: Path
    champion_sha256: str
    champion_metadata: dict[str, Any]
    candidate_reports: tuple[dict[str, Any], ...]
    selection_report: dict[str, Any]
    replay_report: dict[str, Any]
    official_report: dict[str, Any]
    conformal_report: dict[str, Any]
    policy_report: dict[str, Any]

    @property
    def execution_id(self) -> str:
        """Stable identity of this exact completed Loop 4 artifact set."""
        return self.training_manifest_sha256


def load_tracking_artifacts(config: TrainingConfig) -> TrackingArtifacts:
    """Load and verify all local artifacts needed for tracking and registration."""
    root = config.artifacts_dir
    manifest_path = root / "training_manifest.json"
    if not manifest_path.exists():
        raise ArtifactError(f"Loop 4 training manifest is missing: {manifest_path}.")
    try:
        manifest = TrainingManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ArtifactError(f"Loop 4 training manifest is unreadable: {exc}") from exc
    for record in manifest.artifacts:
        path = root / record.filename
        if not path.exists():
            raise ArtifactError(f"Loop 4 artifact is missing: {path}.")
        if sha256_path(path) != record.sha256:
            raise ArtifactError(f"Loop 4 artifact checksum mismatch: {path}.")

    acquisition = config.data_dir / "manifests" / f"cmapss_{config.subset.lower()}.json"
    validation = config.data_dir / "processed" / "cmapss" / config.subset / "processing_report.json"
    feature = config.features_dir / "feature_manifest.json"
    split = config.features_dir / "split_manifest.json"
    for path in (acquisition, validation, feature, split):
        if not path.exists():
            raise ArtifactError(f"Required lineage artifact is missing: {path}.")

    validation_sha = sha256_path(validation)
    if validation_sha != manifest.dataset_manifest_sha256:
        raise ArtifactError(
            "Validation report checksum does not match the completed training manifest."
        )
    if sha256_path(feature) != manifest.feature_manifest_sha256:
        raise ArtifactError("Feature manifest checksum does not match training.")
    if sha256_path(split) != manifest.split_manifest_sha256:
        raise ArtifactError("Split manifest checksum does not match training.")

    champion_path = root / "models" / "champion.joblib"
    metadata = _load_json(root / "models" / "champion_metadata.json")
    champion_sha = sha256_path(champion_path)
    if metadata.get("champion_model_checksum") != champion_sha:
        raise ArtifactError("Champion checksum does not match champion metadata.")

    comparison = _load_json(root / "reports" / "candidate_comparison.json")
    candidates = comparison.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ArtifactError("Candidate comparison does not contain candidate runs.")
    selection = _load_json(root / "reports" / "champion_selection.json")
    if selection.get("selected_model") != manifest.selected_candidate_id:
        raise ArtifactError("Champion selection disagrees with the training manifest.")
    if len(candidates) != len(selection.get("candidates", [])):
        raise ArtifactError("Candidate comparison and selection candidate counts differ.")

    return TrackingArtifacts(
        root=root,
        training_manifest_path=manifest_path,
        training_manifest=manifest,
        training_manifest_sha256=sha256_path(manifest_path),
        acquisition_manifest_path=acquisition,
        acquisition_manifest_sha256=sha256_path(acquisition),
        validation_report_path=validation,
        validation_report_sha256=validation_sha,
        feature_manifest_path=feature,
        split_manifest_path=split,
        champion_path=champion_path,
        champion_sha256=champion_sha,
        champion_metadata=metadata,
        candidate_reports=tuple(candidates),
        selection_report=selection,
        replay_report=_load_json(root / "reports" / "replay_evaluation.json"),
        official_report=_load_json(root / "reports" / "official_test_benchmark.json"),
        conformal_report=_load_json(root / "reports" / "conformal_metrics.json"),
        policy_report=_load_json(root / "reports" / "maintenance_simulation.json"),
    )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"Required JSON artifact is unreadable ({path}): {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactError(f"Required JSON artifact is not an object: {path}.")
    return value

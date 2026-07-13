"""Versioned training-only feature distributions for production drift comparisons."""

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from turbine_guard.features.manifest import load_feature_manifest, sha256_of

REFERENCE_VERSION = "1"
_QUANTILE_PROBABILITIES = tuple(float(value) for value in np.linspace(0.0, 1.0, 101))


class FeatureReference(BaseModel):
    model_config = ConfigDict(frozen=True)

    count: int
    missing_rate: float
    mean: float | None
    std: float | None
    minimum: float | None
    maximum: float | None
    bin_edges: tuple[float, ...]
    bin_probabilities: tuple[float, ...]
    quantiles: tuple[float, ...]


class TrainingReference(BaseModel):
    """Compact distribution record derived exclusively from Loop 3 training rows."""

    model_config = ConfigDict(frozen=True)

    reference_version: str
    created_at: datetime
    model_name: str
    model_version: str
    feature_version: str
    feature_manifest_sha256: str
    training_parquet_sha256: str
    row_count: int
    asset_count: int
    feature_columns: tuple[str, ...]
    quantile_probabilities: tuple[float, ...]
    features: dict[str, FeatureReference]


@dataclass(frozen=True)
class ReferenceArtifact:
    reference: TrainingReference
    path: Path
    sha256: str


def build_training_reference(
    *,
    data_dir: Path,
    model_name: str,
    model_version: str,
    expected_feature_version: str,
) -> ReferenceArtifact:
    """Build or verify an exact champion-bound, training-only reference artifact."""
    features_dir = data_dir / "features" / "cmapss" / "FD001"
    manifest_path = features_dir / "feature_manifest.json"
    manifest = load_feature_manifest(manifest_path)
    if manifest.feature_config.feature_version != expected_feature_version:
        raise ValueError("Champion and feature-manifest versions differ.")
    train_record = next(
        (record for record in manifest.outputs if record.filename == "train.parquet"), None
    )
    if train_record is None:
        raise ValueError("Feature manifest has no training-only Parquet output.")
    train_path = features_dir / train_record.filename
    actual_train_sha = sha256_of(train_path)
    if actual_train_sha != train_record.sha256:
        raise ValueError("Training feature Parquet checksum does not match its manifest.")

    output = (
        data_dir
        / "monitoring"
        / "references"
        / _safe_component(model_name)
        / f"v{_safe_component(model_version)}"
        / "training_reference.json"
    )
    if output.exists():
        reference = TrainingReference.model_validate_json(output.read_text(encoding="utf-8"))
        _verify_identity(
            reference,
            model_name=model_name,
            model_version=model_version,
            feature_version=expected_feature_version,
            feature_manifest_sha256=sha256_of(manifest_path),
            training_parquet_sha256=actual_train_sha,
        )
        return ReferenceArtifact(reference, output, reference_identity(reference))

    frame = pd.read_parquet(train_path)
    columns = tuple(manifest.feature_columns)
    if tuple(column for column in frame.columns if column in columns) != columns:
        raise ValueError("Training feature order does not match the feature manifest.")
    references = {column: _feature_reference(frame[column]) for column in columns}
    reference = TrainingReference(
        reference_version=REFERENCE_VERSION,
        created_at=datetime.now(UTC),
        model_name=model_name,
        model_version=model_version,
        feature_version=expected_feature_version,
        feature_manifest_sha256=sha256_of(manifest_path),
        training_parquet_sha256=actual_train_sha,
        row_count=len(frame),
        asset_count=int(frame["asset_id"].nunique()),
        feature_columns=columns,
        quantile_probabilities=_QUANTILE_PROBABILITIES,
        features=references,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(reference.model_dump_json(indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    return ReferenceArtifact(reference, output, reference_identity(reference))


def reference_identity(reference: TrainingReference) -> str:
    """Stable identity for a reference independent of its creation timestamp."""
    payload: dict[str, Any] = reference.model_dump(mode="json")
    payload.pop("created_at")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _feature_reference(series: pd.Series) -> FeatureReference:
    values = series.to_numpy(dtype="float64")
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return FeatureReference(
            count=0,
            missing_rate=1.0,
            mean=None,
            std=None,
            minimum=None,
            maximum=None,
            bin_edges=(),
            bin_probabilities=(),
            quantiles=(),
        )
    internal = np.unique(np.quantile(finite, np.linspace(0.1, 0.9, 9)))
    histogram_edges = np.concatenate(([-np.inf], internal, [np.inf]))
    counts, _ = np.histogram(finite, bins=histogram_edges)
    probabilities = counts.astype("float64") / float(finite.size)
    return FeatureReference(
        count=int(finite.size),
        missing_rate=float(1.0 - finite.size / values.size),
        mean=float(np.mean(finite)),
        std=float(np.std(finite, ddof=0)),
        minimum=float(np.min(finite)),
        maximum=float(np.max(finite)),
        bin_edges=tuple(float(value) for value in internal),
        bin_probabilities=tuple(float(value) for value in probabilities),
        quantiles=tuple(float(value) for value in np.quantile(finite, _QUANTILE_PROBABILITIES)),
    )


def _verify_identity(reference: TrainingReference, **expected: str) -> None:
    actual = {
        "model_name": reference.model_name,
        "model_version": reference.model_version,
        "feature_version": reference.feature_version,
        "feature_manifest_sha256": reference.feature_manifest_sha256,
        "training_parquet_sha256": reference.training_parquet_sha256,
    }
    if actual != expected:
        raise ValueError("Existing training reference belongs to different model lineage.")


def _safe_component(value: str) -> str:
    normalized = "".join(
        character if character.isalnum() or character in "-_." else "_" for character in value
    )
    if not normalized:
        raise ValueError("Reference path identity cannot be empty.")
    return normalized

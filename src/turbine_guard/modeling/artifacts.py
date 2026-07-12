"""Local Loop 4 artifact serialization, checksums, and idempotency manifest."""

import hashlib
import io
import json
from dataclasses import dataclass
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from turbine_guard.modeling.config import TrainingConfig, config_record

joblib = import_module("joblib")


class ArtifactError(RuntimeError):
    """Raised when generated training artifacts are missing or tampered."""


class ArtifactRecord(BaseModel):
    """Checksum and size of one generated artifact relative to the output root."""

    model_config = ConfigDict(frozen=True)

    filename: str
    sha256: str
    size_bytes: int


class TrainingManifest(BaseModel):
    """Source-of-truth manifest for one complete Loop 4 artifact set."""

    model_config = ConfigDict(frozen=True)

    evaluation_version: str
    created_at: datetime
    git_commit: str | None
    dataset_manifest_sha256: str
    feature_manifest_sha256: str
    split_manifest_sha256: str
    configuration_sha256: str
    input_checksums: dict[str, str]
    selected_candidate_id: str
    artifacts: tuple[ArtifactRecord, ...]


@dataclass(frozen=True)
class LoadedExistingRun:
    """Verified result of an idempotent rerun check."""

    manifest: TrainingManifest
    champion_selection: dict[str, Any]


def configuration_sha256(config: TrainingConfig) -> str:
    """Stable checksum of the typed run configuration excluding ``force``."""
    encoded = json.dumps(config_record(config), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def serialize_joblib(value: Any) -> bytes:
    """Serialize an object using joblib's established sklearn-compatible format."""
    buffer = io.BytesIO()
    joblib.dump(value, buffer, compress=3)
    return buffer.getvalue()


def load_joblib(path: Path) -> Any:
    """Load a trusted local joblib artifact.

    Joblib/pickle deserialization can execute arbitrary code. Callers must only
    load artifacts they trust and whose checksum has been verified.
    """
    return joblib.load(path)


def write_bytes(path: Path, content: bytes) -> ArtifactRecord:
    """Atomically write bytes and return their relative record later."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_bytes(content)
    tmp_path.replace(path)
    return ArtifactRecord(filename=str(path), sha256=sha256_bytes(content), size_bytes=len(content))


def write_json(path: Path, payload: Any) -> None:
    """Atomically write stable pretty JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def write_text(path: Path, content: str) -> None:
    """Atomically write UTF-8 text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(path)


def record_file(path: Path, root: Path) -> ArtifactRecord:
    """Build an artifact record with a path relative to ``root``."""
    return ArtifactRecord(
        filename=str(path.relative_to(root)),
        sha256=sha256_path(path),
        size_bytes=path.stat().st_size,
    )


def write_training_manifest(manifest: TrainingManifest, path: Path) -> None:
    """Write the completion manifest last so partial runs are never current."""
    write_text(path, manifest.model_dump_json(indent=2) + "\n")


def verify_existing_run(
    config: TrainingConfig,
    *,
    dataset_manifest_sha256: str,
    feature_manifest_sha256: str,
    split_manifest_sha256: str,
    input_checksums: dict[str, str],
) -> LoadedExistingRun | None:
    """Verify and return an idempotent current run, or ``None`` if absent/stale."""
    manifest_path = config.artifacts_dir / "training_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = TrainingManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ArtifactError(
            f"Existing training manifest {manifest_path} is unreadable: {exc}. "
            "Use --force for an intentional rebuild."
        ) from exc

    current = (
        manifest.evaluation_version == config.evaluation_version
        and manifest.dataset_manifest_sha256 == dataset_manifest_sha256
        and manifest.feature_manifest_sha256 == feature_manifest_sha256
        and manifest.split_manifest_sha256 == split_manifest_sha256
        and manifest.configuration_sha256 == configuration_sha256(config)
        and manifest.input_checksums == input_checksums
    )
    if not current:
        return None
    for record in manifest.artifacts:
        path = config.artifacts_dir / record.filename
        if not path.exists():
            raise ArtifactError(
                f"Training artifact {path} is missing. Use --force for an intentional rebuild."
            )
        actual = sha256_path(path)
        if actual != record.sha256:
            raise ArtifactError(
                f"Training artifact checksum mismatch for {path}: expected {record.sha256}, "
                f"found {actual}. Use --force for an intentional rebuild."
            )
    selection_path = config.artifacts_dir / "reports" / "champion_selection.json"
    try:
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"Champion selection report is unreadable: {exc}") from exc
    return LoadedExistingRun(manifest, selection)


def sha256_path(path: Path) -> str:
    """Streaming SHA-256 for a local file."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sha256_bytes(content: bytes) -> str:
    """SHA-256 for in-memory serialized content."""
    return hashlib.sha256(content).hexdigest()

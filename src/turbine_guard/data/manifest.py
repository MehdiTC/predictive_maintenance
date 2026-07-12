"""Acquisition manifest models and persistence.

A manifest is the provenance record for one acquired raw dataset subset. It is
written next to the raw layer (under ``data/manifests/``) and is the source of
truth for later verification: re-running acquisition compares the raw files on
disk against the checksums recorded here.
"""

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class FileRecord(BaseModel):
    """Provenance record for a single acquired file."""

    model_config = ConfigDict(frozen=True)

    filename: str
    sha256: str
    size_bytes: int
    record_count: int | None = None
    """Number of non-empty lines; ``None`` for binary files such as archives."""

    asset_count: int | None = None
    """Number of distinct unit IDs (first column); trajectory files only."""


class AcquisitionManifest(BaseModel):
    """Provenance manifest written after acquiring a raw dataset subset."""

    model_config = ConfigDict(frozen=True)

    acquisition_version: str
    dataset_name: str
    dataset_subset: str
    source_name: str
    source_url: str
    retrieved_at: datetime
    acquired_by: str
    git_commit: str | None
    archive: FileRecord
    files: tuple[FileRecord, ...]
    notes: str


def load_manifest(path: Path) -> AcquisitionManifest:
    """Load and validate a manifest from ``path``."""
    return AcquisitionManifest.model_validate_json(path.read_text(encoding="utf-8"))


def write_manifest(manifest: AcquisitionManifest, path: Path) -> None:
    """Atomically write ``manifest`` to ``path`` as pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)

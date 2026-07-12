"""Split and feature manifest models and persistence.

The split manifest records exactly which assets landed in each partition and
how it was derived. The feature manifest records the feature definition, the
inputs it consumed (by checksum), and the outputs it produced (by checksum), so
that a future training run can identify precisely which feature version and
split it used and detect any tampering.

Both are pydantic frozen models written atomically as pretty JSON, mirroring
the Loop 1/2 manifest and report convention.
"""

import hashlib
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from turbine_guard.data.manifest import FileRecord


class SplitManifest(BaseModel):
    """Machine-readable record of one asset-level split assignment."""

    model_config = ConfigDict(frozen=True)

    dataset_name: str
    dataset_subset: str
    split_version: str
    created_at: datetime
    created_by: str
    git_commit: str | None
    seed: int
    strategy: str
    source_report_sha256: str
    """Checksum of the Loop 2 processing report the split was derived from."""

    partitions: dict[str, tuple[int, ...]]
    asset_counts: dict[str, int]
    row_counts: dict[str, int]


class FeatureConfigRecord(BaseModel):
    """The feature configuration captured in the feature manifest."""

    model_config = ConfigDict(frozen=True)

    feature_version: str
    source_columns: tuple[str, ...]
    families: tuple[str, ...]
    windows: tuple[int, ...]
    ewm_spans: tuple[int, ...]
    min_periods: int


class FeatureOutputRecord(FileRecord):
    """Provenance for one generated model-ready Parquet output."""

    model_config = ConfigDict(frozen=True)

    split: str
    """Partition name (train/validation/…) or the benchmark role (test/…)."""

    null_count: int
    """Total number of null cells across all feature columns in the file."""

    has_targets: bool
    """Whether the file carries RUL target columns (never true for test features)."""


class FeatureManifest(BaseModel):
    """Machine-readable record of one feature-generation run."""

    model_config = ConfigDict(frozen=True)

    feature_build_version: str
    schema_version: str
    dataset_name: str
    dataset_subset: str
    created_at: datetime
    created_by: str
    git_commit: str | None
    seed: int

    source_report_sha256: str
    """Links the features to the exact validated Loop 2 inputs."""

    split_manifest_sha256: str
    """Links the features to the exact split assignment used."""

    split_version: str
    feature_config: FeatureConfigRecord
    rul_cap: int | None
    imputation: str | None
    """Imputation policy; ``None`` means imputation is deferred to Loop 4."""

    identifier_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    metadata_columns: tuple[str, ...]
    feature_columns: tuple[str, ...]

    inputs: tuple[FileRecord, ...]
    """Loop 2 outputs consumed (train/test/rul Parquet + report), by checksum."""

    outputs: tuple[FeatureOutputRecord, ...]
    row_counts_by_split: dict[str, int]
    asset_counts_by_split: dict[str, int]


def load_split_manifest(path: Path) -> SplitManifest:
    """Load and validate a split manifest from ``path``."""
    return SplitManifest.model_validate_json(path.read_text(encoding="utf-8"))


def load_feature_manifest(path: Path) -> FeatureManifest:
    """Load and validate a feature manifest from ``path``."""
    return FeatureManifest.model_validate_json(path.read_text(encoding="utf-8"))


def write_manifest(manifest: BaseModel, path: Path) -> None:
    """Atomically write ``manifest`` to ``path`` as pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def sha256_of(path: Path) -> str:
    """Streaming SHA-256 of a file on disk."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

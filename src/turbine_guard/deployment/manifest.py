"""Typed deployment-bundle manifest and the fixed bundle file layout."""

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

BUNDLE_SCHEMA_VERSION = "1"
MANIFEST_FILENAME = "deployment_manifest.json"
STATE_FILENAME = "deployment_bundle_state.json"

CHAMPION_RELATIVE_PATH = Path("models/cmapss/FD001/models/champion.joblib")
FEATURE_MANIFEST_RELATIVE_PATH = Path("features/cmapss/FD001/feature_manifest.json")
SPLIT_MANIFEST_RELATIVE_PATH = Path("features/cmapss/FD001/split_manifest.json")
PROCESSING_REPORT_RELATIVE_PATH = Path("processed/cmapss/FD001/processing_report.json")
TRAIN_PARQUET_RELATIVE_PATH = Path("processed/cmapss/FD001/train_FD001.parquet")

REQUIRED_BUNDLE_FILES = (
    CHAMPION_RELATIVE_PATH,
    FEATURE_MANIFEST_RELATIVE_PATH,
    SPLIT_MANIFEST_RELATIVE_PATH,
    PROCESSING_REPORT_RELATIVE_PATH,
    TRAIN_PARQUET_RELATIVE_PATH,
)
"""Everything serving needs, relative to the application data directory.

The train Parquet, processing report, and split manifest let the Loop 8
replay source re-verify its checksum chain; the feature manifest defines the
shared serving feature contract; the champion joblib is the model itself.
"""


class BundleFileRecord(BaseModel):
    """Checksum and size of one bundled file relative to the data directory."""

    model_config = ConfigDict(frozen=True)

    relative_path: str
    sha256: str
    size_bytes: int


class DeploymentBundleManifest(BaseModel):
    """Immutable exported champion snapshot identity and content checksums.

    Registry fields are captured from the live MLflow registry at export
    time; the public demo serves them read-only and never mutates a registry.
    """

    model_config = ConfigDict(frozen=True)

    bundle_schema_version: str = BUNDLE_SCHEMA_VERSION
    created_at: datetime
    git_commit: str | None

    registered_model_name: str
    registry_version: str
    champion_alias: str
    aliases: dict[str, str]
    source_run_id: str
    model_family: str | None
    target_definition: str
    rul_cap: int | None
    feature_version: str
    feature_count: int

    validation_rmse: float | None
    replay_rmse: float | None
    official_test_rmse: float | None
    conformal_coverage_target: float | None

    champion_bundle_sha256: str
    feature_manifest_sha256: str
    dataset_checksum: str | None
    lineage_id: str | None

    files: tuple[BundleFileRecord, ...]


class DeploymentBundleState(BaseModel):
    """Restore-completion marker written last, binding files to one archive."""

    model_config = ConfigDict(frozen=True)

    archive_sha256: str
    restored_at: datetime


class BundleError(RuntimeError):
    """Raised when a deployment bundle cannot be exported, verified, or restored."""


def load_bundle_manifest(path: Path) -> DeploymentBundleManifest:
    """Load and validate one deployment manifest from disk."""
    try:
        return DeploymentBundleManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BundleError(f"Deployment manifest {path} is unreadable or invalid: {exc}") from exc

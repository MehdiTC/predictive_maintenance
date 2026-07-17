"""Download, verify, and atomically restore a pinned deployment bundle."""

import hashlib
import logging
import shutil
import tarfile
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from turbine_guard.config.settings import Settings
from turbine_guard.deployment.manifest import (
    MANIFEST_FILENAME,
    REQUIRED_BUNDLE_FILES,
    STATE_FILENAME,
    BundleError,
    DeploymentBundleManifest,
    DeploymentBundleState,
    load_bundle_manifest,
)
from turbine_guard.modeling.artifacts import sha256_path

logger = logging.getLogger(__name__)

_DOWNLOAD_CHUNK_BYTES = 1 << 20
_MAX_ARCHIVE_BYTES = 512 * (1 << 20)


class RestoreStatus(StrEnum):
    """Outcome of one restore command."""

    RESTORED = "restored"
    ALREADY_RESTORED = "already_restored"


@dataclass(frozen=True)
class RestoreResult:
    """Verified manifest and archive identity after a restore."""

    status: RestoreStatus
    archive_sha256: str
    manifest: DeploymentBundleManifest


def restore_deployment_bundle(settings: Settings) -> RestoreResult:
    """Restore the configured bundle into the data directory, idempotently.

    The archive must match the pinned SHA-256 exactly, every extracted file
    must match the manifest it carries, and the completion marker is written
    last so an interrupted restore is retried rather than trusted.
    """
    url = settings.deployment_bundle_url
    pinned = settings.deployment_bundle_sha256
    if url is None or pinned is None:
        raise BundleError(
            "Restore requires TURBINE_GUARD_DEPLOYMENT_BUNDLE_URL and "
            "TURBINE_GUARD_DEPLOYMENT_BUNDLE_SHA256."
        )
    data_dir = settings.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    existing = _verified_existing_restore(data_dir, pinned)
    if existing is not None:
        logger.info("deployment_bundle_already_restored", extra={"archive_sha256": pinned})
        return RestoreResult(RestoreStatus.ALREADY_RESTORED, pinned, existing)

    with tempfile.TemporaryDirectory(dir=data_dir, prefix=".bundle-restore-") as scratch:
        scratch_dir = Path(scratch)
        archive_path = scratch_dir / "bundle.tar.gz"
        actual = _download(url, archive_path)
        if actual != pinned:
            raise BundleError(
                f"Deployment bundle checksum mismatch: expected {pinned}, downloaded {actual}. "
                "The pin and the published archive revision must match."
            )
        extracted_dir = scratch_dir / "extracted"
        _extract(archive_path, extracted_dir)
        manifest = load_bundle_manifest(extracted_dir / MANIFEST_FILENAME)
        _verify_extracted(extracted_dir, manifest)
        _move_into_place(extracted_dir, data_dir, manifest)

    state = DeploymentBundleState(archive_sha256=pinned, restored_at=datetime.now(UTC))
    state_tmp = data_dir / (STATE_FILENAME + ".tmp")
    state_tmp.write_text(state.model_dump_json(indent=2) + "\n", encoding="utf-8")
    state_tmp.replace(data_dir / STATE_FILENAME)
    logger.info(
        "deployment_bundle_restored",
        extra={
            "archive_sha256": pinned,
            "registry_version": manifest.registry_version,
            "file_count": len(manifest.files),
        },
    )
    return RestoreResult(RestoreStatus.RESTORED, pinned, manifest)


def _verified_existing_restore(data_dir: Path, pinned: str) -> DeploymentBundleManifest | None:
    """Return the restored manifest only when marker, pin, and files all agree."""
    state_path = data_dir / STATE_FILENAME
    manifest_path = data_dir / MANIFEST_FILENAME
    if not state_path.is_file() or not manifest_path.is_file():
        return None
    try:
        state = DeploymentBundleState.model_validate_json(state_path.read_text(encoding="utf-8"))
        manifest = load_bundle_manifest(manifest_path)
    except (OSError, ValueError, BundleError):
        return None
    if state.archive_sha256 != pinned:
        return None
    for record in manifest.files:
        path = data_dir / record.relative_path
        if not path.is_file() or sha256_path(path) != record.sha256:
            return None
    return manifest


def _download(url: str, destination: Path) -> str:
    """Stream the archive to disk and return its SHA-256."""
    digest = hashlib.sha256()
    total = 0
    try:
        with urllib.request.urlopen(url) as response, destination.open("wb") as output:
            while chunk := response.read(_DOWNLOAD_CHUNK_BYTES):
                total += len(chunk)
                if total > _MAX_ARCHIVE_BYTES:
                    raise BundleError("Deployment bundle archive exceeds the size limit.")
                digest.update(chunk)
                output.write(chunk)
    except (urllib.error.URLError, OSError) as exc:
        raise BundleError(f"Deployment bundle download failed from {url}: {exc}") from exc
    return digest.hexdigest()


def _extract(archive_path: Path, destination: Path) -> None:
    """Extract with the stdlib 'data' filter, rejecting unsafe members."""
    destination.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(archive_path, mode="r:gz") as tar:
            for member in tar.getmembers():
                if not (member.isfile() or member.isdir()):
                    raise BundleError(
                        f"Deployment bundle contains a non-regular member: {member.name}"
                    )
            tar.extractall(destination, filter="data")
    except (tarfile.TarError, OSError) as exc:
        raise BundleError(f"Deployment bundle extraction failed: {exc}") from exc


def _verify_extracted(extracted_dir: Path, manifest: DeploymentBundleManifest) -> None:
    """Every manifest record must be present, sized, and checksum-exact."""
    listed = {record.relative_path for record in manifest.files}
    missing_required = [
        required.as_posix()
        for required in REQUIRED_BUNDLE_FILES
        if required.as_posix() not in listed
    ]
    if missing_required:
        raise BundleError(f"Deployment manifest does not list required files: {missing_required}")
    for record in manifest.files:
        path = extracted_dir / record.relative_path
        if not path.is_file():
            raise BundleError(f"Deployment bundle is missing {record.relative_path}.")
        if path.stat().st_size != record.size_bytes:
            raise BundleError(f"Deployment bundle size mismatch for {record.relative_path}.")
        actual = sha256_path(path)
        if actual != record.sha256:
            raise BundleError(
                f"Deployment bundle checksum mismatch for {record.relative_path}: "
                f"expected {record.sha256}, found {actual}."
            )


def _move_into_place(
    extracted_dir: Path, data_dir: Path, manifest: DeploymentBundleManifest
) -> None:
    """Replace target files one at a time; the marker written later gates trust."""
    for record in manifest.files:
        source = extracted_dir / record.relative_path
        target = data_dir / record.relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
    shutil.move(str(extracted_dir / MANIFEST_FILENAME), str(data_dir / MANIFEST_FILENAME))

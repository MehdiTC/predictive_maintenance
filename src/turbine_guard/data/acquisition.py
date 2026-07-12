"""Idempotent acquisition of the NASA C-MAPSS raw dataset.

Downloads the source archive, extracts the requested subset's files unchanged
into the immutable raw layer, and records a provenance manifest with SHA-256
checksums. Re-running verifies the raw layer against the manifest instead of
downloading again; a deliberate ``force`` re-acquisition replaces it.

The data is simulated turbofan engine degradation data from the NASA
Prognostics Center of Excellence. Sensor channels are anonymous and this
project does not assign them physical interpretations.
"""

import hashlib
import io
import logging
import shutil
import subprocess
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import IO

from turbine_guard import __version__
from turbine_guard.data.manifest import (
    AcquisitionManifest,
    FileRecord,
    load_manifest,
    write_manifest,
)

logger = logging.getLogger(__name__)

ACQUISITION_VERSION = "1"
DATASET_NAME = "NASA C-MAPSS Turbofan Engine Degradation Simulation"
SOURCE_NAME = "NASA Prognostics Center of Excellence data repository"
DEFAULT_SOURCE_URL = (
    "https://phm-datasets.s3.amazonaws.com/NASA/"
    "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
)
PROVENANCE_NOTES = (
    "Simulated turbofan engine run-to-failure data produced with the C-MAPSS "
    "simulator. Sensor channels are anonymous; no physical interpretation "
    "(such as vibration or temperature) is assigned by this project. Files "
    "are stored byte-for-byte as distributed."
)

_MAX_NESTED_ARCHIVE_DEPTH = 2
_DOWNLOAD_CHUNK_BYTES = 1024 * 1024
_READ_ONLY_MODE = 0o444


class AcquisitionStatus(StrEnum):
    """Outcome of an acquisition run."""

    ACQUIRED = "acquired"
    ALREADY_ACQUIRED = "already_acquired"


class AcquisitionError(RuntimeError):
    """Raised when dataset acquisition cannot be completed."""


@dataclass(frozen=True)
class AcquisitionConfig:
    """Inputs controlling one acquisition run."""

    data_dir: Path
    source_url: str = DEFAULT_SOURCE_URL
    subset: str = "FD001"
    force: bool = False
    timeout_seconds: float = 120.0

    @property
    def raw_dir(self) -> Path:
        """Directory holding the cached archive and extracted subsets."""
        return self.data_dir / "raw" / "cmapss"

    @property
    def subset_dir(self) -> Path:
        """Directory holding this subset's immutable raw files."""
        return self.raw_dir / self.subset

    @property
    def manifest_path(self) -> Path:
        """Location of this subset's acquisition manifest."""
        return self.data_dir / "manifests" / f"cmapss_{self.subset.lower()}.json"

    @property
    def archive_path(self) -> Path:
        """Local cache location for the downloaded source archive."""
        return self.raw_dir / _archive_filename(self.source_url)


@dataclass(frozen=True)
class AcquisitionResult:
    """Outcome of :func:`acquire`."""

    status: AcquisitionStatus
    manifest: AcquisitionManifest
    manifest_path: Path


@dataclass(frozen=True)
class _SubsetMember:
    """One file expected inside the source archive for a subset."""

    filename: str
    has_asset_column: bool


def _subset_members(subset: str) -> tuple[_SubsetMember, ...]:
    """C-MAPSS files belonging to ``subset``, per the dataset's own readme."""
    return (
        _SubsetMember(f"train_{subset}.txt", has_asset_column=True),
        _SubsetMember(f"test_{subset}.txt", has_asset_column=True),
        _SubsetMember(f"RUL_{subset}.txt", has_asset_column=False),
    )


def acquire(config: AcquisitionConfig) -> AcquisitionResult:
    """Acquire the configured C-MAPSS subset into the immutable raw layer.

    Idempotent: when the manifest exists and every raw file matches its
    recorded checksum, nothing is downloaded or rewritten. A mismatch raises
    :class:`AcquisitionError` unless ``config.force`` is set, in which case
    the archive is downloaded again and the raw layer is replaced atomically.
    """
    logger.info(
        "acquisition_started",
        extra={
            "subset": config.subset,
            "source_url": config.source_url,
            "data_dir": str(config.data_dir),
            "force": config.force,
        },
    )
    if not config.force:
        existing = verify_raw_layer(config)
        if existing is not None:
            logger.info(
                "acquisition_already_complete",
                extra={"manifest_path": str(config.manifest_path)},
            )
            return AcquisitionResult(
                status=AcquisitionStatus.ALREADY_ACQUIRED,
                manifest=existing,
                manifest_path=config.manifest_path,
            )

    archive_path = _ensure_archive(config)
    members = _extract_members(archive_path, config)
    file_records = _write_member_files(members, config)
    manifest = AcquisitionManifest(
        acquisition_version=ACQUISITION_VERSION,
        dataset_name=DATASET_NAME,
        dataset_subset=config.subset,
        source_name=SOURCE_NAME,
        source_url=config.source_url,
        retrieved_at=datetime.now(UTC),
        acquired_by=f"turbine-guard {__version__}",
        git_commit=current_git_commit(),
        archive=_archive_record(archive_path),
        files=file_records,
        notes=PROVENANCE_NOTES,
    )
    write_manifest(manifest, config.manifest_path)
    logger.info(
        "acquisition_complete",
        extra={
            "manifest_path": str(config.manifest_path),
            "file_count": len(file_records),
        },
    )
    return AcquisitionResult(
        status=AcquisitionStatus.ACQUIRED,
        manifest=manifest,
        manifest_path=config.manifest_path,
    )


def verify_raw_layer(config: AcquisitionConfig) -> AcquisitionManifest | None:
    """Return the manifest when the raw layer is complete and unmodified.

    Returns ``None`` when no manifest exists (fresh acquisition). Raises
    :class:`AcquisitionError` when the manifest or raw files are unreadable,
    missing, or fail checksum verification — the raw layer is immutable, so
    silent repair would hide corruption. Downstream processing uses this to
    confirm the acquisition state before reading raw files.
    """
    manifest_path = config.manifest_path
    if not manifest_path.exists():
        return None
    try:
        manifest = load_manifest(manifest_path)
    except (ValueError, OSError) as exc:
        raise AcquisitionError(
            f"Existing manifest {manifest_path} could not be read: {exc}. "
            "Re-run with --force to acquire the dataset again."
        ) from exc
    for record in manifest.files:
        path = config.subset_dir / record.filename
        if not path.exists():
            raise AcquisitionError(
                f"Manifest {manifest_path} exists but raw file {path} is missing. "
                "The raw layer is immutable; re-run with --force to acquire it again."
            )
        digest = _sha256_of(path)
        if digest != record.sha256:
            raise AcquisitionError(
                f"Checksum mismatch for {path}: the manifest records "
                f"{record.sha256} but the file hashes to {digest}. Raw files "
                "must never change; re-run with --force to replace the raw layer."
            )
    logger.info(
        "raw_files_verified",
        extra={"subset": config.subset, "file_count": len(manifest.files)},
    )
    return manifest


def _ensure_archive(config: AcquisitionConfig) -> Path:
    """Return the local source archive, downloading it when not cached."""
    archive_path = config.archive_path
    if archive_path.exists() and not config.force:
        logger.info("using_cached_archive", extra={"archive_path": str(archive_path)})
        return archive_path

    config.raw_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = archive_path.with_name(archive_path.name + ".download")
    request = urllib.request.Request(
        config.source_url,
        headers={"User-Agent": f"turbine-guard/{__version__}"},
    )
    logger.info("downloading_archive", extra={"source_url": config.source_url})
    try:
        with (
            urllib.request.urlopen(request, timeout=config.timeout_seconds) as response,
            tmp_path.open("wb") as output,
        ):
            shutil.copyfileobj(response, output, length=_DOWNLOAD_CHUNK_BYTES)
    except OSError as exc:
        tmp_path.unlink(missing_ok=True)
        raise AcquisitionError(
            f"Could not download the dataset archive from {config.source_url}: {exc}. "
            "Check network access, or point TURBINE_GUARD_CMAPSS_SOURCE_URL (or --url) "
            "at a mirror or a pre-downloaded archive using a file:// URL."
        ) from exc
    tmp_path.replace(archive_path)
    logger.info(
        "archive_downloaded",
        extra={
            "archive_path": str(archive_path),
            "size_bytes": archive_path.stat().st_size,
        },
    )
    return archive_path


def _extract_members(archive_path: Path, config: AcquisitionConfig) -> dict[str, bytes]:
    """Read the subset's files out of the archive, searching nested zips."""
    members = _subset_members(config.subset)
    targets = {member.filename.lower(): member.filename for member in members}
    try:
        with archive_path.open("rb") as stream:
            found = _find_members(stream, targets, depth=_MAX_NESTED_ARCHIVE_DEPTH)
    except zipfile.BadZipFile as exc:
        raise AcquisitionError(
            f"{archive_path} is not a valid zip archive: {exc}. "
            "Re-run with --force to download it again."
        ) from exc
    missing = sorted(set(targets.values()) - set(found))
    if missing:
        raise AcquisitionError(
            f"The source archive {archive_path} does not contain the expected "
            f"{config.subset} files: missing {', '.join(missing)}."
        )
    return found


def _find_members(stream: IO[bytes], targets: dict[str, str], depth: int) -> dict[str, bytes]:
    """Search a zip stream (and nested zips up to ``depth``) for target files."""
    found: dict[str, bytes] = {}
    nested: list[str] = []
    with zipfile.ZipFile(stream) as archive:
        for member in archive.namelist():
            basename = PurePosixPath(member).name
            if not basename or "__macosx" in member.lower():
                continue
            key = basename.lower()
            if key in targets:
                found.setdefault(targets[key], archive.read(member))
            elif key.endswith(".zip"):
                nested.append(member)
        if len(found) < len(targets) and depth > 0:
            for member in nested:
                inner = io.BytesIO(archive.read(member))
                for filename, data in _find_members(inner, targets, depth - 1).items():
                    found.setdefault(filename, data)
                if len(found) == len(targets):
                    break
    return found


def _write_member_files(
    members: dict[str, bytes], config: AcquisitionConfig
) -> tuple[FileRecord, ...]:
    """Atomically write raw files read-only and return their provenance records."""
    config.subset_dir.mkdir(parents=True, exist_ok=True)
    records: list[FileRecord] = []
    for member in _subset_members(config.subset):
        data = members[member.filename]
        target = config.subset_dir / member.filename
        tmp_path = target.with_name(member.filename + ".tmp")
        tmp_path.write_bytes(data)
        tmp_path.replace(target)
        target.chmod(_READ_ONLY_MODE)
        records.append(
            FileRecord(
                filename=member.filename,
                sha256=hashlib.sha256(data).hexdigest(),
                size_bytes=len(data),
                record_count=_count_records(data),
                asset_count=_count_assets(data) if member.has_asset_column else None,
            )
        )
        logger.info(
            "raw_file_written",
            extra={"path": str(target), "size_bytes": len(data)},
        )
    return tuple(records)


def _count_records(data: bytes) -> int:
    """Number of non-empty lines; content-neutral, no schema interpretation."""
    text = data.decode("utf-8", errors="replace")
    return sum(1 for line in text.splitlines() if line.strip())


def _count_assets(data: bytes) -> int:
    """Number of distinct unit IDs in the first whitespace-delimited column."""
    text = data.decode("utf-8", errors="replace")
    return len({line.split()[0] for line in text.splitlines() if line.strip()})


def _archive_record(path: Path) -> FileRecord:
    """Provenance record for the downloaded archive itself."""
    return FileRecord(
        filename=path.name,
        sha256=_sha256_of(path),
        size_bytes=path.stat().st_size,
    )


def _sha256_of(path: Path) -> str:
    """Streaming SHA-256 of a file on disk."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _archive_filename(source_url: str) -> str:
    """Derive a local cache filename from the source URL."""
    url_path = urllib.parse.unquote(urllib.parse.urlsplit(source_url).path)
    name = Path(url_path.replace("+", " ")).name.strip().replace(" ", "_")
    return name or "cmapss_source_archive.zip"


def current_git_commit() -> str | None:
    """Current git commit SHA, or ``None`` outside a repo or without commits."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None

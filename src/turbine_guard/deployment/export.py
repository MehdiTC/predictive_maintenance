"""Export the verified live champion as an immutable deployment bundle."""

import gzip
import io
import logging
import tarfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from turbine_guard.config.settings import Settings
from turbine_guard.data.acquisition import current_git_commit
from turbine_guard.deployment.manifest import (
    CHAMPION_RELATIVE_PATH,
    FEATURE_MANIFEST_RELATIVE_PATH,
    MANIFEST_FILENAME,
    REQUIRED_BUNDLE_FILES,
    BundleError,
    BundleFileRecord,
    DeploymentBundleManifest,
)
from turbine_guard.modeling.artifacts import sha256_bytes, sha256_path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExportResult:
    """Archive location, pin, and manifest for one completed export."""

    archive_path: Path
    archive_sha256: str
    manifest: DeploymentBundleManifest


def export_deployment_bundle(settings: Settings, output_path: Path) -> ExportResult:
    """Package the loaded, verified champion and its serving inputs.

    The export loads the champion through the normal MLflow serving path
    first, so a bundle can only be produced from exactly the artifact,
    feature contract, and registry identity the online API would serve.
    """
    from turbine_guard.serving.model_loader import ChampionModelLoader

    loader = ChampionModelLoader(settings)
    loaded = loader.get()
    alias_values = loader.registry_aliases()
    metadata = loaded.metadata

    data_dir = settings.data_dir
    records: list[BundleFileRecord] = []
    checksums: dict[Path, str] = {}
    for relative in REQUIRED_BUNDLE_FILES:
        path = data_dir / relative
        if not path.is_file():
            raise BundleError(f"Deployment bundle input {path} is missing.")
        digest = sha256_path(path)
        checksums[relative] = digest
        records.append(
            BundleFileRecord(
                relative_path=relative.as_posix(),
                sha256=digest,
                size_bytes=path.stat().st_size,
            )
        )

    if metadata.checksum is not None and checksums[CHAMPION_RELATIVE_PATH] != metadata.checksum:
        raise BundleError(
            "Local champion.joblib does not match the registered champion checksum; "
            "refusing to export a bundle that differs from the registry artifact."
        )
    if (
        metadata.feature_manifest_checksum is not None
        and checksums[FEATURE_MANIFEST_RELATIVE_PATH] != metadata.feature_manifest_checksum
    ):
        raise BundleError(
            "Local feature manifest does not match the checksum recorded with the "
            "registered champion; refusing to export an inconsistent bundle."
        )

    manifest = DeploymentBundleManifest(
        created_at=datetime.now(UTC),
        git_commit=current_git_commit(),
        registered_model_name=metadata.model_name,
        registry_version=metadata.version,
        champion_alias=metadata.alias,
        aliases=alias_values,
        source_run_id=metadata.source_run_id,
        model_family=metadata.model_family,
        target_definition=metadata.target_definition,
        rul_cap=metadata.rul_cap,
        feature_version=metadata.feature_version,
        feature_count=metadata.feature_count,
        validation_rmse=metadata.validation_rmse,
        replay_rmse=metadata.replay_rmse,
        official_test_rmse=metadata.official_test_rmse,
        conformal_coverage_target=metadata.conformal_coverage_target,
        champion_bundle_sha256=checksums[CHAMPION_RELATIVE_PATH],
        feature_manifest_sha256=checksums[FEATURE_MANIFEST_RELATIVE_PATH],
        dataset_checksum=metadata.dataset_checksum,
        lineage_id=metadata.lineage_id,
        files=tuple(records),
    )

    archive_bytes = _build_archive(data_dir, manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    tmp_path.write_bytes(archive_bytes)
    tmp_path.replace(output_path)
    archive_sha256 = sha256_bytes(archive_bytes)
    logger.info(
        "deployment_bundle_exported",
        extra={
            "archive": str(output_path),
            "archive_sha256": archive_sha256,
            "registry_version": manifest.registry_version,
            "file_count": len(manifest.files),
        },
    )
    return ExportResult(output_path, archive_sha256, manifest)


def _build_archive(data_dir: Path, manifest: DeploymentBundleManifest) -> bytes:
    """Build a normalized tar.gz so identical content yields identical bytes."""
    buffer = io.BytesIO()
    with (
        gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as gz,
        tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar,
    ):
        manifest_bytes = (manifest.model_dump_json(indent=2) + "\n").encode("utf-8")
        _add_member(tar, MANIFEST_FILENAME, manifest_bytes)
        for record in manifest.files:
            _add_member(tar, record.relative_path, (data_dir / record.relative_path).read_bytes())
    return buffer.getvalue()


def _add_member(tar: tarfile.TarFile, name: str, content: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(content)
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mode = 0o644
    tar.addfile(info, io.BytesIO(content))

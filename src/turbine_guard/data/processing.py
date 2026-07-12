"""Processing of acquired C-MAPSS raw files into validated Parquet outputs.

The pipeline is: verify the immutable raw layer against its acquisition
manifest → parse the raw files into typed frames → run structural and
semantic validation → write Parquet outputs and a machine-readable processing
report. A failed required validation blocks publication: no output file is
written.

Re-running is idempotent: when the existing report matches the current inputs
and the outputs on disk match their recorded checksums, nothing is rewritten.
Tampered or incomplete outputs raise :class:`ProcessingError` instead of
being silently repaired; ``force`` deliberately rebuilds the processed layer.
"""

import hashlib
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict

from turbine_guard import __version__
from turbine_guard.data.acquisition import (
    AcquisitionConfig,
    AcquisitionError,
    current_git_commit,
    verify_raw_layer,
)
from turbine_guard.data.manifest import AcquisitionManifest, FileRecord
from turbine_guard.data.parsing import ParseError, parse_rul_file, parse_trajectory_file
from turbine_guard.data.schema import ASSET_ID_COLUMN, SCHEMA_VERSION
from turbine_guard.data.validation import (
    CANONICAL_PROFILES,
    DatasetValidation,
    ValidationCheck,
    validate_rul_frame,
    validate_trajectory_frame,
)

logger = logging.getLogger(__name__)

PROCESSING_VERSION = "1"


class ProcessingStatus(StrEnum):
    """Outcome of a processing run."""

    PROCESSED = "processed"
    ALREADY_PROCESSED = "already_processed"


class ProcessingError(RuntimeError):
    """Raised when raw data cannot be processed into validated outputs."""


@dataclass(frozen=True)
class ProcessingConfig:
    """Inputs controlling one processing run."""

    data_dir: Path
    subset: str = "FD001"
    force: bool = False
    validate_canonical: bool = True
    """Apply the known canonical row/asset counts when the subset has one."""

    @property
    def acquisition_config(self) -> AcquisitionConfig:
        """Acquisition view of the same data directory, for verification."""
        return AcquisitionConfig(data_dir=self.data_dir, subset=self.subset)

    @property
    def processed_dir(self) -> Path:
        """Directory holding this subset's validated Parquet outputs."""
        return self.data_dir / "processed" / "cmapss" / self.subset

    @property
    def report_path(self) -> Path:
        """Location of the machine-readable processing report."""
        return self.processed_dir / "processing_report.json"

    def output_path(self, dataset: str) -> Path:
        """Parquet output path for one dataset (train, test, or rul)."""
        return self.processed_dir / f"{dataset}_{self.subset}.parquet"


@dataclass(frozen=True)
class ProcessingResult:
    """Outcome of :func:`process`."""

    status: ProcessingStatus
    report: "ProcessingReport"
    report_path: Path
    output_paths: tuple[Path, ...]


class ProcessingReport(BaseModel):
    """Machine-readable record of one processing run.

    Written next to the Parquet outputs; it is the source of truth for
    idempotency: re-runs compare inputs and outputs against the checksums
    recorded here.
    """

    model_config = ConfigDict(frozen=True)

    processing_version: str
    schema_version: str
    dataset_name: str
    dataset_subset: str
    processed_at: datetime
    processed_by: str
    git_commit: str | None
    source_archive_sha256: str
    """Checksum of the acquired source archive, linking to the manifest."""

    inputs: tuple[FileRecord, ...]
    outputs: tuple[FileRecord, ...]
    datasets: tuple[DatasetValidation, ...]
    cross_checks: tuple[ValidationCheck, ...]
    passed: bool
    warnings: tuple[str, ...]


_DATASETS = ("train", "test", "rul")


def process(config: ProcessingConfig) -> ProcessingResult:
    """Process the acquired subset into validated Parquet outputs.

    Raises :class:`ProcessingError` when the raw layer is missing or
    corrupted, parsing fails, a required validation check fails, or existing
    outputs were tampered with. Never modifies the raw layer.
    """
    logger.info(
        "processing_started",
        extra={
            "subset": config.subset,
            "data_dir": str(config.data_dir),
            "force": config.force,
        },
    )
    manifest = _verified_manifest(config)

    if not config.force:
        existing = _verify_existing_outputs(config, manifest)
        if existing is not None:
            logger.info(
                "processing_already_complete",
                extra={"report_path": str(config.report_path)},
            )
            return ProcessingResult(
                status=ProcessingStatus.ALREADY_PROCESSED,
                report=existing,
                report_path=config.report_path,
                output_paths=tuple(config.output_path(d) for d in _DATASETS),
            )

    frames = _parse_raw_files(config)
    validations, cross_checks, passed = _validate(config, frames)
    if not passed:
        failed = [
            f"{validation.dataset}:{check.name} ({check.message})"
            for validation in validations
            for check in validation.failed_checks
        ] + [f"cross:{check.name} ({check.message})" for check in cross_checks if not check.passed]
        raise ProcessingError(
            "Validation failed; no processed output was written. Failed checks: "
            + "; ".join(failed)
        )

    output_records = _write_outputs(config, frames)
    warnings = tuple(w for validation in validations for w in validation.warnings)
    report = ProcessingReport(
        processing_version=PROCESSING_VERSION,
        schema_version=SCHEMA_VERSION,
        dataset_name=manifest.dataset_name,
        dataset_subset=config.subset,
        processed_at=datetime.now(UTC),
        processed_by=f"turbine-guard {__version__}",
        git_commit=current_git_commit(),
        source_archive_sha256=manifest.archive.sha256,
        inputs=manifest.files,
        outputs=output_records,
        datasets=validations,
        cross_checks=cross_checks,
        passed=True,
        warnings=warnings,
    )
    _write_report(report, config.report_path)
    logger.info(
        "processing_complete",
        extra={
            "report_path": str(config.report_path),
            "output_count": len(output_records),
            "warning_count": len(warnings),
        },
    )
    return ProcessingResult(
        status=ProcessingStatus.PROCESSED,
        report=report,
        report_path=config.report_path,
        output_paths=tuple(config.output_path(d) for d in _DATASETS),
    )


def load_report(path: Path) -> ProcessingReport:
    """Load and validate a processing report from ``path``."""
    return ProcessingReport.model_validate_json(path.read_text(encoding="utf-8"))


def _verified_manifest(config: ProcessingConfig) -> AcquisitionManifest:
    """Verify the acquisition state and return its manifest."""
    try:
        manifest = verify_raw_layer(config.acquisition_config)
    except AcquisitionError as exc:
        raise ProcessingError(f"Raw layer verification failed: {exc}") from exc
    if manifest is None:
        raise ProcessingError(
            f"No acquisition manifest found for subset {config.subset} under "
            f"{config.data_dir}. Run the acquisition first "
            "(make acquire, or: uv run python scripts/download_data.py)."
        )
    return manifest


def _verify_existing_outputs(
    config: ProcessingConfig, manifest: AcquisitionManifest
) -> ProcessingReport | None:
    """Return the existing report when outputs are current and untampered.

    Returns ``None`` when no report exists (fresh processing). Raises
    :class:`ProcessingError` when the report is unreadable or an output file
    is missing or fails checksum verification.
    """
    report_path = config.report_path
    if not report_path.exists():
        return None
    try:
        report = load_report(report_path)
    except (ValueError, OSError) as exc:
        raise ProcessingError(
            f"Existing processing report {report_path} could not be read: {exc}. "
            "Re-run with --force to rebuild the processed layer."
        ) from exc
    if report.inputs != manifest.files or report.processing_version != PROCESSING_VERSION:
        logger.info(
            "processed_outputs_stale",
            extra={"report_path": str(report_path)},
        )
        return None
    for record in report.outputs:
        path = config.processed_dir / record.filename
        if not path.exists():
            raise ProcessingError(
                f"Processing report {report_path} exists but output {path} is "
                "missing. Re-run with --force to rebuild the processed layer."
            )
        digest = _sha256_of(path)
        if digest != record.sha256:
            raise ProcessingError(
                f"Checksum mismatch for {path}: the report records {record.sha256} "
                f"but the file hashes to {digest}. Re-run with --force to rebuild "
                "the processed layer."
            )
    logger.info(
        "processed_outputs_verified",
        extra={"subset": config.subset, "output_count": len(report.outputs)},
    )
    return report


def _parse_raw_files(config: ProcessingConfig) -> dict[str, pd.DataFrame]:
    """Parse the subset's raw files into typed frames."""
    raw_dir = config.acquisition_config.subset_dir
    try:
        return {
            "train": parse_trajectory_file(raw_dir / f"train_{config.subset}.txt"),
            "test": parse_trajectory_file(raw_dir / f"test_{config.subset}.txt"),
            "rul": parse_rul_file(raw_dir / f"RUL_{config.subset}.txt"),
        }
    except ParseError as exc:
        raise ProcessingError(f"Parsing failed: {exc}") from exc


def _validate(
    config: ProcessingConfig, frames: dict[str, pd.DataFrame]
) -> tuple[tuple[DatasetValidation, ...], tuple[ValidationCheck, ...], bool]:
    """Run general validation plus the optional canonical subset profile."""
    profile = CANONICAL_PROFILES.get(config.subset) if config.validate_canonical else None
    validations = (
        validate_trajectory_frame(
            frames["train"],
            dataset="train",
            expected_rows=profile.train_rows if profile else None,
            expected_assets=profile.train_assets if profile else None,
        ),
        validate_trajectory_frame(
            frames["test"],
            dataset="test",
            expected_rows=profile.test_rows if profile else None,
            expected_assets=profile.test_assets if profile else None,
        ),
        validate_rul_frame(
            frames["rul"],
            dataset="rul",
            expected_values=profile.rul_values if profile else None,
        ),
    )
    test_assets = int(frames["test"][ASSET_ID_COLUMN].nunique())
    rul_rows = len(frames["rul"])
    cross_checks = (
        ValidationCheck(
            name="rul_count_matches_test_assets",
            passed=rul_rows == test_assets,
            required=True,
            message=(
                "ok"
                if rul_rows == test_assets
                else f"RUL file has {rul_rows} values but the test set has {test_assets} assets"
            ),
        ),
    )
    passed = all(v.passed for v in validations) and all(c.passed for c in cross_checks)
    return validations, cross_checks, passed


def _write_outputs(
    config: ProcessingConfig, frames: dict[str, pd.DataFrame]
) -> tuple[FileRecord, ...]:
    """Atomically write the validated frames as Parquet and record checksums.

    Trajectory frames are sorted by (asset_id, cycle) for a deterministic
    output row order; source row order is never relied on for correctness.
    """
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    records: list[FileRecord] = []
    for dataset in _DATASETS:
        frame = frames[dataset]
        if dataset in ("train", "test"):
            frame = frame.sort_values([ASSET_ID_COLUMN, "cycle"], kind="stable").reset_index(
                drop=True
            )
        target = config.output_path(dataset)
        tmp_path = target.with_name(target.name + ".tmp")
        frame.to_parquet(tmp_path, engine="pyarrow", index=False)
        tmp_path.replace(target)
        asset_count = (
            int(frame[ASSET_ID_COLUMN].nunique()) if ASSET_ID_COLUMN in frame.columns else None
        )
        records.append(
            FileRecord(
                filename=target.name,
                sha256=_sha256_of(target),
                size_bytes=target.stat().st_size,
                record_count=len(frame),
                asset_count=asset_count,
            )
        )
        logger.info(
            "processed_file_written",
            extra={"path": str(target), "row_count": len(frame)},
        )
    return tuple(records)


def _write_report(report: ProcessingReport, path: Path) -> None:
    """Atomically write ``report`` to ``path`` as pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _sha256_of(path: Path) -> str:
    """Streaming SHA-256 of a file on disk."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

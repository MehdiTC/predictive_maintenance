"""Orchestration of RUL labels, asset-level splits, and features into outputs.

The build boundary is::

    validated FD001 trajectories (Loop 2 Parquet)
            -> RUL labels + asset-level split
            -> leakage-safe features (shared FeatureBuilder)
            -> model-ready Parquet per partition + split/feature manifests

It reuses the Loop 2 idempotency model: the Loop 2 processing report is the
source of truth for the inputs; a feature build verifies those inputs by
checksum, and on re-run compares the recorded inputs, configuration, and output
checksums. When everything matches, nothing is rewritten. Tampered or missing
outputs raise instead of being silently repaired; ``force`` deliberately
rebuilds.

No model training, scaling, feature selection, or imputation happens here.
Structurally undefined early-cycle feature values are left null and handled by
the Loop 4 model pipeline.
"""

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

import pandas as pd

from turbine_guard import __version__
from turbine_guard.data.acquisition import current_git_commit
from turbine_guard.data.manifest import FileRecord
from turbine_guard.data.processing import ProcessingReport, load_report
from turbine_guard.data.schema import ASSET_ID_COLUMN, CYCLE_COLUMN, SCHEMA_VERSION
from turbine_guard.features.builder import IDENTIFIER_COLUMNS, FeatureBuilder
from turbine_guard.features.config import (
    RUL_CAPPED_COLUMN,
    RUL_COLUMN,
    SPLIT_COLUMN,
    BuildConfig,
)
from turbine_guard.features.labels import (
    add_rul_labels,
    build_test_benchmark_labels,
    validate_rul_labels,
)
from turbine_guard.features.manifest import (
    FeatureConfigRecord,
    FeatureManifest,
    FeatureOutputRecord,
    SplitManifest,
    load_feature_manifest,
    sha256_of,
    write_manifest,
)
from turbine_guard.features.splits import (
    PARTITION_NAMES,
    AssetSplit,
    make_asset_split,
    split_of,
)

logger = logging.getLogger(__name__)

FEATURE_BUILD_VERSION = "1"

TRAIN_PARTITIONS: tuple[str, ...] = PARTITION_NAMES
TEST_FEATURES_SPLIT = "test"
TEST_LABELS_SPLIT = "test_labels"

_PROCESSED_DATASETS = ("train", "test", "rul")


class BuildStatus(StrEnum):
    """Outcome of a feature-build run."""

    BUILT = "built"
    ALREADY_BUILT = "already_built"


class FeatureBuildError(RuntimeError):
    """Raised when model-ready features cannot be produced."""


@dataclass(frozen=True)
class FeatureBuildConfig:
    """Inputs controlling one feature-build run."""

    data_dir: Path
    subset: str = "FD001"
    force: bool = False
    build: BuildConfig = field(default_factory=BuildConfig)

    @property
    def processed_dir(self) -> Path:
        """Directory holding the Loop 2 validated Parquet inputs."""
        return self.data_dir / "processed" / "cmapss" / self.subset

    @property
    def report_path(self) -> Path:
        """Location of the Loop 2 processing report (the input source of truth)."""
        return self.processed_dir / "processing_report.json"

    @property
    def features_dir(self) -> Path:
        """Directory holding this subset's model-ready feature outputs."""
        return self.data_dir / "features" / "cmapss" / self.subset

    @property
    def split_manifest_path(self) -> Path:
        return self.features_dir / "split_manifest.json"

    @property
    def feature_manifest_path(self) -> Path:
        return self.features_dir / "feature_manifest.json"

    def processed_input(self, dataset: str) -> Path:
        return self.processed_dir / f"{dataset}_{self.subset}.parquet"

    def output_path(self, name: str) -> Path:
        return self.features_dir / f"{name}.parquet"


@dataclass(frozen=True)
class FeatureBuildResult:
    """Outcome of :func:`build_features`."""

    status: BuildStatus
    split_manifest: SplitManifest
    feature_manifest: FeatureManifest
    feature_manifest_path: Path
    output_paths: tuple[Path, ...]


def build_features(config: FeatureBuildConfig) -> FeatureBuildResult:
    """Generate labels, splits, features, and manifests for the subset.

    Idempotent: when the feature manifest already reflects the current inputs
    and configuration and every output verifies by checksum, nothing is
    rewritten. Raises :class:`FeatureBuildError` when Loop 2 outputs are
    missing/tampered, generated labels fail validation, or existing outputs are
    tampered with. Never modifies the raw, validated, or processed layers.
    """
    logger.info(
        "feature_build_started",
        extra={
            "subset": config.subset,
            "data_dir": str(config.data_dir),
            "force": config.force,
        },
    )
    report, report_sha = _load_and_verify_inputs(config)
    inputs = report.outputs

    if not config.force:
        existing = _verify_existing_outputs(config, report_sha, inputs)
        if existing is not None:
            logger.info(
                "feature_build_already_complete",
                extra={"feature_manifest_path": str(config.feature_manifest_path)},
            )
            return existing

    return _rebuild(config, report, report_sha, inputs)


def _rebuild(
    config: FeatureBuildConfig,
    report: ProcessingReport,
    report_sha: str,
    inputs: tuple[FileRecord, ...],
) -> FeatureBuildResult:
    """Generate every output afresh and write both manifests."""
    train = pd.read_parquet(config.processed_input("train"))
    test = pd.read_parquet(config.processed_input("test"))
    official_rul = pd.read_parquet(config.processed_input("rul"))

    split = make_asset_split(train[ASSET_ID_COLUMN], config.build.split)
    labelled = add_rul_labels(train, config.build.rul)
    validate_rul_labels(labelled, config.build.rul)

    builder = FeatureBuilder(config.build.features)
    feature_columns = builder.feature_columns()
    target_columns = _target_columns(config)

    config.features_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[FeatureOutputRecord] = []
    row_counts: dict[str, int] = {}
    asset_counts: dict[str, int] = {}

    for partition in TRAIN_PARTITIONS:
        subset = split_of(labelled, split, partition)
        frame = _assemble_model_ready(builder, subset, partition, feature_columns, target_columns)
        record = _write_output(config, partition, frame, feature_columns, has_targets=True)
        outputs.append(record)
        row_counts[partition] = record.record_count or 0
        asset_counts[partition] = record.asset_count or 0

    test_frame = _assemble_model_ready(
        builder, test, TEST_FEATURES_SPLIT, feature_columns, target_columns=()
    )
    test_record = _write_output(
        config, TEST_FEATURES_SPLIT + "_features", test_frame, feature_columns, has_targets=False
    )
    outputs.append(test_record)
    row_counts[TEST_FEATURES_SPLIT] = test_record.record_count or 0
    asset_counts[TEST_FEATURES_SPLIT] = test_record.asset_count or 0

    benchmark = build_test_benchmark_labels(test, official_rul)
    bench_record = _write_output(
        config, TEST_LABELS_SPLIT, benchmark, feature_columns=(), has_targets=True
    )
    outputs.append(bench_record)

    split_manifest = _build_split_manifest(config, report, report_sha, split, labelled)
    write_manifest(split_manifest, config.split_manifest_path)
    split_manifest_sha = sha256_of(config.split_manifest_path)

    feature_manifest = _build_feature_manifest(
        config,
        dataset_name=report.dataset_name,
        report_sha=report_sha,
        split_manifest_sha=split_manifest_sha,
        inputs=inputs,
        outputs=tuple(outputs),
        feature_columns=feature_columns,
        target_columns=target_columns,
        row_counts=row_counts,
        asset_counts=asset_counts,
    )
    write_manifest(feature_manifest, config.feature_manifest_path)

    logger.info(
        "feature_build_complete",
        extra={
            "feature_manifest_path": str(config.feature_manifest_path),
            "output_count": len(outputs),
            "feature_count": len(feature_columns),
            "asset_counts": split.counts(),
        },
    )
    return FeatureBuildResult(
        status=BuildStatus.BUILT,
        split_manifest=split_manifest,
        feature_manifest=feature_manifest,
        feature_manifest_path=config.feature_manifest_path,
        output_paths=tuple(config.features_dir / record.filename for record in outputs),
    )


def _assemble_model_ready(
    builder: FeatureBuilder,
    frame: pd.DataFrame,
    split_name: str,
    feature_columns: tuple[str, ...],
    target_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Build the ordered model-ready frame for one partition.

    Column order is deterministic: identifiers, split metadata, targets (when
    present), then feature columns in :meth:`FeatureBuilder.feature_columns`
    order. Targets are joined on ``(asset_id, cycle)`` — never fabricated for
    the test set, which carries no per-row targets.
    """
    features = builder.transform(frame)
    features[SPLIT_COLUMN] = split_name
    if target_columns:
        labels = frame[[ASSET_ID_COLUMN, CYCLE_COLUMN, *target_columns]]
        features = features.merge(labels, on=list(IDENTIFIER_COLUMNS), how="left")
    ordered = [
        *IDENTIFIER_COLUMNS,
        SPLIT_COLUMN,
        *target_columns,
        *feature_columns,
    ]
    return features[ordered]


def _write_output(
    config: FeatureBuildConfig,
    name: str,
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    has_targets: bool,
) -> FeatureOutputRecord:
    """Atomically write one output Parquet and record its provenance."""
    target = config.output_path(name)
    tmp_path = target.with_name(target.name + ".tmp")
    frame.to_parquet(tmp_path, engine="pyarrow", index=False)
    tmp_path.replace(target)

    null_count = int(frame[list(feature_columns)].isna().sum().sum()) if feature_columns else 0
    asset_count = (
        int(frame[ASSET_ID_COLUMN].nunique()) if ASSET_ID_COLUMN in frame.columns else None
    )
    record = FeatureOutputRecord(
        filename=target.name,
        sha256=sha256_of(target),
        size_bytes=target.stat().st_size,
        record_count=len(frame),
        asset_count=asset_count,
        split=name,
        null_count=null_count,
        has_targets=has_targets,
    )
    logger.info(
        "feature_output_written",
        extra={"path": str(target), "row_count": len(frame), "null_count": null_count},
    )
    return record


def _target_columns(config: FeatureBuildConfig) -> tuple[str, ...]:
    if config.build.rul.produces_capped:
        return (RUL_COLUMN, RUL_CAPPED_COLUMN)
    return (RUL_COLUMN,)


def _build_split_manifest(
    config: FeatureBuildConfig,
    report: ProcessingReport,
    report_sha: str,
    split: AssetSplit,
    labelled: pd.DataFrame,
) -> SplitManifest:
    row_counts = {
        partition: int(labelled[ASSET_ID_COLUMN].isin(split.assets(partition)).sum())
        for partition in TRAIN_PARTITIONS
    }
    return SplitManifest(
        dataset_name=report.dataset_name,
        dataset_subset=config.subset,
        split_version=config.build.split.split_version,
        created_at=datetime.now(UTC),
        created_by=f"turbine-guard {__version__}",
        git_commit=current_git_commit(),
        seed=config.build.split.seed,
        strategy=config.build.split.strategy,
        source_report_sha256=report_sha,
        partitions=dict(split.partitions),
        asset_counts=split.counts(),
        row_counts=row_counts,
    )


def _build_feature_manifest(
    config: FeatureBuildConfig,
    *,
    dataset_name: str,
    report_sha: str,
    split_manifest_sha: str,
    inputs: tuple[FileRecord, ...],
    outputs: tuple[FeatureOutputRecord, ...],
    feature_columns: tuple[str, ...],
    target_columns: tuple[str, ...],
    row_counts: dict[str, int],
    asset_counts: dict[str, int],
) -> FeatureManifest:
    features = config.build.features
    return FeatureManifest(
        feature_build_version=FEATURE_BUILD_VERSION,
        schema_version=SCHEMA_VERSION,
        dataset_name=dataset_name,
        dataset_subset=config.subset,
        created_at=datetime.now(UTC),
        created_by=f"turbine-guard {__version__}",
        git_commit=current_git_commit(),
        seed=config.build.split.seed,
        source_report_sha256=report_sha,
        split_manifest_sha256=split_manifest_sha,
        split_version=config.build.split.split_version,
        feature_config=FeatureConfigRecord(
            feature_version=features.feature_version,
            source_columns=features.source_columns,
            families=features.families,
            windows=features.windows,
            ewm_spans=features.ewm_spans,
            min_periods=features.min_periods,
        ),
        rul_cap=config.build.rul.cap,
        imputation=None,
        identifier_columns=IDENTIFIER_COLUMNS,
        target_columns=target_columns,
        metadata_columns=(SPLIT_COLUMN,),
        feature_columns=feature_columns,
        inputs=inputs,
        outputs=outputs,
        row_counts_by_split=row_counts,
        asset_counts_by_split=asset_counts,
    )


def _load_and_verify_inputs(config: FeatureBuildConfig) -> tuple[ProcessingReport, str]:
    """Load the Loop 2 report and verify its Parquet outputs are untampered."""
    report_path = config.report_path
    if not report_path.exists():
        raise FeatureBuildError(
            f"No Loop 2 processing report found at {report_path}. Run processing first "
            "(make process, or: uv run python scripts/process_data.py)."
        )
    try:
        report = load_report(report_path)
    except (ValueError, OSError) as exc:
        raise FeatureBuildError(f"Loop 2 report {report_path} could not be read: {exc}") from exc
    if not report.passed:
        raise FeatureBuildError(
            f"Loop 2 report {report_path} did not pass validation; cannot build features."
        )
    for record in report.outputs:
        path = config.processed_dir / record.filename
        if not path.exists():
            raise FeatureBuildError(
                f"Loop 2 output {path} is missing. Re-run processing before building features."
            )
        digest = sha256_of(path)
        if digest != record.sha256:
            raise FeatureBuildError(
                f"Loop 2 output {path} does not match the processing report checksum "
                f"({record.sha256} expected, {digest} found). Re-run processing."
            )
    return report, sha256_of(report_path)


def _verify_existing_outputs(
    config: FeatureBuildConfig,
    report_sha: str,
    inputs: tuple[FileRecord, ...],
) -> FeatureBuildResult | None:
    """Return a result when existing outputs are current and untampered.

    Returns ``None`` when no feature manifest exists or the inputs/configuration
    have changed (triggering a rebuild). Raises :class:`FeatureBuildError` when
    the manifest is unreadable or an output is missing or fails its checksum.
    """
    manifest_path = config.feature_manifest_path
    if not manifest_path.exists():
        return None
    try:
        manifest = load_feature_manifest(manifest_path)
    except (ValueError, OSError) as exc:
        raise FeatureBuildError(
            f"Existing feature manifest {manifest_path} could not be read: {exc}. "
            "Re-run with --force to rebuild the feature layer."
        ) from exc

    if _configuration_changed(config, manifest, report_sha, inputs):
        logger.info("feature_outputs_stale", extra={"feature_manifest_path": str(manifest_path)})
        return None

    _verify_output_checksums(config, manifest)
    logger.info(
        "feature_outputs_verified",
        extra={"subset": config.subset, "output_count": len(manifest.outputs)},
    )
    return FeatureBuildResult(
        status=BuildStatus.ALREADY_BUILT,
        split_manifest=_load_split_manifest_or_fail(config),
        feature_manifest=manifest,
        feature_manifest_path=manifest_path,
        output_paths=tuple(config.features_dir / record.filename for record in manifest.outputs),
    )


def _configuration_changed(
    config: FeatureBuildConfig,
    manifest: FeatureManifest,
    report_sha: str,
    inputs: tuple[FileRecord, ...],
) -> bool:
    features = config.build.features
    expected = FeatureConfigRecord(
        feature_version=features.feature_version,
        source_columns=features.source_columns,
        families=features.families,
        windows=features.windows,
        ewm_spans=features.ewm_spans,
        min_periods=features.min_periods,
    )
    return (
        manifest.source_report_sha256 != report_sha
        or manifest.inputs != inputs
        or manifest.feature_build_version != FEATURE_BUILD_VERSION
        or manifest.feature_config != expected
        or manifest.rul_cap != config.build.rul.cap
        or manifest.seed != config.build.split.seed
        or manifest.split_version != config.build.split.split_version
    )


def _verify_output_checksums(config: FeatureBuildConfig, manifest: FeatureManifest) -> None:
    for record in manifest.outputs:
        path = config.features_dir / record.filename
        if not path.exists():
            raise FeatureBuildError(
                f"Feature manifest exists but output {path} is missing. "
                "Re-run with --force to rebuild the feature layer."
            )
        digest = sha256_of(path)
        if digest != record.sha256:
            raise FeatureBuildError(
                f"Checksum mismatch for {path}: the manifest records {record.sha256} "
                f"but the file hashes to {digest}. Re-run with --force to rebuild."
            )
    split_path = config.split_manifest_path
    if not split_path.exists() or sha256_of(split_path) != manifest.split_manifest_sha256:
        raise FeatureBuildError(
            f"Split manifest {split_path} is missing or does not match its recorded "
            "checksum. Re-run with --force to rebuild the feature layer."
        )


def _load_split_manifest_or_fail(config: FeatureBuildConfig) -> SplitManifest:
    from turbine_guard.features.manifest import load_split_manifest

    return load_split_manifest(config.split_manifest_path)

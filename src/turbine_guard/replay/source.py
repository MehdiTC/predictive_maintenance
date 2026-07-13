"""Checksum-verified access to held-out replay trajectories.

The replay data source is the validated Loop 2 trajectory Parquet restricted
to the Loop 3 ``replay`` split. Every load re-verifies the provenance chain

    feature manifest -> split manifest -> processing report -> train Parquet

by SHA-256 before any row is exposed, so a tampered or inconsistent input
fails before a partial replay can start. Only raw sensor columns are loaded;
replay RUL labels are never read here, and nothing in this module writes.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from turbine_guard.data.processing import ProcessingReport, load_report
from turbine_guard.data.schema import (
    ASSET_ID_COLUMN,
    CYCLE_COLUMN,
    TRAJECTORY_COLUMNS,
)
from turbine_guard.features.manifest import (
    SplitManifest,
    load_feature_manifest,
    load_split_manifest,
    sha256_of,
)
from turbine_guard.replay.errors import ReplaySourceError

logger = logging.getLogger(__name__)

REPLAY_PARTITION = "replay"


@dataclass(frozen=True)
class ReplaySourceConfig:
    """Location of the verified data layers a replay reads from."""

    data_dir: Path
    subset: str = "FD001"

    @property
    def features_dir(self) -> Path:
        return self.data_dir / "features" / "cmapss" / self.subset

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed" / "cmapss" / self.subset

    @property
    def split_manifest_path(self) -> Path:
        return self.features_dir / "split_manifest.json"

    @property
    def feature_manifest_path(self) -> Path:
        return self.features_dir / "feature_manifest.json"

    @property
    def report_path(self) -> Path:
        return self.processed_dir / "processing_report.json"

    @property
    def train_parquet_path(self) -> Path:
        return self.processed_dir / f"train_{self.subset}.parquet"


@dataclass(frozen=True, eq=False)
class ReplayTrajectory:
    """One verified held-out trajectory plus its provenance checksums.

    ``final_cycle`` is replay-internal ground truth: it must never be passed
    to the inference service before the trajectory has been fully replayed.
    """

    dataset_name: str
    dataset_subset: str
    source_asset_id: int
    final_cycle: int
    frame: pd.DataFrame = field(repr=False)
    source_checksums: dict[str, str] = field(default_factory=dict)

    def row(self, cycle: int) -> dict[str, float]:
        """Return exactly one observed cycle as a plain column/value mapping."""
        selected = self.frame.loc[self.frame[CYCLE_COLUMN] == cycle]
        if len(selected) != 1:
            raise ReplaySourceError(
                f"Cycle {cycle} of source asset {self.source_asset_id} is not uniquely "
                "present in the verified trajectory."
            )
        return {str(name): float(value) for name, value in selected.iloc[0].items()}


class ReplaySource:
    """Verified reader for replay-split assets."""

    def __init__(self, config: ReplaySourceConfig) -> None:
        self._config = config

    def verify(self) -> tuple[SplitManifest, ProcessingReport]:
        """Re-verify the full provenance chain; raise before exposing any data."""
        config = self._config
        for path in (
            config.feature_manifest_path,
            config.split_manifest_path,
            config.report_path,
            config.train_parquet_path,
        ):
            if not path.exists():
                raise ReplaySourceError(
                    f"Required replay input {path} is missing. Run the Loop 2/3 pipelines first."
                )
        try:
            feature_manifest = load_feature_manifest(config.feature_manifest_path)
            split_manifest = load_split_manifest(config.split_manifest_path)
            report = load_report(config.report_path)
        except (ValueError, OSError) as exc:
            raise ReplaySourceError(f"Replay provenance inputs could not be read: {exc}") from exc

        split_sha = sha256_of(config.split_manifest_path)
        if split_sha != feature_manifest.split_manifest_sha256:
            raise ReplaySourceError(
                "Split manifest does not match the checksum recorded by the feature "
                "manifest; the replay split may have been tampered with."
            )
        report_sha = sha256_of(config.report_path)
        if report_sha != split_manifest.source_report_sha256:
            raise ReplaySourceError(
                "Processing report does not match the checksum recorded by the split "
                "manifest; the validated layer may have been tampered with."
            )
        if not report.passed:
            raise ReplaySourceError("The recorded Loop 2 processing report did not pass.")
        record = next(
            (
                output
                for output in report.outputs
                if output.filename == config.train_parquet_path.name
            ),
            None,
        )
        if record is None:
            raise ReplaySourceError(
                f"Processing report does not record {config.train_parquet_path.name}."
            )
        parquet_sha = sha256_of(config.train_parquet_path)
        if parquet_sha != record.sha256:
            raise ReplaySourceError(
                f"Trajectory source {config.train_parquet_path} does not match its "
                "recorded checksum; refusing to replay tampered input."
            )
        if REPLAY_PARTITION not in split_manifest.partitions:
            raise ReplaySourceError("The split manifest defines no replay partition.")
        return split_manifest, report

    def replay_asset_ids(self) -> tuple[int, ...]:
        """Source asset IDs assigned to the replay split, in stable order."""
        split_manifest, _ = self.verify()
        return tuple(sorted(split_manifest.partitions[REPLAY_PARTITION]))

    def load_trajectory(self, source_asset_id: int) -> ReplayTrajectory:
        """Load and validate one replay-split trajectory."""
        config = self._config
        split_manifest, report = self.verify()
        partitions = split_manifest.partitions
        if source_asset_id not in partitions[REPLAY_PARTITION]:
            owner = next(
                (name for name, assets in partitions.items() if source_asset_id in assets),
                None,
            )
            if owner is None:
                raise ReplaySourceError(
                    f"Source asset {source_asset_id} does not exist in the split manifest."
                )
            raise ReplaySourceError(
                f"Source asset {source_asset_id} belongs to the {owner!r} split; only "
                "replay-split assets may be replayed."
            )

        frame = pd.read_parquet(config.train_parquet_path)
        if tuple(frame.columns) != TRAJECTORY_COLUMNS:
            raise ReplaySourceError(
                "Trajectory source columns do not match the canonical sensor schema."
            )
        trajectory = (
            frame.loc[frame[ASSET_ID_COLUMN] == source_asset_id]
            .sort_values(CYCLE_COLUMN, kind="stable")
            .reset_index(drop=True)
        )
        if trajectory.empty:
            raise ReplaySourceError(
                f"Source asset {source_asset_id} has no rows in the trajectory source."
            )
        cycles = trajectory[CYCLE_COLUMN].to_list()
        final_cycle = int(cycles[-1])
        if cycles != list(range(1, final_cycle + 1)):
            raise ReplaySourceError(
                f"Source asset {source_asset_id} cycles are not contiguous from 1; "
                "refusing to replay a malformed trajectory."
            )
        values = trajectory.drop(columns=[ASSET_ID_COLUMN, CYCLE_COLUMN])
        try:
            finite = bool(np.isfinite(values.to_numpy(dtype="float64")).all())
        except (TypeError, ValueError) as exc:
            raise ReplaySourceError(
                f"Source asset {source_asset_id} contains non-numeric sensor values."
            ) from exc
        if not finite:
            raise ReplaySourceError(
                f"Source asset {source_asset_id} contains non-finite sensor values."
            )
        logger.info(
            "replay_trajectory_loaded",
            extra={
                "dataset_subset": config.subset,
                "source_asset_id": source_asset_id,
                "cycle_count": final_cycle,
            },
        )
        return ReplayTrajectory(
            dataset_name=report.dataset_name,
            dataset_subset=config.subset,
            source_asset_id=source_asset_id,
            final_cycle=final_cycle,
            frame=trajectory,
            source_checksums={
                "processing_report_sha256": split_manifest.source_report_sha256,
                "split_manifest_sha256": sha256_of(config.split_manifest_path),
                "trajectory_parquet_sha256": sha256_of(config.train_parquet_path),
            },
        )

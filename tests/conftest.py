"""Shared test fixtures."""

import io
import zipfile
from collections.abc import Callable, Iterator
from pathlib import Path

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from turbine_guard.api.app import create_app
from turbine_guard.config.settings import Environment, Settings
from turbine_guard.data.acquisition import AcquisitionConfig, acquire
from turbine_guard.data.processing import ProcessingConfig, process
from turbine_guard.data.schema import SENSOR_COLUMNS, TRAJECTORY_COLUMNS, TRAJECTORY_DTYPES


def make_trajectory_line(asset_id: int, cycle: int) -> str:
    """One deterministic 26-field raw line, with real-file trailing spaces.

    ``sensor_01`` and ``sensor_05`` are constant, ``sensor_06`` is
    near-constant, everything else varies with asset and cycle, mirroring the
    real dataset's mix of informative and uninformative channels.
    """
    settings = [f"{0.001 * (asset_id + cycle):.4f}", f"{-0.0002 * cycle:.4f}", "100.0"]
    sensors: list[str] = []
    for index in range(1, len(SENSOR_COLUMNS) + 1):
        if index in (1, 5):
            sensors.append("518.67")
        elif index == 6:
            sensors.append("21.61" if cycle % 2 else "21.6101")
        else:
            sensors.append(f"{100 + index + 0.5 * cycle + 0.1 * asset_id:.2f}")
    return " ".join([str(asset_id), str(cycle), *settings, *sensors]) + "  "


def make_trajectory_text(trajectory_lengths: dict[int, int]) -> str:
    """Raw trajectory file content: each asset runs cycles 1..n contiguously."""
    lines = [
        make_trajectory_line(asset_id, cycle)
        for asset_id, length in trajectory_lengths.items()
        for cycle in range(1, length + 1)
    ]
    return "\n".join(lines) + "\n"


def make_trajectory_frame(trajectory_lengths: dict[int, int]) -> pd.DataFrame:
    """Canonical typed trajectory frame for offline label/feature unit tests.

    Uses the same deterministic per-cell values as :func:`make_trajectory_line`
    so constant and near-constant sensors are represented, then applies the
    canonical dtypes. No files or acquisition are involved.
    """
    rows = [
        make_trajectory_line(asset_id, cycle).split()
        for asset_id, length in trajectory_lengths.items()
        for cycle in range(1, length + 1)
    ]
    frame = pd.DataFrame(rows, columns=list(TRAJECTORY_COLUMNS), dtype="object")
    return frame.astype(TRAJECTORY_DTYPES)


def _feature_fixture_contents() -> dict[str, str]:
    """20 train assets (lengths 21..40) and 20 truncated test assets (11..30).

    Twenty training assets partition cleanly under the default 70/15/5/10 split
    into 14/3/1/2 assets, which keeps split assertions exact.
    """
    train_lengths = {asset: 20 + asset for asset in range(1, 21)}
    test_lengths = {asset: 10 + asset for asset in range(1, 21)}
    rul_values = "\n".join(str(40 + asset) for asset in range(1, 21)) + "\n"
    return {
        "train_FD001.txt": make_trajectory_text(train_lengths),
        "test_FD001.txt": make_trajectory_text(test_lengths),
        "RUL_FD001.txt": rul_values,
    }


@pytest.fixture
def feature_fixture_contents() -> dict[str, str]:
    """Schema-complete FD001 stand-ins sized for asset-level split tests."""
    return _feature_fixture_contents()


@pytest.fixture
def processed_data_dir(
    tmp_path: Path,
    cmapss_archive_factory: Callable[..., Path],
    feature_fixture_contents: dict[str, str],
) -> Path:
    """A data directory with the split-sized FD001 fixture acquired and processed.

    Produces the on-disk Loop 2 Parquet outputs and processing report that the
    feature pipeline and CLI consume, without the real dataset or the network.
    """
    archive = cmapss_archive_factory(contents=feature_fixture_contents)
    data_dir = tmp_path / "data"
    acquire(AcquisitionConfig(data_dir=data_dir, source_url=archive.as_uri()))
    process(ProcessingConfig(data_dir=data_dir, validate_canonical=False))
    return data_dir


@pytest.fixture
def app_settings() -> Settings:
    """Explicit settings so tests do not depend on the ambient environment."""
    return Settings(environment=Environment.TESTING, log_level="WARNING")


@pytest.fixture
def app(app_settings: Settings) -> FastAPI:
    """Application instance built from explicit test settings."""
    return create_app(app_settings)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    """HTTP client running the application in-process."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def cmapss_member_contents() -> dict[str, str]:
    """Tiny deterministic stand-ins for the C-MAPSS FD001 files.

    Trajectory files: two engine units (first column) with a few cycles each.
    RUL file: one value per test unit. Small enough to assert exact counts.
    """
    return {
        "train_FD001.txt": (
            "1 1 -0.0007 0.0003 100.0 518.67 641.82\n"
            "1 2 0.0019 -0.0003 100.0 518.67 642.15\n"
            "1 3 -0.0043 0.0003 100.0 518.67 642.35\n"
            "2 1 0.0007 0.0001 100.0 518.67 641.71\n"
            "2 2 -0.0016 -0.0002 100.0 518.67 642.46\n"
        ),
        "test_FD001.txt": (
            "1 1 0.0023 0.0003 100.0 518.67 643.02\n"
            "1 2 -0.0027 -0.0003 100.0 518.67 641.71\n"
            "2 1 0.0003 0.0001 100.0 518.67 642.46\n"
        ),
        "RUL_FD001.txt": "112\n98\n",
    }


@pytest.fixture
def full_cmapss_contents() -> dict[str, str]:
    """Schema-complete FD001 stand-ins: 26-column trajectories, 1-column RUL."""
    return {
        "train_FD001.txt": make_trajectory_text({1: 4, 2: 3}),
        "test_FD001.txt": make_trajectory_text({1: 3, 2: 2}),
        "RUL_FD001.txt": "112 \n98 \n",
    }


@pytest.fixture
def cmapss_archive_factory(
    tmp_path: Path, cmapss_member_contents: dict[str, str]
) -> Callable[..., Path]:
    """Build small fake C-MAPSS zip archives for offline acquisition tests."""

    def build(
        *,
        nested: bool = False,
        omit: tuple[str, ...] = (),
        name: str = "CMAPSSData.zip",
        contents: dict[str, str] | None = None,
    ) -> Path:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for filename, content in (contents or cmapss_member_contents).items():
                if filename not in omit:
                    archive.writestr(filename, content)
        archive_path = tmp_path / ("outer.zip" if nested else name)
        if nested:
            with zipfile.ZipFile(archive_path, "w") as outer:
                outer.writestr(
                    "6. Turbofan Engine Degradation Simulation Data Set/CMAPSSData.zip",
                    buffer.getvalue(),
                )
        else:
            archive_path.write_bytes(buffer.getvalue())
        return archive_path

    return build


@pytest.fixture
def acquired_data_dir(
    tmp_path: Path,
    cmapss_archive_factory: Callable[..., Path],
    full_cmapss_contents: dict[str, str],
) -> Path:
    """A data directory with the schema-complete FD001 fixture acquired."""
    archive = cmapss_archive_factory(contents=full_cmapss_contents)
    data_dir = tmp_path / "data"
    acquire(AcquisitionConfig(data_dir=data_dir, source_url=archive.as_uri()))
    return data_dir

"""Shared test fixtures."""

import io
import zipfile
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from turbine_guard.api.app import create_app
from turbine_guard.config.settings import Environment, Settings


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
def cmapss_archive_factory(
    tmp_path: Path, cmapss_member_contents: dict[str, str]
) -> Callable[..., Path]:
    """Build small fake C-MAPSS zip archives for offline acquisition tests."""

    def build(
        *,
        nested: bool = False,
        omit: tuple[str, ...] = (),
        name: str = "CMAPSSData.zip",
    ) -> Path:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for filename, content in cmapss_member_contents.items():
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

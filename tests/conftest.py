"""Shared test fixtures."""

from collections.abc import Iterator

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

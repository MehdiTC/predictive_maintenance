"""Guarded fixtures for real PostgreSQL operational integration tests."""

import os
from collections.abc import Iterator

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from turbine_guard.database.session import DatabaseConfig, create_database_engine


def _test_url() -> str:
    value = os.getenv("TURBINE_GUARD_DATABASE_TEST_URL")
    if value is None:
        pytest.skip("TURBINE_GUARD_DATABASE_TEST_URL is not configured")
    url = make_url(value)
    database = url.database or ""
    if url.drivername != "postgresql+psycopg" or "test" not in database.lower():
        pytest.fail(
            "PostgreSQL integration tests require postgresql+psycopg and a database name "
            "containing 'test'."
        )
    return value


@pytest.fixture(scope="session")
def postgres_engine() -> Iterator[Engine]:
    url = _test_url()
    os.environ["TURBINE_GUARD_DATABASE_URL"] = url
    engine = create_database_engine(DatabaseConfig(url=url))
    with engine.connect() as connection:
        name = connection.scalar(text("SELECT current_database()"))
        if not isinstance(name, str) or "test" not in name.lower():
            pytest.fail("Connected PostgreSQL database name does not contain 'test'.")
    alembic = Config("alembic.ini")
    command.downgrade(alembic, "base")
    command.upgrade(alembic, "head")
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(postgres_engine: Engine) -> Iterator[Session]:
    connection = postgres_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection, expire_on_commit=False, autoflush=False)
    try:
        yield session
    finally:
        session.close()
        if transaction.is_active:
            transaction.rollback()
        connection.close()

"""Explicit engine, session, transaction, and connectivity lifecycle."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.orm import Session, sessionmaker

from turbine_guard.config.settings import Settings


@dataclass(frozen=True)
class DatabaseConfig:
    """Engine settings independent of the global application configuration."""

    url: str
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout_seconds: float = 30.0
    pool_recycle_seconds: int = 1800
    connect_timeout_seconds: int = 5
    statement_timeout_ms: int = 30_000
    echo: bool = False

    def __post_init__(self) -> None:
        parsed = make_url(self.url)
        if parsed.drivername != "postgresql+psycopg":
            raise ValueError("Operational database URL must use postgresql+psycopg://.")
        if not parsed.database:
            raise ValueError("Operational database URL must include a database name.")
        if self.pool_size < 1 or self.max_overflow < 0:
            raise ValueError("Database pool_size must be positive and max_overflow non-negative.")
        if (
            min(
                self.pool_timeout_seconds,
                self.pool_recycle_seconds,
                self.connect_timeout_seconds,
                self.statement_timeout_ms,
            )
            <= 0
        ):
            raise ValueError("Database timeout values must be positive.")

    @classmethod
    def from_settings(cls, settings: Settings, *, test: bool = False) -> "DatabaseConfig":
        """Build engine configuration, optionally requiring the guarded test URL."""
        url = settings.database_test_url if test else settings.database_url
        if url is None:
            raise ValueError("TURBINE_GUARD_DATABASE_TEST_URL is required for database tests.")
        return cls(
            url=url,
            pool_size=settings.database_pool_size,
            max_overflow=settings.database_max_overflow,
            pool_timeout_seconds=settings.database_pool_timeout_seconds,
            pool_recycle_seconds=settings.database_pool_recycle_seconds,
            connect_timeout_seconds=settings.database_connect_timeout_seconds,
            statement_timeout_ms=settings.database_statement_timeout_ms,
            echo=settings.database_echo,
        )

    @property
    def parsed_url(self) -> URL:
        return make_url(self.url)


def create_database_engine(config: DatabaseConfig) -> Engine:
    """Create a lazy SQLAlchemy engine; no connection is opened here."""
    return create_engine(
        config.url,
        echo=config.echo,
        pool_pre_ping=True,
        pool_size=config.pool_size,
        max_overflow=config.max_overflow,
        pool_timeout=config.pool_timeout_seconds,
        pool_recycle=config.pool_recycle_seconds,
        connect_args={
            "connect_timeout": config.connect_timeout_seconds,
            "options": f"-c statement_timeout={config.statement_timeout_ms}",
        },
    )


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Create an explicit future FastAPI-compatible session factory."""
    return sessionmaker(bind=engine, class_=Session, expire_on_commit=False, autoflush=False)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    """Own one transaction, committing on success and rolling back on error."""
    with factory() as session, session.begin():
        yield session


def check_database_connection(engine: Engine) -> bool:
    """Return whether a trivial PostgreSQL query succeeds; never leak driver errors."""
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception:  # database drivers expose several unrelated exception families
        return False
    return True

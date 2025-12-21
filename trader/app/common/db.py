"""
Database connection and session management.
Supports SQLite (default) and PostgreSQL.
"""

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker, declarative_base

from trader.app.common.config import get_config

Base = declarative_base()

_engine = None
_SessionLocal = None


def get_engine():
    """Get or create database engine."""
    global _engine
    if _engine is None:
        config = get_config()
        # Handle SQLite vs PostgreSQL connection args
        if config.database_url.startswith("sqlite"):
            _engine = create_engine(
                config.database_url,
                connect_args={"check_same_thread": False},
                echo=False,
            )
        else:
            _engine = create_engine(config.database_url, echo=False, pool_pre_ping=True)
    return _engine


def get_session_factory():
    """Get or create session factory."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=get_engine()
        )
    return _SessionLocal


def init_db() -> None:
    """Initialize database tables."""
    from trader.app.common.models import Trade, Order, Fill, PnL  # noqa: F401

    engine = get_engine()
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Get database session as generator (for FastAPI dependency injection)."""
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Get database session as context manager."""
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def reset_db() -> None:
    """Reset database connections (for testing)."""
    global _engine, _SessionLocal
    if _engine:
        _engine.dispose()
    _engine = None
    _SessionLocal = None

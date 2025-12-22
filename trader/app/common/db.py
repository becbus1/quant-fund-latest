"""
Database connection and session management.
Primary target: Supabase Postgres.
Fallback: SQLite (local dev only).
"""

import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker, declarative_base
from sqlalchemy.engine.url import make_url

Base = declarative_base()

_engine = None
_SessionLocal = None


def _resolve_database_url() -> str:
    """
    Resolve database URL with strict priority:
    1. DATABASE_URL env var (Supabase / Railway / prod)
    2. config.database_url (local dev fallback)
    """
    if "DATABASE_URL" in os.environ:
        return os.environ["DATABASE_URL"]

    # Fallback only for local dev
    from trader.app.common.config import get_config
    return get_config().database_url


def get_engine():
    """Create (or reuse) SQLAlchemy engine."""
    global _engine

    if _engine is None:
        database_url = _resolve_database_url()
        url = make_url(database_url)

        if url.drivername.startswith("sqlite"):
            # ⚠️ SQLite = local dev only
            _engine = create_engine(
                database_url,
                connect_args={"check_same_thread": False},
                echo=False,
            )
        else:
            # ✅ Supabase / Postgres
            _engine = create_engine(
                database_url,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,
                pool_recycle=300,
                echo=False,
            )

    return _engine


def get_session_factory():
    """Create (or reuse) session factory."""
    global _SessionLocal

    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=get_engine(),
        )

    return _SessionLocal


def init_db() -> None:
    """
    Initialize DB schema.

    ❗ Supabase schema is managed externally.
    ❗ Do NOT run create_all against Supabase.
    """
    pass


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency-style session."""
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Context-managed DB session (used by trading engine)."""
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
    """Reset DB connections (tests / hot reload)."""
    global _engine, _SessionLocal

    if _engine:
        _engine.dispose()

    _engine = None
    _SessionLocal = None

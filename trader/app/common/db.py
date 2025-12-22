"""
Database connection and session management.

Primary target: Supabase Postgres (HTTP client only).
SQLAlchemy is DISABLED in production to prevent accidental SQLite usage.
"""

import os
from contextlib import contextmanager
from typing import Generator

# ⛔ SQLAlchemy intentionally disabled
# from sqlalchemy import create_engine
# from sqlalchemy.orm import Session, sessionmaker, declarative_base
# from sqlalchemy.engine.url import make_url

# Base = declarative_base()

_engine = None
_SessionLocal = None


def _resolve_database_url() -> str:
    """
    Resolve database URL.

    NOTE:
    This is intentionally unused while SQLAlchemy is disabled.
    """
    if "DATABASE_URL" in os.environ:
        return os.environ["DATABASE_URL"]

    from trader.app.common.config import get_config
    return get_config().database_url


def get_engine():
    """
    SQLAlchemy engine creation is DISABLED.

    Any attempt to use SQLAlchemy in production is a bug.
    """
    raise RuntimeError(
        "SQLAlchemy is disabled (Supabase-only mode). "
        "Do not call get_engine()."
    )


def get_session_factory():
    """
    SQLAlchemy session factory is DISABLED.
    """
    raise RuntimeError(
        "SQLAlchemy is disabled (Supabase-only mode). "
        "Do not call get_session_factory()."
    )


def init_db() -> None:
    """
    DB schema is managed externally (Supabase).

    Intentionally a no-op.
    """
    return


def get_db() -> Generator[None, None, None]:
    """
    FastAPI dependency-style session.

    Disabled — yields None.
    """
    yield None


@contextmanager
def get_db_session() -> Generator[None, None, None]:
    """
    Context-managed DB session.

    Disabled — yields None.
    """
    yield None


def reset_db() -> None:
    """
    Reset DB connections.

    No-op while SQLAlchemy is disabled.
    """
    global _engine, _SessionLocal
    _engine = None
    _SessionLocal = None

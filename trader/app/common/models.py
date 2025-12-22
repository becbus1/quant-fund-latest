"""
Database models (SQLAlchemy DISABLED).

These classes are now inert placeholders to prevent
accidental ORM usage in production.

Supabase is the source of truth.
"""

from datetime import datetime
import uuid

from shared.schemas import Side, OrderStatus, PositionStatus


class Trade:
    """Raw trade data from exchange."""
    # SQLAlchemy model disabled
    pass


class Order:
    """Paper trading orders."""
    # SQLAlchemy model disabled
    pass


class Fill:
    """Paper trading fills."""
    # SQLAlchemy model disabled
    pass


class PnL:
    """Realized PnL tracking."""
    # SQLAlchemy model disabled
    pass


class Position:
    """Open positions tracking."""
    # SQLAlchemy model disabled
    pass


class SignalLog:
    """Signal snapshots for ML training."""
    # SQLAlchemy model disabled
    pass

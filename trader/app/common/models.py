"""
SQLAlchemy models for trader database.
Tables: trades, orders, fills, pnl
"""

from datetime import datetime
import uuid

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Enum as SQLEnum,
    ForeignKey,
    Index,
    JSON,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from trader.app.common.db import Base
from shared.schemas import Side, OrderStatus, PositionStatus


class Trade(Base):
    """Raw trade data from exchange."""

    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    price = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)
    side = Column(SQLEnum(Side), nullable=False)
    trade_id = Column(String(50), nullable=False, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (Index("ix_trades_symbol_timestamp", "symbol", "timestamp"),)


class Order(Base):
    """Paper trading orders."""

    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(String(50), nullable=False, unique=True, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(SQLEnum(Side), nullable=False)
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    notional = Column(Float, nullable=False)
    status = Column(SQLEnum(OrderStatus), nullable=False, default=OrderStatus.PENDING)
    strategy_name = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    fills = relationship("Fill", back_populates="order")


class Fill(Base):
    """Paper trading fills."""

    __tablename__ = "fills"

    id = Column(Integer, primary_key=True, autoincrement=True)
    fill_id = Column(String(50), nullable=False, unique=True, index=True)
    order_id = Column(String(50), ForeignKey("orders.order_id"), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(SQLEnum(Side), nullable=False)
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    notional = Column(Float, nullable=False)
    fee = Column(Float, nullable=False, default=0.0)
    slippage = Column(Float, nullable=False, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)

    order = relationship("Order", back_populates="fills")


class PnL(Base):
    """Realized PnL tracking."""

    __tablename__ = "pnl"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    strategy_name = Column(String(100), nullable=True)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)
    side = Column(SQLEnum(Side), nullable=False)
    gross_pnl = Column(Float, nullable=False)
    fees = Column(Float, nullable=False)
    net_pnl = Column(Float, nullable=False)
    pnl_bps = Column(Float, nullable=False)
    hold_time_sec = Column(Float, nullable=False)
    exit_reason = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (Index("ix_pnl_symbol_timestamp", "symbol", "timestamp"),)


class Position(Base):
    """Open positions tracking."""

    __tablename__ = "positions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, unique=True, index=True)
    side = Column(SQLEnum(Side), nullable=False)
    quantity = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    notional = Column(Float, nullable=False)
    entry_time = Column(DateTime, nullable=False)
    strategy_name = Column(String(100), nullable=True)
    status = Column(
        SQLEnum(PositionStatus), nullable=False, default=PositionStatus.OPEN
    )
    take_profit_price = Column(Float, nullable=True)
    stop_loss_price = Column(Float, nullable=True)
    time_stop_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# 🧠 Signal snapshots for ML training
class SignalLog(Base):
    __tablename__ = "signal_logs"

    signal_id = Column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )

    symbol = Column(String, nullable=False, index=True)
    strategy = Column(String, nullable=False)

    # Frozen decision context
    features = Column(JSON, nullable=False)
    z_score = Column(Float, nullable=False)

    entry_price = Column(Float, nullable=False)

    timestamp = Column(
        DateTime(timezone=True),
        nullable=False,
        default=func.now(),
        index=True,
    )

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=func.now(),
    )

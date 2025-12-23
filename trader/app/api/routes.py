"""
FastAPI routes for trader API.
Endpoints: /health, /metrics, /killswitch
"""

import logging
from datetime import datetime
from typing import Dict, List

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from trader.app.common.config import get_config
from trader.app.common.db import get_db_session
from trader.app.common.models import PnL
from shared.schemas import PositionStatus

logger = logging.getLogger(__name__)

router = APIRouter()

# Will be set by main.py
_executor = None
_risk_manager = None
_ingester = None


def set_components(executor, risk_manager, ingester):
    """Set component references from main app."""
    global _executor, _risk_manager, _ingester
    _executor = executor
    _risk_manager = risk_manager
    _ingester = ingester


# =======================
# Response Models
# =======================

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    mode: str
    symbols: List[str]
    kill_switch_active: bool


class MetricsResponse(BaseModel):
    timestamp: str
    daily_pnl: float
    total_pnl: float
    unrealized_pnl: float
    open_positions: int
    total_trades: int
    win_rate: float
    avg_pnl_bps: float
    risk_status: Dict


class PositionResponse(BaseModel):
    symbol: str
    side: str
    quantity: float
    entry_price: float
    unrealized_pnl: float
    entry_time: str
    strategy_name: str


class KillSwitchResponse(BaseModel):
    status: str
    message: str
    positions_closed: int
    total_pnl: float


# =======================
# Routes
# =======================

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    config = get_config()
    kill_switch_active = _risk_manager.is_kill_switch_active() if _risk_manager else False

    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        mode=config.mode,
        symbols=config.symbols,
        kill_switch_active=kill_switch_active,
    )


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get trading metrics."""
    if not _executor or not _risk_manager:
        raise HTTPException(status_code=503, detail="System not initialized")

    # Realized PnL
    daily_pnl = _executor.get_daily_pnl()
    total_pnl = _executor.get_total_pnl()

    # Unrealized PnL
    prices = _get_current_prices()
    unrealized_pnl = _executor.get_unrealized_pnl(prices)

    # Trade statistics (DB REMOVED — Supabase-only mode)
    total_trades = 0
    win_rate = 0.0
    avg_pnl_bps = 0.0

    # ✅ SAFE: snapshots only
    open_positions = len(_executor.get_all_position_snapshots())
    risk_status = _risk_manager.get_risk_status()

    return MetricsResponse(
        timestamp=datetime.utcnow().isoformat(),
        daily_pnl=daily_pnl,
        total_pnl=total_pnl,
        unrealized_pnl=unrealized_pnl,
        open_positions=open_positions,
        total_trades=total_trades,
        win_rate=win_rate,
        avg_pnl_bps=avg_pnl_bps,
        risk_status=risk_status,
    )


@router.get("/positions", response_model=List[PositionResponse])
async def get_positions():
    """Get all open positions (SAFE snapshots)."""
    if not _executor:
        raise HTTPException(status_code=503, detail="System not initialized")

    positions = _executor.get_all_position_snapshots()
    prices = _get_current_prices()

    result = []
    for pos in positions:
        current_price = prices.get(pos["symbol"], pos["entry_price"])

        if pos["side"].value == "buy":
            unrealized = (current_price - pos["entry_price"]) * pos["quantity"]
        else:
            unrealized = (pos["entry_price"] - current_price) * pos["quantity"]

        result.append(
            PositionResponse(
                symbol=pos["symbol"],
                side=pos["side"].value,
                quantity=pos["quantity"],
                entry_price=pos["entry_price"],
                unrealized_pnl=unrealized,
                entry_time=pos["entry_time"].isoformat(),
                strategy_name=pos["strategy_name"] or "",
            )
        )

    return result


@router.get("/killswitch", response_model=KillSwitchResponse)
async def activate_killswitch(token: str = Query(...)):
    """Activate kill switch to close all positions."""
    config = get_config()

    if token != config.killswitch_token:
        logger.warning("Kill switch activation attempted with invalid token")
        raise HTTPException(status_code=403, detail="Invalid token")

    if not _executor or not _risk_manager:
        raise HTTPException(status_code=503, detail="System not initialized")

    logger.critical("Kill switch activated via API")
    _risk_manager.activate_kill_switch()

    prices = _get_current_prices()
    pnl_records = _executor.close_all_positions(prices, "kill_switch")

    total_pnl = sum(p.net_pnl for p in pnl_records)

    return KillSwitchResponse(
        status="activated",
        message="Kill switch activated. All positions closed. Trading halted.",
        positions_closed=len(pnl_records),
        total_pnl=total_pnl,
    )


@router.get("/liquidity")
async def get_liquidity_status():
    """Get liquidity status for all symbols."""
    if not _risk_manager:
        raise HTTPException(status_code=503, detail="System not initialized")

    return _risk_manager.get_liquidity_status()


@router.get("/trades/recent")
async def get_recent_trades(symbol: str = Query(...), limit: int = Query(100, le=1000)):
    """Get recent trades for a symbol."""
    if not _ingester:
        raise HTTPException(status_code=503, detail="System not initialized")

    trades = _ingester.get_recent_trades(symbol, limit)
    return [
        {
            "timestamp": t.timestamp.isoformat(),
            "symbol": t.symbol,
            "price": t.price,
            "quantity": t.quantity,
            "side": t.side.value,
        }
        for t in trades
    ]


# =======================
# Helpers
# =======================

def _get_current_prices() -> Dict[str, float]:
    """Get current prices for all symbols."""
    prices = {}
    if _ingester:
        config = get_config()
        for symbol in config.symbols:
            trades = _ingester.get_recent_trades(symbol, limit=1)
            if trades:
                prices[symbol] = trades[-1].price
    return prices

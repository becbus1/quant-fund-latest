"""
Paper trading executor.
Simulates order fills with configurable fees and slippage.
"""

import logging
import uuid
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from trader.app.common.config import get_config
from trader.app.common.supabase_client import insert_row, supabase
from shared.schemas import Side

# ✅ NEW: edge computation imports
from trader.app.edge_factory.confidence import compute_confidence

logger = logging.getLogger(__name__)


class PaperExecutor:
    """Paper trading execution engine (Supabase-only mode)."""

    def __init__(self):
        config = get_config()
        self.notional_usdt = config.notional_usdt
        self.fees_bps = config.fees_bps
        self.slippage_bps = config.slippage_bps
        self.time_stop_sec = config.time_stop_sec

        # In-memory positions only
        self._positions: Dict[str, dict] = {}
        logger.info("Running in Supabase-only mode (no SQLAlchemy, no DB positions)")

    def has_position(self, symbol: str) -> bool:
        return symbol in self._positions

    def get_position_snapshot(self, symbol: str) -> Optional[dict]:
        return self._positions.get(symbol)

    def get_all_position_snapshots(self) -> List[dict]:
        return list(self._positions.values())

    # ============================================================
    # 🔹 Volatility regime bucket (unchanged)
    # ============================================================
    def _get_volatility_bucket(self, symbol: str) -> str:
        response = (
            supabase
            .table("ml_training_events")
            .select("pnl_bps")
            .eq("symbol", symbol)
            .order("signal_timestamp", desc=True)
            .limit(50)
            .execute()
        )

        rows = response.data or []
        if len(rows) < 20:
            return "unknown"

        vol = math.sqrt(sum((r["pnl_bps"] or 0) ** 2 for r in rows) / len(rows))

        if vol < 5:
            return "low"
        elif vol < 15:
            return "normal"
        else:
            return "high"

    # ============================================================
    # 🔹 NEW: edge_registry lookup (SURVIVORS ONLY)
    # ============================================================
    def _get_active_edge(
        self,
        symbol: str,
        strategy: str,
        side: str,
        z_bucket: float,
        vol_bucket: str,
    ) -> Optional[dict]:
        response = (
            supabase
            .table("edge_registry")
            .select("*")
            .eq("symbol", symbol)
            .eq("strategy", strategy)
            .eq("side", side)
            .eq("z_score_bucket", z_bucket)
            .eq("volatility_bucket", vol_bucket)
            .eq("status", "active")
            .limit(1)
            .execute()
        )

        rows = response.data or []
        return rows[0] if rows else None

    def execute_entry(
        self,
        symbol: str,
        side: Side,
        price: float,
        strategy_name: str,
        take_profit_bps: float,
        stop_loss_bps: float,
        z_score: float,
    ) -> Optional[dict]:
        if self.has_position(symbol):
            logger.warning(f"Already have position in {symbol}, rejecting entry")
            return None

        # ========================================================
        # 🔹 Bucketization (must match ml_training_data view)
        # ========================================================
        z_bucket = math.floor(abs(z_score) * 2) / 2
        vol_bucket = self._get_volatility_bucket(symbol)

        edge = self._get_active_edge(
            symbol=symbol,
            strategy=strategy_name,
            side=side.value.upper(),
            z_bucket=z_bucket,
            vol_bucket=vol_bucket,
        )

        if edge is None:
            logger.info(
                f"Exploration trade: no edge for "
                f"{symbol} z={z_bucket:.2f} vol={vol_bucket} side={side.value}"
            )
        else:
            logger.info(
                f"Edge confirmed: "
                f"{symbol} z={z_bucket:.2f} vol={vol_bucket} side={side.value} "
                f"(lb95={edge['lb95']:.2%}, n={edge['n']})"
            )

        confidence = compute_confidence(z_score)

        signal_id = uuid.uuid4()

        insert_row(
            "signal_logs",
            {
                "signal_id": str(signal_id),
                "symbol": symbol,
                "strategy": strategy_name,
                "features": {
                    "side": side.value.upper(),
                    "z_bucket": z_bucket,
                    "volatility_bucket": vol_bucket,
                },
                "z_score": z_score,
                "entry_price": price,
                "timestamp": datetime.utcnow().isoformat(),
                "confidence": confidence,
                "created_at": datetime.utcnow().isoformat(),
            },
        )

        slippage_mult = 1 + (self.slippage_bps / 10000)
        fill_price = price * slippage_mult if side == Side.BUY else price / slippage_mult

        effective_notional = self.notional_usdt * confidence
        quantity = effective_notional / fill_price

        notional = fill_price * quantity
        fee = notional * (self.fees_bps / 10000)

        now = datetime.utcnow()

        if side == Side.BUY:
            take_profit_price = fill_price * (1 + take_profit_bps / 10000)
            stop_loss_price = fill_price * (1 - stop_loss_bps / 10000)
        else:
            take_profit_price = fill_price * (1 - take_profit_bps / 10000)
            stop_loss_price = fill_price * (1 + stop_loss_bps / 10000)

        time_stop_at = now + timedelta(seconds=self.time_stop_sec)

        position = {
            "signal_id": str(signal_id),
            "features": {
                "side": side.value.upper(),
                "z_bucket": z_bucket,
                "volatility_bucket": vol_bucket,
            },
            "z_score": z_score,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "entry_price": fill_price,
            "notional": notional,
            "entry_time": now,
            "strategy_name": strategy_name,
            "take_profit_price": take_profit_price,
            "stop_loss_price": stop_loss_price,
            "time_stop_at": time_stop_at,
            "fees_paid": fee,
        }

        self._positions[symbol] = position

        logger.info(
            f"Entry fill: {side.value} {quantity:.6f} {symbol} @ {fill_price:.4f} "
            f"(confidence={confidence:.2f})"
        )

        return position

    # ============================================================
    # 🔹 RESTORED: exit logic (unchanged)
    # ============================================================
    def check_exits(self, symbol: str, current_price: float) -> Optional[dict]:
        position = self._positions.get(symbol)
        if not position:
            return None

        now = datetime.utcnow()

        if position["time_stop_at"] and now >= position["time_stop_at"]:
            return self.execute_exit(symbol, current_price, "time_stop")

        if position["side"] == Side.BUY:
            if current_price >= position["take_profit_price"]:
                return self.execute_exit(symbol, current_price, "take_profit")
            if current_price <= position["stop_loss_price"]:
                return self.execute_exit(symbol, current_price, "stop_loss")
        else:
            if current_price <= position["take_profit_price"]:
                return self.execute_exit(symbol, current_price, "take_profit")
            if current_price >= position["stop_loss_price"]:
                return self.execute_exit(symbol, current_price, "stop_loss")

        return None

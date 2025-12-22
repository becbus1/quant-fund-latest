"""
Paper trading executor.
Simulates order fills with configurable fees and slippage.
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from trader.app.common.config import get_config
from trader.app.common.supabase_client import insert_row
from shared.schemas import Side

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

    def execute_entry(
        self,
        symbol: str,
        side: Side,
        price: float,
        strategy_name: str,
        take_profit_bps: float,
        stop_loss_bps: float,
    ) -> Optional[dict]:
        if self.has_position(symbol):
            logger.warning(f"Already have position in {symbol}, rejecting entry")
            return None

        # Log signal to Supabase
        signal_id = uuid.uuid4()
        insert_row(
            "signal_logs",
            {
                "signal_id": str(signal_id),
                "symbol": symbol,
                "strategy": strategy_name,
                "side": side.value,
                "confidence": 1.0,
                "created_at": datetime.utcnow().isoformat(),
            },
        )

        slippage_mult = 1 + (self.slippage_bps / 10000)
        fill_price = price * slippage_mult if side == Side.BUY else price / slippage_mult

        quantity = self.notional_usdt / fill_price
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
            f"Entry fill: {side.value} {quantity:.6f} {symbol} @ {fill_price:.4f}"
        )

        return position

    def execute_exit(
        self,
        symbol: str,
        price: float,
        exit_reason: str,
    ) -> Optional[dict]:
        position = self._positions.get(symbol)
        if not position:
            logger.warning(f"No position to exit for {symbol}")
            return None

        side = position["side"]
        quantity = position["quantity"]
        entry_price = position["entry_price"]

        slippage_mult = 1 + (self.slippage_bps / 10000)
        exit_price = price * slippage_mult if side == Side.SELL else price / slippage_mult

        notional = exit_price * quantity
        fee = notional * (self.fees_bps / 10000)

        if side == Side.BUY:
            gross_pnl = (exit_price - entry_price) * quantity
        else:
            gross_pnl = (entry_price - exit_price) * quantity

        total_fees = position["fees_paid"] + fee
        net_pnl = gross_pnl - total_fees
        pnl_bps = (net_pnl / position["notional"]) * 10000

        now = datetime.utcnow()
        hold_time_sec = (now - position["entry_time"]).total_seconds()

        # Write PnL to Supabase
        insert_row(
            "pnl",
            {
                "timestamp": now.isoformat(),
                "symbol": symbol,
                "strategy_name": position["strategy_name"],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "quantity": quantity,
                "side": side.value,
                "gross_pnl": gross_pnl,
                "fees": total_fees,
                "net_pnl": net_pnl,
                "pnl_bps": pnl_bps,
                "hold_time_sec": hold_time_sec,
                "exit_reason": exit_reason,
            },
        )

        del self._positions[symbol]

        logger.info(
            f"Exit fill: {symbol} @ {exit_price:.4f} "
            f"(reason={exit_reason}, pnl_bps={pnl_bps:.2f})"
        )

        return {
            "symbol": symbol,
            "net_pnl": net_pnl,
            "pnl_bps": pnl_bps,
            "exit_reason": exit_reason,
        }

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

"""
Paper trading executor.
Simulates order fills with configurable fees and slippage.
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from trader.app.common.config import get_config
from trader.app.common.db import get_db_session
from trader.app.common.models import Order, Fill, PnL, Position
from shared.schemas import Side, OrderStatus, PositionStatus

logger = logging.getLogger(__name__)


class PaperExecutor:
    """Paper trading execution engine."""

    def __init__(self):
        config = get_config()
        self.notional_usdt = config.notional_usdt
        self.fees_bps = config.fees_bps
        self.slippage_bps = config.slippage_bps
        self.time_stop_sec = config.time_stop_sec

        # Store ONLY position IDs (never ORM objects)
        self._positions: Dict[str, int] = {}
        self._load_positions()

    def _load_positions(self) -> None:
        """Load open positions from database."""
        with get_db_session() as db:
            open_positions = (
                db.query(Position)
                .filter(Position.status == PositionStatus.OPEN)
                .all()
            )
            for pos in open_positions:
                self._positions[pos.symbol] = pos.id

            logger.info(f"Loaded {len(self._positions)} open positions")

    def has_position(self, symbol: str) -> bool:
        """Check if there's an open position for symbol."""
        return symbol in self._positions

    # ⚠️ SAFE but should NOT be used by strategy code
    def get_position(self, symbol: str) -> Optional[Position]:
        position_id = self._positions.get(symbol)
        if not position_id:
            return None

        with get_db_session() as db:
            return db.query(Position).get(position_id)

    # ✅ THIS IS WHAT STRATEGIES MUST USE
    def get_position_snapshot(self, symbol: str) -> Optional[dict]:
        """
        Return a SAFE, session-free snapshot of a position.
        This is the ONLY thing strategies should ever use.
        """
        position_id = self._positions.get(symbol)
        if not position_id:
            return None

        with get_db_session() as db:
            position = db.query(Position).get(position_id)
            if not position:
                return None

            return {
                "symbol": position.symbol,
                "side": position.side,
                "quantity": position.quantity,
                "entry_price": position.entry_price,
                "take_profit_price": position.take_profit_price,
                "stop_loss_price": position.stop_loss_price,
                "entry_time": position.entry_time,
                "strategy_name": position.strategy_name,
            }

    # ✅✅✅ THIS WAS THE ONLY MISSING METHOD
    def get_all_position_snapshots(self) -> List[dict]:
        """
        Return SAFE snapshots of all open positions.
        Never returns ORM objects.
        """
        snapshots = []

        with get_db_session() as db:
            for symbol, position_id in self._positions.items():
                position = db.query(Position).get(position_id)
                if not position:
                    continue

                snapshots.append(
                    {
                        "symbol": position.symbol,
                        "side": position.side,
                        "quantity": position.quantity,
                        "entry_price": position.entry_price,
                        "take_profit_price": position.take_profit_price,
                        "stop_loss_price": position.stop_loss_price,
                        "entry_time": position.entry_time,
                        "strategy_name": position.strategy_name,
                    }
                )

        return snapshots

    def execute_entry(
        self,
        symbol: str,
        side: Side,
        price: float,
        strategy_name: str,
        take_profit_bps: float,
        stop_loss_bps: float,
    ) -> Optional[Fill]:
        """
        Execute entry order with simulated fill.
        Returns fill if successful, None if rejected.
        """
        if self.has_position(symbol):
            logger.warning(f"Already have position in {symbol}, rejecting entry")
            return None

        slippage_mult = 1 + (self.slippage_bps / 10000)
        fill_price = price * slippage_mult if side == Side.BUY else price / slippage_mult

        quantity = self.notional_usdt / fill_price
        notional = fill_price * quantity
        fee = notional * (self.fees_bps / 10000)

        order_id = f"ORD-{uuid.uuid4().hex[:12]}"
        fill_id = f"FILL-{uuid.uuid4().hex[:12]}"
        now = datetime.utcnow()

        if side == Side.BUY:
            take_profit_price = fill_price * (1 + take_profit_bps / 10000)
            stop_loss_price = fill_price * (1 - stop_loss_bps / 10000)
        else:
            take_profit_price = fill_price * (1 - take_profit_bps / 10000)
            stop_loss_price = fill_price * (1 + stop_loss_bps / 10000)

        time_stop_at = now + timedelta(seconds=self.time_stop_sec)

        with get_db_session() as db:
            order = Order(
                order_id=order_id,
                timestamp=now,
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
                notional=notional,
                status=OrderStatus.FILLED,
                strategy_name=strategy_name,
            )
            db.add(order)

            fill = Fill(
                fill_id=fill_id,
                order_id=order_id,
                timestamp=now,
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=fill_price,
                notional=notional,
                fee=fee,
                slippage=abs(fill_price - price) * quantity,
            )
            db.add(fill)

            position = Position(
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=fill_price,
                notional=notional,
                entry_time=now,
                strategy_name=strategy_name,
                status=PositionStatus.OPEN,
                take_profit_price=take_profit_price,
                stop_loss_price=stop_loss_price,
                time_stop_at=time_stop_at,
            )
            db.add(position)
            db.flush()

            # ✅ store ID only
            self._positions[symbol] = position.id

            logger.info(
                f"Entry fill: {side.value} {quantity:.6f} {symbol} @ {fill_price:.4f}"
            )

            return fill

    def execute_exit(
        self,
        symbol: str,
        price: float,
        exit_reason: str,
    ) -> Optional[PnL]:
        position_id = self._positions.get(symbol)
        if not position_id:
            logger.warning(f"No position to exit for {symbol}")
            return None

        with get_db_session() as db:
            position = db.query(Position).get(position_id)
            if not position:
                return None

            exit_side = Side.SELL if position.side == Side.BUY else Side.BUY

            slippage_mult = 1 + (self.slippage_bps / 10000)
            fill_price = price * slippage_mult if exit_side == Side.BUY else price / slippage_mult

            notional = fill_price * position.quantity
            fee = notional * (self.fees_bps / 10000)

            if position.side == Side.BUY:
                gross_pnl = (fill_price - position.entry_price) * position.quantity
            else:
                gross_pnl = (position.entry_price - fill_price) * position.quantity

            total_fees = (
                position.notional * (self.fees_bps / 10000)
                + fee
            )
            net_pnl = gross_pnl - total_fees
            pnl_bps = (net_pnl / position.notional) * 10000

            now = datetime.utcnow()
            hold_time_sec = (now - position.entry_time).total_seconds()

            order = Order(
                order_id=f"ORD-{uuid.uuid4().hex[:12]}",
                timestamp=now,
                symbol=symbol,
                side=exit_side,
                quantity=position.quantity,
                price=price,
                notional=notional,
                status=OrderStatus.FILLED,
                strategy_name=position.strategy_name,
            )
            db.add(order)

            pnl_record = PnL(
                timestamp=now,
                symbol=symbol,
                strategy_name=position.strategy_name,
                entry_price=position.entry_price,
                exit_price=fill_price,
                quantity=position.quantity,
                side=position.side,
                gross_pnl=gross_pnl,
                fees=total_fees,
                net_pnl=net_pnl,
                pnl_bps=pnl_bps,
                hold_time_sec=hold_time_sec,
                exit_reason=exit_reason,
            )
            db.add(pnl_record)

            position.status = PositionStatus.CLOSED
            position.updated_at = now

            del self._positions[symbol]

            logger.info(
                f"Exit fill: {exit_side.value} {position.quantity:.6f} {symbol} @ {fill_price:.4f} "
                f"(reason={exit_reason}, pnl_bps={pnl_bps:.2f})"
            )

            return pnl_record

    def check_exits(self, symbol: str, current_price: float) -> Optional[PnL]:
        position_id = self._positions.get(symbol)
        if not position_id:
            return None

        with get_db_session() as db:
            position = db.query(Position).get(position_id)
            if not position:
                return None

            now = datetime.utcnow()

            if position.time_stop_at and now >= position.time_stop_at:
                return self.execute_exit(symbol, current_price, "time_stop")

            if position.side == Side.BUY:
                if current_price >= position.take_profit_price:
                    return self.execute_exit(symbol, current_price, "take_profit")
                if current_price <= position.stop_loss_price:
                    return self.execute_exit(symbol, current_price, "stop_loss")
            else:
                if current_price <= position.take_profit_price:
                    return self.execute_exit(symbol, current_price, "take_profit")
                if current_price >= position.stop_loss_price:
                    return self.execute_exit(symbol, current_price, "stop_loss")

            return None

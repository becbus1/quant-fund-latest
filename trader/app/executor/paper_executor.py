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
        self._positions: Dict[str, Position] = {}
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
                self._positions[pos.symbol] = pos
            logger.info(f"Loaded {len(self._positions)} open positions")

    def has_position(self, symbol: str) -> bool:
        """Check if there's an open position for symbol."""
        return symbol in self._positions

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get open position for symbol."""
        return self._positions.get(symbol)

    def get_all_positions(self) -> List[Position]:
        """Get all open positions."""
        return list(self._positions.values())

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
        # Check for existing position
        if self.has_position(symbol):
            logger.warning(f"Already have position in {symbol}, rejecting entry")
            return None

        # Calculate fill price with slippage
        slippage_mult = 1 + (self.slippage_bps / 10000)
        if side == Side.BUY:
            fill_price = price * slippage_mult
        else:
            fill_price = price / slippage_mult

        # Calculate quantity
        quantity = self.notional_usdt / fill_price
        notional = fill_price * quantity

        # Calculate fee
        fee = notional * (self.fees_bps / 10000)

        # Generate IDs
        order_id = f"ORD-{uuid.uuid4().hex[:12]}"
        fill_id = f"FILL-{uuid.uuid4().hex[:12]}"
        now = datetime.utcnow()

        # Calculate exit levels
        if side == Side.BUY:
            take_profit_price = fill_price * (1 + take_profit_bps / 10000)
            stop_loss_price = fill_price * (1 - stop_loss_bps / 10000)
        else:
            take_profit_price = fill_price * (1 - take_profit_bps / 10000)
            stop_loss_price = fill_price * (1 + stop_loss_bps / 10000)

        time_stop_at = now + timedelta(seconds=self.time_stop_sec)

        with get_db_session() as db:
            # Create order
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

            # Create fill
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

            # Create position
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

            self._positions[symbol] = position
            logger.info(
                f"Entry fill: {side.value} {quantity:.6f} {symbol} @ {fill_price:.4f} "
                f"(fee: {fee:.4f}, slippage: {fill.slippage:.4f})"
            )

            return fill

    def execute_exit(
        self,
        symbol: str,
        price: float,
        exit_reason: str,
    ) -> Optional[PnL]:
        """
        Execute exit order and calculate PnL.
        Returns PnL record if successful, None if no position.
        """
        position = self._positions.get(symbol)
        if not position:
            logger.warning(f"No position to exit for {symbol}")
            return None

        # Determine exit side (opposite of entry)
        exit_side = Side.SELL if position.side == Side.BUY else Side.BUY

        # Calculate fill price with slippage
        slippage_mult = 1 + (self.slippage_bps / 10000)
        if exit_side == Side.BUY:
            fill_price = price * slippage_mult
        else:
            fill_price = price / slippage_mult

        notional = fill_price * position.quantity
        fee = notional * (self.fees_bps / 10000)

        # Calculate PnL
        if position.side == Side.BUY:
            gross_pnl = (fill_price - position.entry_price) * position.quantity
        else:
            gross_pnl = (position.entry_price - fill_price) * position.quantity

        # Total fees (entry + exit)
        entry_fee = position.notional * (self.fees_bps / 10000)
        total_fees = entry_fee + fee
        net_pnl = gross_pnl - total_fees

        # Calculate PnL in basis points
        pnl_bps = (net_pnl / position.notional) * 10000

        # Calculate hold time
        now = datetime.utcnow()
        hold_time_sec = (now - position.entry_time).total_seconds()

        # Generate IDs
        order_id = f"ORD-{uuid.uuid4().hex[:12]}"
        fill_id = f"FILL-{uuid.uuid4().hex[:12]}"

        with get_db_session() as db:
            # Create exit order
            order = Order(
                order_id=order_id,
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

            # Create exit fill
            fill = Fill(
                fill_id=fill_id,
                order_id=order_id,
                timestamp=now,
                symbol=symbol,
                side=exit_side,
                quantity=position.quantity,
                price=fill_price,
                notional=notional,
                fee=fee,
                slippage=abs(fill_price - price) * position.quantity,
            )
            db.add(fill)

            # Create PnL record
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

            # Close position
            db_position = (
                db.query(Position)
                .filter(
                    Position.symbol == symbol,
                    Position.status == PositionStatus.OPEN,
                )
                .first()
            )
            if db_position:
                db_position.status = PositionStatus.CLOSED
                db_position.updated_at = now

            # Remove from memory
            del self._positions[symbol]

            logger.info(
                f"Exit fill: {exit_side.value} {position.quantity:.6f} {symbol} @ {fill_price:.4f} "
                f"(reason: {exit_reason}, net_pnl: {net_pnl:.4f}, pnl_bps: {pnl_bps:.2f})"
            )

            return pnl_record

    def check_exits(self, symbol: str, current_price: float) -> Optional[PnL]:
        """
        Check if position should be exited (TP, SL, or time stop).
        Returns PnL if exit triggered, None otherwise.
        """
        position = self._positions.get(symbol)
        if not position:
            return None

        now = datetime.utcnow()

        # Check time stop
        if position.time_stop_at and now >= position.time_stop_at:
            return self.execute_exit(symbol, current_price, "time_stop")

        # Check take profit
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

    def close_all_positions(self, prices: Dict[str, float], reason: str) -> List[PnL]:
        """Close all open positions (kill switch)."""
        pnl_records = []
        for symbol in list(self._positions.keys()):
            price = prices.get(symbol)
            if price:
                pnl = self.execute_exit(symbol, price, reason)
                if pnl:
                    pnl_records.append(pnl)
        return pnl_records

    def get_daily_pnl(self) -> float:
        """Get total realized PnL for today."""
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        with get_db_session() as db:
            pnl_records = (
                db.query(PnL)
                .filter(PnL.timestamp >= today_start)
                .all()
            )
            return sum(p.net_pnl for p in pnl_records)

    def get_total_pnl(self) -> float:
        """Get total realized PnL all time."""
        with get_db_session() as db:
            pnl_records = db.query(PnL).all()
            return sum(p.net_pnl for p in pnl_records)

    def get_unrealized_pnl(self, prices: Dict[str, float]) -> float:
        """Get total unrealized PnL from open positions."""
        total = 0.0
        for symbol, position in self._positions.items():
            current_price = prices.get(symbol)
            if current_price:
                if position.side == Side.BUY:
                    pnl = (current_price - position.entry_price) * position.quantity
                else:
                    pnl = (position.entry_price - current_price) * position.quantity
                total += pnl
        return total

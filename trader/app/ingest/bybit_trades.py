"""
Bybit public WebSocket trade data ingestion.
Subscribes to trades for configurable symbols.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Callable, Dict, List, Optional, Set

import websockets
from websockets.exceptions import ConnectionClosed

from trader.app.common.config import get_config
from trader.app.common.db import get_db_session
from trader.app.common.models import Trade
from shared.schemas import Side, TradeData

logger = logging.getLogger(__name__)

BYBIT_WS_URL = "wss://stream.bybit.com/v5/public/linear"


class BybitTradeIngester:
    """Ingests trade data from Bybit public WebSocket."""

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        on_trade: Optional[Callable[[TradeData], None]] = None,
    ):
        config = get_config()
        self.symbols = symbols or config.symbols
        self.on_trade = on_trade
        self.ws_reconnect_delay = config.ws_reconnect_delay

        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        self._enabled_symbols: Set[str] = set(self.symbols)

        self._trade_buffer: List[TradeData] = []
        self._buffer_size = 100
        self._last_flush = datetime.utcnow()
        self._flush_interval_sec = 5

    async def start(self) -> None:
        """Start the WebSocket connection and ingestion."""
        self._running = True
        while self._running:
            try:
                await self._connect_and_subscribe()
            except ConnectionClosed as e:
                logger.warning(f"WebSocket connection closed: {e}")
                if self._running:
                    logger.info(
                        f"Reconnecting in {self.ws_reconnect_delay} seconds..."
                    )
                    await asyncio.sleep(self.ws_reconnect_delay)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                if self._running:
                    await asyncio.sleep(self.ws_reconnect_delay)

    async def stop(self) -> None:
        """Stop the WebSocket connection."""
        self._running = False
        if self._ws:
            await self._ws.close()
            self._ws = None
        await self._flush_buffer()

    def enable_symbol(self, symbol: str) -> None:
        """Enable trading for a symbol."""
        self._enabled_symbols.add(symbol)
        logger.info(f"Enabled symbol: {symbol}")

    def disable_symbol(self, symbol: str) -> None:
        """Disable trading for a symbol (liquidity gate)."""
        self._enabled_symbols.discard(symbol)
        logger.warning(f"Disabled symbol due to liquidity: {symbol}")

    def is_symbol_enabled(self, symbol: str) -> bool:
        """Check if symbol is enabled for trading."""
        return symbol in self._enabled_symbols

    async def _connect_and_subscribe(self) -> None:
        """Connect to WebSocket and subscribe to trade topics."""
        logger.info(f"Connecting to Bybit WebSocket: {BYBIT_WS_URL}")

        async with websockets.connect(BYBIT_WS_URL, ping_interval=20) as ws:
            self._ws = ws
            logger.info("Connected to Bybit WebSocket")

            topics = [f"publicTrade.{symbol}" for symbol in self.symbols]
            subscribe_msg = {"op": "subscribe", "args": topics}
            await ws.send(json.dumps(subscribe_msg))
            logger.info(f"Subscribed to topics: {topics}")

            asyncio.create_task(self._heartbeat())

            async for message in ws:
                if not self._running:
                    break
                await self._handle_message(message)

    async def _heartbeat(self) -> None:
        """Send periodic ping to keep connection alive."""
        while self._running and self._ws:
            try:
                await asyncio.sleep(20)
                if self._ws:
                    await self._ws.send(json.dumps({"op": "ping"}))
            except Exception as e:
                logger.debug(f"Heartbeat error: {e}")
                break

    async def _handle_message(self, message: str) -> None:
        """Handle incoming WebSocket message."""
        try:
            data = json.loads(message)

            if data.get("op") == "pong":
                return

            if data.get("op") == "subscribe":
                if data.get("success"):
                    logger.info("Subscription confirmed")
                else:
                    logger.error(f"Subscription failed: {data}")
                return

            topic = data.get("topic", "")
            if topic.startswith("publicTrade."):
                await self._process_trades(data)

            now = datetime.utcnow()
            if (now - self._last_flush).total_seconds() >= self._flush_interval_sec:
                await self._flush_buffer()

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse message: {e}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")

    async def _process_trades(self, data: Dict) -> None:
        """Process incoming trade data."""
        topic = data.get("topic", "")
        symbol = topic.replace("publicTrade.", "")
        trades_data = data.get("data", [])

        for trade in trades_data:
            try:
                timestamp = datetime.fromtimestamp(int(trade["T"]) / 1000)
                side = Side.BUY if trade["S"] == "Buy" else Side.SELL

                trade_obj = TradeData(
                    timestamp=timestamp,
                    symbol=symbol,
                    price=float(trade["p"]),
                    quantity=float(trade["v"]),
                    side=side,
                    trade_id=str(trade["i"]),
                )

                self._trade_buffer.append(trade_obj)

                if self.on_trade:
                    try:
                        self.on_trade(trade_obj)
                    except Exception as e:
                        logger.error(f"Strategy error: {e}")

                if len(self._trade_buffer) >= self._buffer_size:
                    await self._flush_buffer()

            except (KeyError, ValueError) as e:
                logger.error(f"Error parsing trade: {e}")

    async def _flush_buffer(self) -> None:
        """Flush trade buffer (DISABLED — Supabase-only mode)."""
        if not self._trade_buffer:
            return

        flushed_count = len(self._trade_buffer)
        self._trade_buffer.clear()
        self._last_flush = datetime.utcnow()

        logger.info(
            f"Skipping trade DB flush (Supabase-only mode). "
            f"Dropped {flushed_count} buffered trades."
        )

        return

    def get_recent_trades(
        self, symbol: str, limit: int = 1000
    ) -> List[TradeData]:
        """Get recent trades from database."""
        return []

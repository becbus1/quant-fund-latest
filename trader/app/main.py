"""
Main FastAPI application for paper trading system.
Runs continuously, ingests trades, executes paper trades, enforces risk.
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI

from trader.app.common.config import get_config
from trader.app.common.db import init_db, get_db_session
from trader.app.common.models import SignalLog
from trader.app.ingest.bybit_trades import BybitTradeIngester
from trader.app.executor.paper_executor import PaperExecutor
from trader.app.risk.limits import RiskManager
from trader.app.api.routes import router, set_components
from trader.app.dashboard.routes import router as dashboard_router  # ✅ NEW
from shared.schemas import StrategyEntry, TradeData, Side

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global components
ingester: BybitTradeIngester = None
executor: PaperExecutor = None
risk_manager: RiskManager = None
strategies: List[StrategyEntry] = []

# ✅ OPTION B: edge-trigger / cooldown guard
last_entry_ts: Dict[str, float] = {}


def load_strategies() -> List[StrategyEntry]:
    """Load enabled strategies from registry."""
    config = get_config()
    registry_path = Path(config.registry_path)

    if not registry_path.exists():
        logger.warning(f"Strategy registry not found at {registry_path}")
        return []

    try:
        with open(registry_path) as f:
            data = json.load(f)

        loaded = []
        for entry in data.get("strategies", []):
            strategy = StrategyEntry.from_dict(entry)
            if strategy.enabled:
                loaded.append(strategy)
                logger.info(f"Loaded strategy: {strategy.name}")

        logger.info(f"Loaded {len(loaded)} enabled strategies")
        return loaded

    except Exception as e:
        logger.error(f"Failed to load strategies: {e}")
        return []


class FeatureCalculator:
    """Calculate trading features from recent trades."""

    def __init__(self, window_size: int = 120):
        self.window_size = window_size
        self._trade_buffers: Dict[str, List[TradeData]] = {}

    def add_trade(self, trade: TradeData) -> None:
        """Add trade to buffer."""
        if trade.symbol not in self._trade_buffers:
            self._trade_buffers[trade.symbol] = []

        buffer = self._trade_buffers[trade.symbol]
        buffer.append(trade)

        cutoff = trade.timestamp.timestamp() - self.window_size
        while buffer and buffer[0].timestamp.timestamp() < cutoff:
            buffer.pop(0)

    def get_features(self, symbol: str) -> Dict[str, float]:
        buffer = self._trade_buffers.get(symbol, [])

        if len(buffer) < 10:
            return {}

        prices = [t.price for t in buffer]
        volumes = [t.quantity * t.price for t in buffer]

        buy_volume = sum(t.quantity * t.price for t in buffer if t.side == Side.BUY)
        sell_volume = sum(t.quantity * t.price for t in buffer if t.side == Side.SELL)
        total_volume = buy_volume + sell_volume

        if total_volume == 0:
            return {}

        ofi = (buy_volume - sell_volume) / total_volume

        avg_volume = sum(volumes) / len(volumes)
        recent_volume = sum(volumes[-10:]) / 10 if len(volumes) >= 10 else avg_volume
        volume_spike = recent_volume / avg_volume if avg_volume > 0 else 1.0

        time_span = (buffer[-1].timestamp - buffer[0].timestamp).total_seconds()
        intensity = len(buffer) / time_span if time_span > 0 else 0

        if len(prices) > 1:
            import statistics

            returns = [
                (prices[i] - prices[i - 1]) / prices[i - 1]
                for i in range(1, len(prices))
            ]
            volatility = statistics.stdev(returns) if len(returns) > 1 else 0
        else:
            volatility = 0

        price_range = (max(prices) - min(prices)) / min(prices) if min(prices) > 0 else 0

        total_qty = sum(t.quantity for t in buffer)
        vwap = sum(t.price * t.quantity for t in buffer) / total_qty if total_qty > 0 else prices[-1]
        vwap_deviation = (prices[-1] - vwap) / vwap if vwap > 0 else 0

        return {
            "order_flow_imbalance": ofi,
            "volume_spike_ratio": volume_spike,
            "trade_intensity": intensity,
            "realized_volatility": volatility,
            "price_range_pct": price_range * 100,
            "vwap_deviation": vwap_deviation,
        }

    def get_z_score(self, symbol: str, feature: str = "order_flow_imbalance") -> float:
        features = self.get_features(symbol)
        value = features.get(feature, 0)
        return value * 3


feature_calculator = FeatureCalculator()


def on_trade_received(trade: TradeData) -> None:
    """Callback for each received trade."""
    global executor, risk_manager, strategies, feature_calculator, last_entry_ts

    if not executor or not risk_manager:
        return

    if risk_manager.is_kill_switch_active():
        return

    feature_calculator.add_trade(trade)

    # Check exits for existing positions
    if executor.has_position(trade.symbol):
        executor.check_exits(trade.symbol, trade.price)
        return

    # 🔒 HARD GUARD: never attempt entry if a position already exists
    if executor.has_position(trade.symbol):
        return

    # ✅ OPTION B: EDGE TRIGGER / COOLDOWN
    now = trade.timestamp.timestamp()
    if now - last_entry_ts.get(trade.symbol, 0) < 1.0:
        return
    last_entry_ts[trade.symbol] = now

    # Check for entry signals
    for strategy in strategies:
        if trade.symbol not in strategy.symbols and strategy.symbols:
            continue

        features = feature_calculator.get_features(trade.symbol)
        if not features:
            continue

        z_score = feature_calculator.get_z_score(trade.symbol)

        should_enter = False
        side = None

        if z_score >= strategy.parameters.entry_z_score:
            should_enter = True
            side = Side.SELL
        elif z_score <= -strategy.parameters.entry_z_score:
            should_enter = True
            side = Side.BUY

        if not should_enter:
            continue

        allowed, reason = risk_manager.can_open_position(
            trade.symbol,
            len(executor.get_all_position_snapshots()),
        )

        if not allowed:
            logger.debug(f"Entry rejected for {trade.symbol}: {reason}")
            continue

        fill = executor.execute_entry(
            symbol=trade.symbol,
            side=side,
            price=trade.price,
            strategy_name=strategy.name,
            take_profit_bps=strategy.parameters.take_profit_bps,
            stop_loss_bps=strategy.parameters.stop_loss_bps,
            z_score=z_score,   # ✅ ADD THIS
        )

        if fill:
            logger.info(
                f"Strategy {strategy.name} entered {side.value} {trade.symbol} "
                f"@ {trade.price:.4f} (z={z_score:.2f})"
            )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ingester, executor, risk_manager, strategies, feature_calculator

    logger.info("Starting paper trading system...")
    config = get_config()

    if config.mode != "paper":
        raise ValueError("Only paper trading mode is allowed")

    init_db()
    logger.info("Database initialized")

    executor = PaperExecutor()
    risk_manager = RiskManager()
    feature_calculator = FeatureCalculator(config.window_size)

    strategies = load_strategies()

    ingester = BybitTradeIngester(
        symbols=config.symbols,
        on_trade=on_trade_received,
    )

    set_components(executor, risk_manager, ingester)

    async def start_ingester():
        await asyncio.sleep(1)  # allow FastAPI to boot
        await ingester.start()

    ingester_task = asyncio.create_task(start_ingester())

    logger.info(
        f"Paper trading system started. "
        f"Symbols: {config.symbols}, Strategies: {len(strategies)}"
    )

    yield

    logger.info("Shutting down paper trading system...")
    await ingester.stop()
    ingester_task.cancel()
    try:
        await ingester_task
    except asyncio.CancelledError:
        pass
    logger.info("Paper trading system stopped")


app = FastAPI(
    title="Paper Trading System",
    description="Quantitative paper trading with ML edge strategies",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)
app.include_router(dashboard_router)  # ✅ NEW


@app.get("/")
async def root():
    return {
        "service": "paper-trading-system",
        "mode": "paper",
        "status": "running",
    }

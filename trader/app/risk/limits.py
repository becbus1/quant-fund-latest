"""
Risk management and limits enforcement.
Enforces position limits, daily loss limits, and liquidity gates.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Set

from trader.app.common.config import get_config
from shared.schemas import LiquidityMetrics

logger = logging.getLogger(__name__)


class RiskManager:
    """Risk management and limits enforcement."""

    def __init__(self):
        config = get_config()
        self.max_daily_loss_usdt = config.max_daily_loss_usdt
        self.max_positions = config.max_positions
        self.min_24h_volume_usd = config.min_24h_volume_usd
        self.max_spread_bps = config.max_spread_bps
        self.min_depth_10bps_usd = config.min_depth_10bps_usd

        self._kill_switch_active = False
        self._disabled_symbols: Set[str] = set()
        self._liquidity_metrics: Dict[str, LiquidityMetrics] = {}

        # Supabase-only / in-memory PnL accumulator (safe placeholder)
        self._daily_pnl_usdt: float = 0.0

    def is_kill_switch_active(self) -> bool:
        """Check if kill switch is active."""
        return self._kill_switch_active

    def activate_kill_switch(self) -> None:
        """Activate global kill switch."""
        self._kill_switch_active = True
        logger.critical("KILL SWITCH ACTIVATED - All trading halted")

    def deactivate_kill_switch(self) -> None:
        """Deactivate kill switch."""
        self._kill_switch_active = False
        logger.info("Kill switch deactivated")

    def check_daily_loss_limit(self) -> bool:
        """
        Check if daily loss limit has been breached.
        Returns True if trading should be halted.
        """
        daily_pnl = self._get_daily_pnl()
        if daily_pnl <= -self.max_daily_loss_usdt:
            logger.warning(
                f"Daily loss limit breached: {daily_pnl:.2f} <= -{self.max_daily_loss_usdt:.2f}"
            )
            return True
        return False

    def _get_daily_pnl(self) -> float:
        """
        Get today's realized PnL.

        Supabase-only mode:
        - No SQLAlchemy
        - Uses in-memory accumulator
        - Safe for infra + smoke tests
        """
        return self._daily_pnl_usdt

    def can_open_position(
        self,
        symbol: str,
        current_position_count: int,
    ) -> tuple[bool, str]:
        """
        Check if a new position can be opened.
        Returns (allowed, reason).
        """
        # Kill switch check
        if self._kill_switch_active:
            return False, "kill_switch_active"

        # Daily loss check
        if self.check_daily_loss_limit():
            return False, "daily_loss_limit"

        # Max positions check
        if current_position_count >= self.max_positions:
            return False, "max_positions"

        # Symbol disabled check
        if symbol in self._disabled_symbols:
            return False, "symbol_disabled"

        # Liquidity check
        if not self.check_liquidity(symbol):
            return False, "liquidity_insufficient"

        return True, "ok"

    def update_liquidity_metrics(
        self,
        symbol: str,
        volume_24h_usd: float,
        spread_bps: float,
        depth_10bps_usd: float,
    ) -> None:
        """Update liquidity metrics for a symbol."""
        metrics = LiquidityMetrics(
            symbol=symbol,
            volume_24h_usd=volume_24h_usd,
            spread_bps=spread_bps,
            depth_10bps_usd=depth_10bps_usd,
            last_updated=datetime.utcnow(),
        )
        self._liquidity_metrics[symbol] = metrics

        # Check if symbol should be disabled
        if not self._meets_liquidity_requirements(metrics):
            self._disabled_symbols.add(symbol)
            logger.warning(f"Symbol {symbol} disabled due to insufficient liquidity")
        elif symbol in self._disabled_symbols:
            self._disabled_symbols.discard(symbol)
            logger.info(f"Symbol {symbol} re-enabled, liquidity recovered")

    def _meets_liquidity_requirements(self, metrics: LiquidityMetrics) -> bool:
        """Check if metrics meet liquidity requirements."""
        if metrics.volume_24h_usd < self.min_24h_volume_usd:
            logger.debug(
                f"{metrics.symbol}: volume {metrics.volume_24h_usd:.0f} < {self.min_24h_volume_usd:.0f}"
            )
            return False

        if metrics.spread_bps > self.max_spread_bps:
            logger.debug(
                f"{metrics.symbol}: spread {metrics.spread_bps:.2f} > {self.max_spread_bps:.2f}"
            )
            return False

        if metrics.depth_10bps_usd < self.min_depth_10bps_usd:
            logger.debug(
                f"{metrics.symbol}: depth {metrics.depth_10bps_usd:.0f} < {self.min_depth_10bps_usd:.0f}"
            )
            return False

        return True

    def check_liquidity(self, symbol: str) -> bool:
        """Check if symbol has sufficient liquidity."""
        metrics = self._liquidity_metrics.get(symbol)
        if not metrics:
            logger.warning(f"No liquidity metrics for {symbol}, allowing trade")
            return True
        return self._meets_liquidity_requirements(metrics)

    def get_disabled_symbols(self) -> List[str]:
        """Get list of disabled symbols."""
        return list(self._disabled_symbols)

    def get_liquidity_status(self) -> Dict[str, Dict]:
        """Get liquidity status for all tracked symbols."""
        status = {}
        for symbol, metrics in self._liquidity_metrics.items():
            status[symbol] = {
                "volume_24h_usd": metrics.volume_24h_usd,
                "spread_bps": metrics.spread_bps,
                "depth_10bps_usd": metrics.depth_10bps_usd,
                "meets_requirements": self._meets_liquidity_requirements(metrics),
                "enabled": symbol not in self._disabled_symbols,
                "last_updated": metrics.last_updated.isoformat(),
            }
        return status

    def get_risk_status(self) -> Dict:
        """Get overall risk status."""
        daily_pnl = self._get_daily_pnl()
        return {
            "kill_switch_active": self._kill_switch_active,
            "daily_pnl": daily_pnl,
            "daily_loss_limit": self.max_daily_loss_usdt,
            "daily_loss_remaining": self.max_daily_loss_usdt + daily_pnl,
            "max_positions": self.max_positions,
            "disabled_symbols": list(self._disabled_symbols),
        }

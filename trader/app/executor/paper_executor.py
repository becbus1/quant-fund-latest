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
    # 🔹 NEW: volatility regime bucket (simple + stable)
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
    # 🔹 NEW: Bayesian lower bound (Wilson score, 95%)
    # ============================================================
    def _bayesian_lower_bound(self, wins: int, total: int, z: float = 1.96) -> float:
        if total == 0:
            return 0.0

        phat = wins / total
        denom = 1 + z**2 / total
        centre = phat + z**2 / (2 * total)
        margin = z * math.sqrt(
            (phat * (1 - phat) + z**2 / (4 * total)) / total
        )
        return (centre - margin) / denom

    # ============================================================
    # 🔹 NEW: empirical stats lookup (with buckets)
    # ============================================================
    def _get_edge_stats(
        self,
        symbol: str,
        strategy: str,
        z_bucket: float,
        vol_bucket: str,
    ) -> Optional[dict]:
        response = (
            supabase
            .table("ml_training_data")
            .select("label")
            .eq("symbol", symbol)
            .eq("strategy", strategy)
            .eq("z_score_bucket", z_bucket)
            .eq("volatility_bucket", vol_bucket)
            .execute()
        )

        rows = response.data or []
        total = len(rows)

        if total < 20:
            return None  # exploration mode

        wins = sum(r["label"] for r in rows)
        lower_bound = self._bayesian_lower_bound(wins, total)

        return {
            "total": total,
            "wins": wins,
            "lower_bound": lower_bound,
        }

    # ============================================================
    # 🔹 NEW: exploration decay threshold
    # ============================================================
    def _min_required_edge(self, sample_size: int) -> float:
        """
        Exploration decay:
        - Early → permissive
        - Later → stricter
        """
        return 0.48 + min(0.06, math.log10(sample_size + 1) * 0.02)

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
        # 🔹 NEW: bucketization
        # ========================================================
        z_bucket = math.floor(abs(z_score) * 2) / 2
        vol_bucket = self._get_volatility_bucket(symbol)

        stats = self._get_edge_stats(
            symbol=symbol,
            strategy=strategy_name,
            z_bucket=z_bucket,
            vol_bucket=vol_bucket,
        )

        if stats is not None:
            min_edge = self._min_required_edge(stats["total"])
            if stats["lower_bound"] < min_edge:
                logger.info(
                    f"Rejected {symbol} z={z_bucket:.2f} vol={vol_bucket} "
                    f"(LB={stats['lower_bound']:.2%}, req={min_edge:.2%})"
                )
                return None

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

    # === execute_exit and check_exits remain unchanged ===

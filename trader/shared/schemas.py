"""
Shared schemas and data contracts between trader and edge_factory.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class Side(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class PositionStatus(Enum):
    OPEN = "open"
    CLOSED = "closed"


class MetaLabel(Enum):
    PROFITABLE = "profitable"
    NEUTRAL = "neutral"
    ADVERSE = "adverse"


@dataclass
class TradeData:
    """Raw trade data from exchange."""
    timestamp: datetime
    symbol: str
    price: float
    quantity: float
    side: Side
    trade_id: str


@dataclass
class AnomalySignal:
    """Output from anomaly detection."""
    timestamp: datetime
    symbol: str
    anomaly_score: float
    features: Dict[str, float] = field(default_factory=dict)


@dataclass
class StrategyParameters:
    """Parameters defining a trading strategy."""
    entry_z_score: float = 2.0
    take_profit_bps: float = 8.0
    stop_loss_bps: float = 5.0
    time_stop_seconds: int = 60
    min_anomaly_score: float = 0.7


@dataclass
class StrategyMetrics:
    """Expected/realized performance metrics."""
    sharpe_ratio: float
    win_rate: float
    avg_profit_bps: float
    avg_loss_bps: float
    max_drawdown_pct: float
    trade_count: int
    profit_factor: float


@dataclass
class StrategyEntry:
    """Registry entry for a validated strategy."""
    name: str
    parameters: StrategyParameters
    expected_metrics: StrategyMetrics
    enabled: bool = True
    symbols: List[str] = field(default_factory=list)
    created_at: str = ""
    expires_at: str = ""
    review_at: str = ""
    version: int = 1
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "parameters": {
                "entry_z_score": self.parameters.entry_z_score,
                "take_profit_bps": self.parameters.take_profit_bps,
                "stop_loss_bps": self.parameters.stop_loss_bps,
                "time_stop_seconds": self.parameters.time_stop_seconds,
                "min_anomaly_score": self.parameters.min_anomaly_score,
            },
            "expected_metrics": {
                "sharpe_ratio": self.expected_metrics.sharpe_ratio,
                "win_rate": self.expected_metrics.win_rate,
                "avg_profit_bps": self.expected_metrics.avg_profit_bps,
                "avg_loss_bps": self.expected_metrics.avg_loss_bps,
                "max_drawdown_pct": self.expected_metrics.max_drawdown_pct,
                "trade_count": self.expected_metrics.trade_count,
                "profit_factor": self.expected_metrics.profit_factor,
            },
            "enabled": self.enabled,
            "symbols": self.symbols,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "review_at": self.review_at,
            "version": self.version,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyEntry":
        params = data.get("parameters", {})
        metrics = data.get("expected_metrics", {})
        return cls(
            name=data["name"],
            parameters=StrategyParameters(
                entry_z_score=params.get("entry_z_score", 2.0),
                take_profit_bps=params.get("take_profit_bps", 8.0),
                stop_loss_bps=params.get("stop_loss_bps", 5.0),
                time_stop_seconds=params.get("time_stop_seconds", 60),
                min_anomaly_score=params.get("min_anomaly_score", 0.7),
            ),
            expected_metrics=StrategyMetrics(
                sharpe_ratio=metrics.get("sharpe_ratio", 0.0),
                win_rate=metrics.get("win_rate", 0.0),
                avg_profit_bps=metrics.get("avg_profit_bps", 0.0),
                avg_loss_bps=metrics.get("avg_loss_bps", 0.0),
                max_drawdown_pct=metrics.get("max_drawdown_pct", 0.0),
                trade_count=metrics.get("trade_count", 0),
                profit_factor=metrics.get("profit_factor", 0.0),
            ),
            enabled=data.get("enabled", True),
            symbols=data.get("symbols", []),
            created_at=data.get("created_at", ""),
            expires_at=data.get("expires_at", ""),
            review_at=data.get("review_at", ""),
            version=data.get("version", 1),
            notes=data.get("notes", ""),
        )


@dataclass
class LiquidityMetrics:
    """Liquidity metrics for symbol gating."""
    symbol: str
    volume_24h_usd: float
    spread_bps: float
    depth_10bps_usd: float
    last_updated: datetime


# Liquidity thresholds
MIN_24H_VOLUME_USD = 500_000_000  # 500M
MAX_SPREAD_BPS = 5
MIN_DEPTH_10BPS_USD = 1_000_000  # 1M

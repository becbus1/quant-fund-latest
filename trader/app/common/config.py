"""
Configuration management via environment variables.
All configuration with sensible defaults.
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    """Trading system configuration from environment variables."""

    # Mode
    mode: str = field(default_factory=lambda: os.getenv("MODE", "paper"))

    # Symbols to trade
    symbols: List[str] = field(
        default_factory=lambda: os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT").split(
            ","
        )
    )

    # Feature calculation
    window_size: int = field(
        default_factory=lambda: int(os.getenv("WINDOW_SIZE", "120"))
    )

    # Entry/Exit parameters
    entry_z: float = field(default_factory=lambda: float(os.getenv("ENTRY_Z", "2.0")))
    take_profit_bps: float = field(
        default_factory=lambda: float(os.getenv("TP_BPS", "8"))
    )
    stop_loss_bps: float = field(
        default_factory=lambda: float(os.getenv("SL_BPS", "5"))
    )
    time_stop_sec: int = field(
        default_factory=lambda: int(os.getenv("TIME_STOP_SEC", "60"))
    )

    # Position sizing
    notional_usdt: float = field(
        default_factory=lambda: float(os.getenv("NOTIONAL_USDT", "25"))
    )

    # Fees and slippage
    fees_bps: float = field(default_factory=lambda: float(os.getenv("FEES_BPS", "4")))
    slippage_bps: float = field(
        default_factory=lambda: float(os.getenv("SLIPPAGE_BPS", "2"))
    )

    # Database
    database_url: str = field(
        default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///./trader.db")
    )

    # Risk limits
    max_daily_loss_usdt: float = field(
        default_factory=lambda: float(os.getenv("MAX_DAILY_LOSS_USDT", "100"))
    )
    max_positions: int = field(
        default_factory=lambda: int(os.getenv("MAX_POSITIONS", "3"))
    )

    # Kill switch
    killswitch_token: str = field(
        default_factory=lambda: os.getenv("KILLSWITCH_TOKEN", "SECRET")
    )

    # Liquidity thresholds
    min_24h_volume_usd: float = field(
        default_factory=lambda: float(os.getenv("MIN_24H_VOLUME_USD", "500000000"))
    )
    max_spread_bps: float = field(
        default_factory=lambda: float(os.getenv("MAX_SPREAD_BPS", "5"))
    )
    min_depth_10bps_usd: float = field(
        default_factory=lambda: float(os.getenv("MIN_DEPTH_10BPS_USD", "1000000"))
    )

    # WebSocket
    ws_reconnect_delay: int = field(
        default_factory=lambda: int(os.getenv("WS_RECONNECT_DELAY", "5"))
    )

    # Registry path
    registry_path: str = field(
        default_factory=lambda: os.getenv("REGISTRY_PATH", "registry/strategies.json")
    )

    def validate(self) -> None:
        """Validate configuration."""
        if self.mode != "paper":
            raise ValueError("Only paper trading mode is allowed")
        if self.notional_usdt <= 0:
            raise ValueError("Notional must be positive")
        if self.fees_bps < 0:
            raise ValueError("Fees cannot be negative")
        if not self.symbols:
            raise ValueError("At least one symbol required")


# Global config instance
_config: Config | None = None


def get_config() -> Config:
    """Get or create global config instance."""
    global _config
    if _config is None:
        _config = Config()
        _config.validate()
    return _config


def reset_config() -> None:
    """Reset config for testing."""
    global _config
    _config = None

"""
Strategy evaluation and selection.
Manages strategy lifecycle: validation, promotion, degradation, removal.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from edge_factory.backtests.backtest_runner import BacktestResult
from shared.schemas import StrategyEntry, StrategyParameters, StrategyMetrics

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for strategy evaluation."""
    min_sharpe: float = 0.5
    min_win_rate: float = 0.45
    min_profit_factor: float = 1.2
    min_trades: int = 50
    max_drawdown_pct: float = 15.0
    strategy_lifetime_days: int = 7
    review_interval_days: int = 1


class StrategyEvaluator:
    """
    Evaluates and manages strategy lifecycle.

    Responsibilities:
    - Validate candidate strategies
    - Promote passing strategies to registry
    - Track expected vs realized performance
    - Flag and remove degraded strategies
    """

    def __init__(
        self,
        registry_path: str = "registry/strategies.json",
        config: Optional[EvaluationConfig] = None,
    ):
        self.registry_path = Path(registry_path)
        self.config = config or EvaluationConfig()
        self._strategies: Dict[str, StrategyEntry] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        """Load strategies from registry."""
        if not self.registry_path.exists():
            logger.info("Registry not found, starting fresh")
            return

        try:
            with open(self.registry_path) as f:
                data = json.load(f)

            for entry in data.get("strategies", []):
                strategy = StrategyEntry.from_dict(entry)
                self._strategies[strategy.name] = strategy

            logger.info(f"Loaded {len(self._strategies)} strategies from registry")

        except Exception as e:
            logger.error(f"Failed to load registry: {e}")

    def _save_registry(self) -> None:
        """Save strategies to registry."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "updated_at": datetime.utcnow().isoformat(),
            "strategies": [s.to_dict() for s in self._strategies.values()],
        }

        with open(self.registry_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(self._strategies)} strategies to registry")

    def evaluate_candidate(
        self,
        result: BacktestResult,
        symbols: List[str],
    ) -> Tuple[bool, str, Optional[StrategyEntry]]:
        """
        Evaluate a backtest result for promotion.

        Returns:
            (passed, reason, strategy_entry)
        """
        # Check if backtest was valid
        if not result.is_valid:
            return False, result.rejection_reason, None

        # Check Sharpe ratio
        if result.sharpe_ratio < self.config.min_sharpe:
            reason = f"Sharpe {result.sharpe_ratio:.2f} < {self.config.min_sharpe}"
            return False, reason, None

        # Check win rate
        if result.win_rate < self.config.min_win_rate:
            reason = f"Win rate {result.win_rate:.2%} < {self.config.min_win_rate:.2%}"
            return False, reason, None

        # Check profit factor
        if result.profit_factor < self.config.min_profit_factor:
            reason = f"Profit factor {result.profit_factor:.2f} < {self.config.min_profit_factor}"
            return False, reason, None

        # Check trade count
        if result.total_trades < self.config.min_trades:
            reason = f"Trades {result.total_trades} < {self.config.min_trades}"
            return False, reason, None

        # Check drawdown
        if result.max_drawdown_pct > self.config.max_drawdown_pct:
            reason = f"Drawdown {result.max_drawdown_pct:.2f}% > {self.config.max_drawdown_pct}%"
            return False, reason, None

        # Create strategy entry
        now = datetime.utcnow()
        entry = StrategyEntry(
            name=result.strategy_name,
            parameters=result.parameters,
            expected_metrics=result.to_metrics(),
            enabled=True,
            symbols=symbols,
            created_at=now.isoformat(),
            expires_at=(now + timedelta(days=self.config.strategy_lifetime_days)).isoformat(),
            review_at=(now + timedelta(days=self.config.review_interval_days)).isoformat(),
            version=1,
            notes=f"Auto-generated from backtest. Sharpe: {result.sharpe_ratio:.2f}",
        )

        return True, "Passed all criteria", entry

    def promote_strategy(self, entry: StrategyEntry) -> None:
        """Add or update strategy in registry."""
        existing = self._strategies.get(entry.name)

        if existing:
            # Update version
            entry.version = existing.version + 1
            logger.info(f"Updating strategy {entry.name} to version {entry.version}")
        else:
            logger.info(f"Adding new strategy: {entry.name}")

        self._strategies[entry.name] = entry
        self._save_registry()

    def disable_strategy(self, name: str, reason: str) -> None:
        """Disable a strategy."""
        if name not in self._strategies:
            logger.warning(f"Strategy {name} not found")
            return

        self._strategies[name].enabled = False
        self._strategies[name].notes += f" | Disabled: {reason}"
        self._save_registry()
        logger.warning(f"Disabled strategy {name}: {reason}")

    def remove_strategy(self, name: str) -> None:
        """Remove a strategy from registry."""
        if name in self._strategies:
            del self._strategies[name]
            self._save_registry()
            logger.info(f"Removed strategy: {name}")

    def check_expiration(self) -> List[str]:
        """Check for expired strategies and disable them."""
        now = datetime.utcnow()
        expired = []

        for name, strategy in self._strategies.items():
            if not strategy.enabled:
                continue

            if strategy.expires_at:
                expires_at = datetime.fromisoformat(strategy.expires_at)
                if now >= expires_at:
                    self.disable_strategy(name, "Expired")
                    expired.append(name)

        return expired

    def check_review_due(self) -> List[str]:
        """Check for strategies due for review."""
        now = datetime.utcnow()
        due = []

        for name, strategy in self._strategies.items():
            if not strategy.enabled:
                continue

            if strategy.review_at:
                review_at = datetime.fromisoformat(strategy.review_at)
                if now >= review_at:
                    due.append(name)

        return due

    def compare_performance(
        self,
        name: str,
        realized_metrics: StrategyMetrics,
    ) -> Tuple[bool, Dict]:
        """
        Compare expected vs realized performance.

        Returns:
            (is_degraded, comparison_dict)
        """
        if name not in self._strategies:
            return True, {"error": "Strategy not found"}

        expected = self._strategies[name].expected_metrics

        comparison = {
            "sharpe_expected": expected.sharpe_ratio,
            "sharpe_realized": realized_metrics.sharpe_ratio,
            "sharpe_degradation": (expected.sharpe_ratio - realized_metrics.sharpe_ratio) / max(expected.sharpe_ratio, 0.01),
            "win_rate_expected": expected.win_rate,
            "win_rate_realized": realized_metrics.win_rate,
            "profit_factor_expected": expected.profit_factor,
            "profit_factor_realized": realized_metrics.profit_factor,
        }

        # Check for significant degradation
        is_degraded = (
            realized_metrics.sharpe_ratio < expected.sharpe_ratio * 0.5
            or realized_metrics.win_rate < expected.win_rate * 0.8
            or realized_metrics.profit_factor < expected.profit_factor * 0.7
        )

        if is_degraded:
            logger.warning(f"Strategy {name} showing degradation: {comparison}")

        return is_degraded, comparison

    def get_enabled_strategies(self) -> List[StrategyEntry]:
        """Get all enabled strategies."""
        return [s for s in self._strategies.values() if s.enabled]

    def get_all_strategies(self) -> List[StrategyEntry]:
        """Get all strategies."""
        return list(self._strategies.values())

    def get_strategy(self, name: str) -> Optional[StrategyEntry]:
        """Get a specific strategy."""
        return self._strategies.get(name)

    def get_registry_summary(self) -> Dict:
        """Get summary of registry state."""
        strategies = list(self._strategies.values())

        return {
            "total_strategies": len(strategies),
            "enabled_strategies": len([s for s in strategies if s.enabled]),
            "disabled_strategies": len([s for s in strategies if not s.enabled]),
            "avg_sharpe": (
                sum(s.expected_metrics.sharpe_ratio for s in strategies) / len(strategies)
                if strategies else 0
            ),
            "strategies": [
                {
                    "name": s.name,
                    "enabled": s.enabled,
                    "sharpe": s.expected_metrics.sharpe_ratio,
                    "symbols": s.symbols,
                }
                for s in strategies
            ],
        }

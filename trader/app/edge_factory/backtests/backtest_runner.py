"""
Backtesting engine with walk-forward validation.
All backtests must use walk-forward to prevent overfitting.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from edge_factory.data.datasets import TradeDataset, DataSplit
from edge_factory.features.feature_engineering import FeatureEngineer
from edge_factory.models.anomaly_iforest import AnomalyDetector
from edge_factory.models.meta_labeler import MetaLabeler, MetaLabelResult
from shared.schemas import StrategyParameters, StrategyMetrics

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    n_splits: int = 5
    train_pct: float = 0.8
    gap_minutes: int = 5
    min_trades: int = 30
    min_sharpe: float = 0.0
    max_drawdown_pct: float = 20.0
    fees_bps: float = 4.0
    slippage_bps: float = 2.0


@dataclass
class BacktestTrade:
    """Record of a single backtest trade."""
    entry_time: datetime
    exit_time: datetime
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    gross_pnl: float
    fees: float
    net_pnl: float
    pnl_bps: float
    exit_reason: str


@dataclass
class BacktestResult:
    """Result of a backtest run."""
    symbol: str
    strategy_name: str
    parameters: StrategyParameters
    start_time: datetime
    end_time: datetime
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)

    # Metrics
    total_trades: int = 0
    win_rate: float = 0.0
    avg_win_bps: float = 0.0
    avg_loss_bps: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    total_return_pct: float = 0.0
    is_valid: bool = False
    rejection_reason: str = ""

    def to_metrics(self) -> StrategyMetrics:
        """Convert to StrategyMetrics."""
        return StrategyMetrics(
            sharpe_ratio=self.sharpe_ratio,
            win_rate=self.win_rate,
            avg_profit_bps=self.avg_win_bps,
            avg_loss_bps=abs(self.avg_loss_bps),
            max_drawdown_pct=self.max_drawdown_pct,
            trade_count=self.total_trades,
            profit_factor=self.profit_factor,
        )


class BacktestRunner:
    """
    Backtesting engine with walk-forward validation.

    CRITICAL REQUIREMENTS:
    - Must use walk-forward validation (no training on future data)
    - Must include fees and slippage
    - Must reject strategies with insufficient trades or poor metrics
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.feature_engineer = FeatureEngineer()
        self._total_cost_bps = (
            self.config.fees_bps * 2 + self.config.slippage_bps * 2
        )

    def run_walk_forward(
        self,
        dataset: TradeDataset,
        strategy_name: str,
        parameters: StrategyParameters,
        anomaly_threshold: float = 0.8,
    ) -> BacktestResult:
        """
        Run walk-forward backtest.

        For each split:
        1. Train anomaly detector on training data
        2. Generate signals on validation data
        3. Simulate trades with fees/slippage
        4. Aggregate results
        """
        logger.info(f"Running walk-forward backtest for {strategy_name}")

        # Get walk-forward splits
        splits = dataset.get_walk_forward_splits(
            n_splits=self.config.n_splits,
            train_pct=self.config.train_pct,
            gap_minutes=self.config.gap_minutes,
        )

        if not splits:
            return self._create_failed_result(
                dataset, strategy_name, parameters, "No valid splits"
            )

        all_trades = []
        equity = [0.0]

        for i, split in enumerate(splits):
            logger.info(f"Processing split {i + 1}/{len(splits)}")

            # Calculate features for train and val
            train_featured = self.feature_engineer.calculate_features(split.train_data)
            val_featured = self.feature_engineer.calculate_features(split.val_data)

            feature_cols = self.feature_engineer.get_feature_columns()
            feature_cols = [c for c in feature_cols if c in train_featured.columns]

            if len(feature_cols) == 0:
                continue

            # Train anomaly detector
            X_train = train_featured[feature_cols].values
            detector = AnomalyDetector()

            try:
                detector.fit(X_train, feature_cols)
            except ValueError as e:
                logger.warning(f"Could not fit detector: {e}")
                continue

            # Detect anomalies on validation
            signals = detector.detect_anomalies(
                val_featured,
                feature_cols,
                threshold=anomaly_threshold,
                symbol=dataset.symbol,
            )

            # Simulate trades
            split_trades = self._simulate_trades(
                signals=signals,
                price_data=val_featured,
                parameters=parameters,
            )

            all_trades.extend(split_trades)

            # Update equity curve
            for trade in split_trades:
                equity.append(equity[-1] + trade.net_pnl)

        # Calculate metrics
        result = self._calculate_metrics(
            trades=all_trades,
            equity_curve=equity,
            dataset=dataset,
            strategy_name=strategy_name,
            parameters=parameters,
        )

        # Validate result
        result = self._validate_result(result)

        return result

    def _simulate_trades(
        self,
        signals: List,
        price_data: pd.DataFrame,
        parameters: StrategyParameters,
    ) -> List[BacktestTrade]:
        """Simulate trades from anomaly signals."""
        trades = []
        notional = 100.0  # Normalized notional for backtest

        for signal in signals:
            # Find entry point
            entry_mask = price_data["timestamp"] >= signal.timestamp
            if not entry_mask.any():
                continue

            entry_idx = entry_mask.idxmax()
            entry_row = price_data.loc[entry_idx]
            entry_price = entry_row["close"]
            entry_time = entry_row["timestamp"]

            # Determine side based on order flow imbalance
            ofi = signal.features.get("order_flow_imbalance_60s", 0)
            side = "sell" if ofi > 0 else "buy"  # Mean reversion

            # Calculate exit levels
            if side == "buy":
                tp_price = entry_price * (1 + parameters.take_profit_bps / 10000)
                sl_price = entry_price * (1 - parameters.stop_loss_bps / 10000)
            else:
                tp_price = entry_price * (1 - parameters.take_profit_bps / 10000)
                sl_price = entry_price * (1 + parameters.stop_loss_bps / 10000)

            # Find exit
            horizon_end = entry_time + pd.Timedelta(seconds=parameters.time_stop_seconds)
            forward_mask = (
                (price_data["timestamp"] > entry_time)
                & (price_data["timestamp"] <= horizon_end)
            )
            forward_data = price_data[forward_mask]

            if len(forward_data) == 0:
                continue

            # Check TP/SL
            exit_price = None
            exit_reason = "time_stop"
            exit_time = forward_data.iloc[-1]["timestamp"]

            for _, row in forward_data.iterrows():
                if side == "buy":
                    if row["high"] >= tp_price:
                        exit_price = tp_price
                        exit_reason = "take_profit"
                        exit_time = row["timestamp"]
                        break
                    if row["low"] <= sl_price:
                        exit_price = sl_price
                        exit_reason = "stop_loss"
                        exit_time = row["timestamp"]
                        break
                else:
                    if row["low"] <= tp_price:
                        exit_price = tp_price
                        exit_reason = "take_profit"
                        exit_time = row["timestamp"]
                        break
                    if row["high"] >= sl_price:
                        exit_price = sl_price
                        exit_reason = "stop_loss"
                        exit_time = row["timestamp"]
                        break

            if exit_price is None:
                exit_price = forward_data.iloc[-1]["close"]

            # Calculate PnL
            quantity = notional / entry_price
            if side == "buy":
                gross_pnl = (exit_price - entry_price) * quantity
            else:
                gross_pnl = (entry_price - exit_price) * quantity

            fees = notional * (self._total_cost_bps / 10000)
            net_pnl = gross_pnl - fees
            pnl_bps = net_pnl / notional * 10000

            trade = BacktestTrade(
                entry_time=entry_time,
                exit_time=exit_time,
                symbol=signal.symbol,
                side=side,
                entry_price=entry_price,
                exit_price=exit_price,
                quantity=quantity,
                gross_pnl=gross_pnl,
                fees=fees,
                net_pnl=net_pnl,
                pnl_bps=pnl_bps,
                exit_reason=exit_reason,
            )
            trades.append(trade)

        return trades

    def _calculate_metrics(
        self,
        trades: List[BacktestTrade],
        equity_curve: List[float],
        dataset: TradeDataset,
        strategy_name: str,
        parameters: StrategyParameters,
    ) -> BacktestResult:
        """Calculate backtest metrics."""
        result = BacktestResult(
            symbol=dataset.symbol,
            strategy_name=strategy_name,
            parameters=parameters,
            start_time=dataset.start_time,
            end_time=dataset.end_time,
            trades=trades,
            equity_curve=equity_curve,
            total_trades=len(trades),
        )

        if not trades:
            return result

        pnls = [t.pnl_bps for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        result.win_rate = len(wins) / len(trades)
        result.avg_win_bps = np.mean(wins) if wins else 0
        result.avg_loss_bps = np.mean(losses) if losses else 0

        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 0
        result.profit_factor = (
            total_wins / total_losses if total_losses > 0 else float("inf")
        )

        # Sharpe ratio (annualized, assuming 1-minute bars)
        if len(pnls) > 1:
            mean_pnl = np.mean(pnls)
            std_pnl = np.std(pnls)
            result.sharpe_ratio = (
                mean_pnl / std_pnl * np.sqrt(525600) if std_pnl > 0 else 0
            )

        # Max drawdown
        equity = np.array(equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity)
        if peak.max() > 0:
            result.max_drawdown_pct = (drawdown / np.maximum(peak, 1)).max() * 100
        else:
            result.max_drawdown_pct = 0

        # Total return
        if len(equity_curve) > 1:
            result.total_return_pct = (equity_curve[-1] / 100) * 100  # Based on normalized notional

        return result

    def _validate_result(self, result: BacktestResult) -> BacktestResult:
        """Validate backtest result against thresholds."""
        # Check trade count
        if result.total_trades < self.config.min_trades:
            result.is_valid = False
            result.rejection_reason = (
                f"Insufficient trades: {result.total_trades} < {self.config.min_trades}"
            )
            logger.warning(result.rejection_reason)
            return result

        # Check Sharpe ratio
        if result.sharpe_ratio <= self.config.min_sharpe:
            result.is_valid = False
            result.rejection_reason = (
                f"Sharpe too low: {result.sharpe_ratio:.2f} <= {self.config.min_sharpe}"
            )
            logger.warning(result.rejection_reason)
            return result

        # Check drawdown
        if result.max_drawdown_pct > self.config.max_drawdown_pct:
            result.is_valid = False
            result.rejection_reason = (
                f"Drawdown too high: {result.max_drawdown_pct:.2f}% > {self.config.max_drawdown_pct}%"
            )
            logger.warning(result.rejection_reason)
            return result

        result.is_valid = True
        logger.info(
            f"Strategy validated: {result.strategy_name} "
            f"(trades={result.total_trades}, sharpe={result.sharpe_ratio:.2f}, "
            f"dd={result.max_drawdown_pct:.2f}%)"
        )
        return result

    def _create_failed_result(
        self,
        dataset: TradeDataset,
        strategy_name: str,
        parameters: StrategyParameters,
        reason: str,
    ) -> BacktestResult:
        """Create a failed backtest result."""
        return BacktestResult(
            symbol=dataset.symbol,
            strategy_name=strategy_name,
            parameters=parameters,
            start_time=dataset.start_time,
            end_time=dataset.end_time,
            is_valid=False,
            rejection_reason=reason,
        )

    def run_regime_validation(
        self,
        dataset: TradeDataset,
        strategy_name: str,
        parameters: StrategyParameters,
    ) -> Dict[str, BacktestResult]:
        """
        Run backtest across different volatility regimes.
        Strategy must work across all regimes to be valid.
        """
        # First calculate features to get volatility column
        featured = self.feature_engineer.calculate_features(dataset.data)

        # Create featured dataset
        featured_dataset = TradeDataset(featured, dataset.symbol)

        regimes = featured_dataset.get_regime_splits(
            volatility_col="realized_volatility_60s"
        )

        results = {}
        for regime_name, regime_data in regimes.items():
            regime_dataset = TradeDataset(regime_data, dataset.symbol)

            if len(regime_data) < 200:
                logger.warning(f"Insufficient data for regime {regime_name}")
                continue

            result = self.run_walk_forward(
                dataset=regime_dataset,
                strategy_name=f"{strategy_name}_{regime_name}",
                parameters=parameters,
            )
            results[regime_name] = result

        return results

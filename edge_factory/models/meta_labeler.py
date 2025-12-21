"""
Meta-labeling for anomaly signals.
Evaluates whether anomalies historically produced positive expectancy.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from shared.schemas import MetaLabel, AnomalySignal

logger = logging.getLogger(__name__)


@dataclass
class MetaLabelConfig:
    """Configuration for meta-labeling."""
    horizon_minutes: int = 1
    take_profit_bps: float = 8.0
    stop_loss_bps: float = 5.0
    fees_bps: float = 4.0
    slippage_bps: float = 2.0
    min_samples: int = 30


@dataclass
class MetaLabelResult:
    """Result of meta-labeling an anomaly."""
    timestamp: pd.Timestamp
    symbol: str
    anomaly_score: float
    label: MetaLabel
    forward_return_bps: float
    net_return_bps: float
    hit_tp: bool
    hit_sl: bool
    exit_reason: str


class MetaLabeler:
    """
    Meta-labeler for anomaly signals.

    CRITICAL RULES:
    - Must include fees and slippage in all calculations
    - Must be strictly causal (no future leakage in training data)
    - Labels are based on historical outcomes, not predictions

    The meta-labeler does NOT predict future returns.
    It only evaluates whether past anomalies were profitable.
    """

    def __init__(self, config: Optional[MetaLabelConfig] = None):
        self.config = config or MetaLabelConfig()
        self._total_cost_bps = (
            self.config.fees_bps * 2 + self.config.slippage_bps * 2
        )

    def label_anomalies(
        self,
        signals: List[AnomalySignal],
        price_data: pd.DataFrame,
        timestamp_col: str = "timestamp",
        close_col: str = "close",
        high_col: str = "high",
        low_col: str = "low",
    ) -> List[MetaLabelResult]:
        """
        Label anomaly signals based on forward outcomes.

        IMPORTANT: This uses future data for labeling.
        Only use for training, never for live trading.
        """
        results = []

        for signal in signals:
            result = self._label_single(
                signal=signal,
                price_data=price_data,
                timestamp_col=timestamp_col,
                close_col=close_col,
                high_col=high_col,
                low_col=low_col,
            )
            if result:
                results.append(result)

        return results

    def _label_single(
        self,
        signal: AnomalySignal,
        price_data: pd.DataFrame,
        timestamp_col: str,
        close_col: str,
        high_col: str,
        low_col: str,
    ) -> Optional[MetaLabelResult]:
        """Label a single anomaly signal."""
        # Find entry point
        entry_mask = price_data[timestamp_col] >= signal.timestamp
        if not entry_mask.any():
            return None

        entry_idx = entry_mask.idxmax()
        entry_price = price_data.loc[entry_idx, close_col]

        # Calculate exit levels
        tp_price = entry_price * (1 + self.config.take_profit_bps / 10000)
        sl_price = entry_price * (1 - self.config.stop_loss_bps / 10000)

        # Look forward for horizon period
        horizon_end = signal.timestamp + pd.Timedelta(minutes=self.config.horizon_minutes)
        forward_mask = (
            (price_data[timestamp_col] > signal.timestamp)
            & (price_data[timestamp_col] <= horizon_end)
        )

        forward_data = price_data[forward_mask]
        if len(forward_data) == 0:
            return None

        # Check for TP/SL hits
        hit_tp = (forward_data[high_col] >= tp_price).any()
        hit_sl = (forward_data[low_col] <= sl_price).any()

        # Determine exit
        if hit_tp and hit_sl:
            # Both hit, need to determine which first
            tp_idx = forward_data[forward_data[high_col] >= tp_price].index[0]
            sl_idx = forward_data[forward_data[low_col] <= sl_price].index[0]
            if tp_idx <= sl_idx:
                exit_price = tp_price
                exit_reason = "take_profit"
                hit_sl = False
            else:
                exit_price = sl_price
                exit_reason = "stop_loss"
                hit_tp = False
        elif hit_tp:
            exit_price = tp_price
            exit_reason = "take_profit"
        elif hit_sl:
            exit_price = sl_price
            exit_reason = "stop_loss"
        else:
            # Exit at end of horizon
            exit_price = forward_data.iloc[-1][close_col]
            exit_reason = "time_stop"

        # Calculate returns
        gross_return_bps = (exit_price / entry_price - 1) * 10000
        net_return_bps = gross_return_bps - self._total_cost_bps

        # Assign label
        if net_return_bps > 2:  # Small positive threshold
            label = MetaLabel.PROFITABLE
        elif net_return_bps < -2:  # Small negative threshold
            label = MetaLabel.ADVERSE
        else:
            label = MetaLabel.NEUTRAL

        return MetaLabelResult(
            timestamp=signal.timestamp,
            symbol=signal.symbol,
            anomaly_score=signal.anomaly_score,
            label=label,
            forward_return_bps=gross_return_bps,
            net_return_bps=net_return_bps,
            hit_tp=hit_tp,
            hit_sl=hit_sl,
            exit_reason=exit_reason,
        )

    def calculate_statistics(
        self,
        results: List[MetaLabelResult],
    ) -> Dict:
        """Calculate statistics from meta-labeling results."""
        if not results:
            return {}

        df = pd.DataFrame([
            {
                "label": r.label.value,
                "net_return_bps": r.net_return_bps,
                "hit_tp": r.hit_tp,
                "hit_sl": r.hit_sl,
                "exit_reason": r.exit_reason,
            }
            for r in results
        ])

        n_total = len(df)
        n_profitable = len(df[df["label"] == "profitable"])
        n_adverse = len(df[df["label"] == "adverse"])
        n_neutral = len(df[df["label"] == "neutral"])

        returns = df["net_return_bps"].values
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe = mean_return / std_return if std_return > 0 else 0

        wins = df[df["net_return_bps"] > 0]
        losses = df[df["net_return_bps"] < 0]

        win_rate = len(wins) / n_total if n_total > 0 else 0
        avg_win = wins["net_return_bps"].mean() if len(wins) > 0 else 0
        avg_loss = losses["net_return_bps"].mean() if len(losses) > 0 else 0
        profit_factor = (
            abs(wins["net_return_bps"].sum() / losses["net_return_bps"].sum())
            if len(losses) > 0 and losses["net_return_bps"].sum() != 0
            else float("inf")
        )

        return {
            "total_signals": n_total,
            "profitable": n_profitable,
            "adverse": n_adverse,
            "neutral": n_neutral,
            "win_rate": win_rate,
            "mean_return_bps": mean_return,
            "std_return_bps": std_return,
            "sharpe_ratio": sharpe,
            "avg_win_bps": avg_win,
            "avg_loss_bps": avg_loss,
            "profit_factor": profit_factor,
            "tp_hit_rate": df["hit_tp"].mean(),
            "sl_hit_rate": df["hit_sl"].mean(),
        }

    def filter_profitable_signals(
        self,
        results: List[MetaLabelResult],
        min_win_rate: float = 0.5,
    ) -> Tuple[List[MetaLabelResult], Dict]:
        """
        Filter to only profitable signal types.
        Returns filtered results and statistics.
        """
        stats = self.calculate_statistics(results)

        if stats.get("win_rate", 0) < min_win_rate:
            logger.warning(
                f"Win rate {stats.get('win_rate', 0):.2%} below threshold {min_win_rate:.2%}"
            )
            return [], stats

        profitable_results = [
            r for r in results
            if r.label in [MetaLabel.PROFITABLE, MetaLabel.NEUTRAL]
        ]

        return profitable_results, stats

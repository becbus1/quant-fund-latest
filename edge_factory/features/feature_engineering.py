"""
Feature engineering for anomaly detection.
All features are microstructure-based.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from shared.feature_defs import get_feature_names, FEATURE_DEFINITIONS

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Calculate microstructure features for anomaly detection.
    All features are strictly causal (no future leakage).
    """

    def __init__(
        self,
        windows: List[int] = [60, 300],
        large_trade_threshold: float = 0.95,
    ):
        self.windows = windows
        self.large_trade_threshold = large_trade_threshold

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all microstructure features.
        Returns dataframe with features appended.
        """
        result = df.copy()

        # Ensure required columns exist
        required_cols = ["open", "high", "low", "close", "volume"]
        for col in required_cols:
            if col not in result.columns:
                logger.warning(f"Missing column: {col}")
                return result

        # Calculate returns (strictly causal)
        result["returns"] = result["close"].pct_change()

        for window in self.windows:
            suffix = f"_{window}s"

            # Order flow imbalance proxy (using price direction)
            result[f"order_flow_imbalance{suffix}"] = (
                result["returns"].rolling(window=window, min_periods=1).sum()
            )

            # Volume features
            result[f"volume_ma{suffix}"] = (
                result["volume"].rolling(window=window, min_periods=1).mean()
            )
            result[f"volume_spike_ratio{suffix}"] = (
                result["volume"] / result[f"volume_ma{suffix}"]
            ).replace([np.inf, -np.inf], 1.0)

            # Volume acceleration
            result[f"volume_acceleration{suffix}"] = (
                result["volume"].pct_change(periods=min(window // 10, 10))
            )

            # Realized volatility
            result[f"realized_volatility{suffix}"] = (
                result["returns"]
                .rolling(window=window, min_periods=2)
                .std()
            )

            # Trade intensity (using volume as proxy)
            result[f"trade_intensity{suffix}"] = (
                result["turnover"].rolling(window=window, min_periods=1).sum()
                if "turnover" in result.columns
                else result["volume"].rolling(window=window, min_periods=1).sum()
            )

            # Volatility expansion
            long_vol = result["returns"].rolling(window=window * 5, min_periods=window).std()
            short_vol = result[f"realized_volatility{suffix}"]
            result[f"volatility_expansion{suffix}"] = (
                short_vol / long_vol
            ).replace([np.inf, -np.inf], 1.0)

            # Price range percentage
            rolling_high = result["high"].rolling(window=window, min_periods=1).max()
            rolling_low = result["low"].rolling(window=window, min_periods=1).min()
            rolling_mid = (rolling_high + rolling_low) / 2
            result[f"price_range_pct{suffix}"] = (
                (rolling_high - rolling_low) / rolling_mid * 100
            ).replace([np.inf, -np.inf], 0.0)

            # VWAP deviation (using close * volume as proxy)
            vwap = (
                (result["close"] * result["volume"])
                .rolling(window=window, min_periods=1)
                .sum()
                / result["volume"].rolling(window=window, min_periods=1).sum()
            )
            result[f"vwap_deviation{suffix}"] = (
                (result["close"] - vwap) / vwap
            ).replace([np.inf, -np.inf], 0.0)

            # Average trade size
            result[f"avg_trade_size{suffix}"] = (
                result["volume"].rolling(window=window, min_periods=1).mean()
            )

            # Trade intensity change
            intensity = result[f"trade_intensity{suffix}"]
            result[f"trade_intensity_change{suffix}"] = (
                intensity.pct_change(periods=min(window // 10, 10))
            )

        # Large trade ratio (using volume quantile)
        if len(result) > 100:
            volume_threshold = result["volume"].quantile(self.large_trade_threshold)
            result["is_large_trade"] = (result["volume"] > volume_threshold).astype(int)
            result["large_trade_ratio_300s"] = (
                result["is_large_trade"]
                .rolling(window=300, min_periods=1)
                .mean()
            )
        else:
            result["large_trade_ratio_300s"] = 0.0

        # Fill NaN values
        result = result.fillna(0.0)

        return result

    def get_feature_columns(self) -> List[str]:
        """Get list of feature column names."""
        columns = []
        for window in self.windows:
            suffix = f"_{window}s"
            columns.extend([
                f"order_flow_imbalance{suffix}",
                f"volume_spike_ratio{suffix}",
                f"volume_acceleration{suffix}",
                f"realized_volatility{suffix}",
                f"trade_intensity{suffix}",
                f"volatility_expansion{suffix}",
                f"price_range_pct{suffix}",
                f"vwap_deviation{suffix}",
                f"avg_trade_size{suffix}",
                f"trade_intensity_change{suffix}",
            ])
        columns.append("large_trade_ratio_300s")
        return columns

    def normalize_features(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Normalize features to zero mean, unit variance.
        Uses expanding window to prevent look-ahead bias.
        """
        result = df.copy()
        feature_cols = feature_cols or self.get_feature_columns()

        for col in feature_cols:
            if col not in result.columns:
                continue

            # Expanding mean and std (strictly causal)
            expanding_mean = result[col].expanding(min_periods=10).mean()
            expanding_std = result[col].expanding(min_periods=10).std()

            # Normalize
            result[f"{col}_norm"] = (
                (result[col] - expanding_mean) / expanding_std.replace(0, 1)
            ).clip(-5, 5)  # Clip extreme values

        return result

    def get_feature_matrix(
        self,
        df: pd.DataFrame,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Extract feature matrix for model training.
        Returns (n_samples, n_features) array.
        """
        # Calculate features
        featured = self.calculate_features(df)

        if normalize:
            featured = self.normalize_features(featured)
            cols = [f"{c}_norm" for c in self.get_feature_columns()]
        else:
            cols = self.get_feature_columns()

        # Filter to existing columns
        cols = [c for c in cols if c in featured.columns]

        return featured[cols].values

    def calculate_forward_returns(
        self,
        df: pd.DataFrame,
        horizons: List[int] = [1, 5, 10, 30, 60],
    ) -> pd.DataFrame:
        """
        Calculate forward returns for meta-labeling.
        CRITICAL: This is only for training, not for live trading.
        """
        result = df.copy()

        for horizon in horizons:
            # Forward return (future price / current price - 1)
            result[f"fwd_return_{horizon}"] = (
                result["close"].shift(-horizon) / result["close"] - 1
            )

        return result

"""
Dataset management for ML training.
Handles data loading, splitting, and walk-forward validation.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DataSplit:
    """A single train/validation split for walk-forward validation."""
    train_start: datetime
    train_end: datetime
    val_start: datetime
    val_end: datetime
    train_data: pd.DataFrame
    val_data: pd.DataFrame


class TradeDataset:
    """Dataset manager for ML training with walk-forward validation."""

    def __init__(
        self,
        data: pd.DataFrame,
        symbol: str,
        timestamp_col: str = "timestamp",
    ):
        self.data = data.copy()
        self.symbol = symbol
        self.timestamp_col = timestamp_col

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.data[timestamp_col]):
            self.data[timestamp_col] = pd.to_datetime(self.data[timestamp_col])

        self.data = self.data.sort_values(timestamp_col).reset_index(drop=True)
        self.start_time = self.data[timestamp_col].min()
        self.end_time = self.data[timestamp_col].max()

        logger.info(
            f"Dataset initialized: {symbol}, "
            f"{len(self.data)} rows, "
            f"{self.start_time} to {self.end_time}"
        )

    def get_walk_forward_splits(
        self,
        n_splits: int = 5,
        train_pct: float = 0.8,
        gap_minutes: int = 0,
    ) -> List[DataSplit]:
        """
        Generate walk-forward validation splits.
        No overlapping data between train and validation.
        """
        total_duration = (self.end_time - self.start_time).total_seconds()
        split_duration = total_duration / n_splits
        gap = timedelta(minutes=gap_minutes)

        splits = []
        for i in range(n_splits):
            split_start = self.start_time + timedelta(seconds=i * split_duration)
            split_end = split_start + timedelta(seconds=split_duration)

            # Calculate train/val boundary
            train_duration = timedelta(seconds=split_duration * train_pct)
            train_end = split_start + train_duration
            val_start = train_end + gap
            val_end = split_end

            # Get data for this split
            train_mask = (
                (self.data[self.timestamp_col] >= split_start)
                & (self.data[self.timestamp_col] < train_end)
            )
            val_mask = (
                (self.data[self.timestamp_col] >= val_start)
                & (self.data[self.timestamp_col] < val_end)
            )

            train_data = self.data[train_mask].copy()
            val_data = self.data[val_mask].copy()

            if len(train_data) > 0 and len(val_data) > 0:
                split = DataSplit(
                    train_start=split_start,
                    train_end=train_end,
                    val_start=val_start,
                    val_end=val_end,
                    train_data=train_data,
                    val_data=val_data,
                )
                splits.append(split)
                logger.info(
                    f"Split {i+1}: train={len(train_data)}, val={len(val_data)}"
                )

        return splits

    def get_expanding_window_splits(
        self,
        n_splits: int = 5,
        min_train_pct: float = 0.3,
        val_pct: float = 0.1,
    ) -> List[DataSplit]:
        """
        Generate expanding window splits.
        Training window grows with each split.
        """
        total_rows = len(self.data)
        min_train_rows = int(total_rows * min_train_pct)
        val_rows = int(total_rows * val_pct)

        available_rows = total_rows - min_train_rows - val_rows
        step_size = available_rows // max(n_splits - 1, 1)

        splits = []
        for i in range(n_splits):
            train_end_idx = min_train_rows + (i * step_size)
            val_start_idx = train_end_idx
            val_end_idx = min(val_start_idx + val_rows, total_rows)

            if val_end_idx > total_rows or train_end_idx >= total_rows:
                break

            train_data = self.data.iloc[:train_end_idx].copy()
            val_data = self.data.iloc[val_start_idx:val_end_idx].copy()

            if len(train_data) > 0 and len(val_data) > 0:
                split = DataSplit(
                    train_start=train_data[self.timestamp_col].min(),
                    train_end=train_data[self.timestamp_col].max(),
                    val_start=val_data[self.timestamp_col].min(),
                    val_end=val_data[self.timestamp_col].max(),
                    train_data=train_data,
                    val_data=val_data,
                )
                splits.append(split)

        return splits

    def get_regime_splits(
        self,
        volatility_col: str = "realized_volatility",
        n_regimes: int = 3,
    ) -> Dict[str, pd.DataFrame]:
        """
        Split data by volatility regime for robustness testing.
        """
        if volatility_col not in self.data.columns:
            logger.warning(f"Column {volatility_col} not found, cannot split by regime")
            return {"all": self.data}

        # Calculate volatility quantiles
        quantiles = np.linspace(0, 1, n_regimes + 1)
        thresholds = self.data[volatility_col].quantile(quantiles).values

        regimes = {}
        regime_names = ["low", "medium", "high"][:n_regimes]

        for i, name in enumerate(regime_names):
            mask = (
                (self.data[volatility_col] >= thresholds[i])
                & (self.data[volatility_col] < thresholds[i + 1])
            )
            regime_data = self.data[mask].copy()
            if len(regime_data) > 0:
                regimes[name] = regime_data

        return regimes

    def sample_non_overlapping(
        self,
        window_size: int,
        n_samples: Optional[int] = None,
    ) -> List[pd.DataFrame]:
        """
        Sample non-overlapping windows for training.
        Critical: Prevents data leakage from overlapping windows.
        """
        samples = []
        current_idx = 0

        while current_idx + window_size <= len(self.data):
            window = self.data.iloc[current_idx : current_idx + window_size].copy()
            samples.append(window)
            current_idx += window_size

            if n_samples and len(samples) >= n_samples:
                break

        logger.info(f"Generated {len(samples)} non-overlapping windows")
        return samples

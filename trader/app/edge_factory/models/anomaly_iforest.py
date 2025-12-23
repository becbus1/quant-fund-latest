"""
Anomaly detection using Isolation Forest.
Detects statistically rare market states.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from shared.schemas import AnomalySignal

logger = logging.getLogger(__name__)


@dataclass
class AnomalyDetectorConfig:
    """Configuration for anomaly detector."""
    n_estimators: int = 100
    contamination: float = 0.05  # Expected proportion of anomalies
    max_samples: str = "auto"
    random_state: int = 42
    min_samples_for_fit: int = 100


class AnomalyDetector:
    """
    Isolation Forest based anomaly detector.
    Detects rare market microstructure states.

    IMPORTANT: This does NOT predict prices or returns.
    It only identifies statistically unusual market conditions.
    """

    def __init__(self, config: Optional[AnomalyDetectorConfig] = None):
        self.config = config or AnomalyDetectorConfig()
        self.model: Optional[IsolationForest] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []
        self.is_fitted = False
        self._fit_timestamp: Optional[datetime] = None

    def fit(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "AnomalyDetector":
        """
        Fit the anomaly detector.

        Args:
            X: Feature matrix (n_samples, n_features)
            feature_names: Names of features for interpretability
        """
        if len(X) < self.config.min_samples_for_fit:
            raise ValueError(
                f"Insufficient samples: {len(X)} < {self.config.min_samples_for_fit}"
            )

        logger.info(f"Fitting anomaly detector on {len(X)} samples")

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Fit Isolation Forest
        self.model = IsolationForest(
            n_estimators=self.config.n_estimators,
            contamination=self.config.contamination,
            max_samples=self.config.max_samples,
            random_state=self.config.random_state,
            n_jobs=-1,
        )
        self.model.fit(X_scaled)

        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        self.is_fitted = True
        self._fit_timestamp = datetime.utcnow()

        logger.info("Anomaly detector fitted successfully")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly labels.

        Returns:
            Array of -1 (anomaly) or 1 (normal)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores for samples.
        More negative = more anomalous.

        Returns:
            Array of anomaly scores (higher = more anomalous, normalized 0-1)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_scaled = self.scaler.transform(X)
        raw_scores = self.model.score_samples(X_scaled)

        # Convert to 0-1 range (higher = more anomalous)
        # Isolation Forest returns negative scores, more negative = more anomalous
        normalized = 1 - (raw_scores - raw_scores.min()) / (
            raw_scores.max() - raw_scores.min() + 1e-10
        )

        return normalized

    def detect_anomalies(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        threshold: float = 0.8,
        symbol: str = "",
    ) -> List[AnomalySignal]:
        """
        Detect anomalies in a dataframe.

        Args:
            df: DataFrame with features
            feature_cols: Columns to use as features
            threshold: Anomaly score threshold (0-1)
            symbol: Symbol for signals

        Returns:
            List of anomaly signals
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Extract features
        feature_cols = [c for c in feature_cols if c in df.columns]
        X = df[feature_cols].values

        # Get scores
        scores = self.score_samples(X)

        # Generate signals for anomalies above threshold
        signals = []
        for i, score in enumerate(scores):
            if score >= threshold:
                row = df.iloc[i]
                timestamp = row.get("timestamp", datetime.utcnow())
                if isinstance(timestamp, str):
                    timestamp = pd.to_datetime(timestamp)

                features = {col: float(row[col]) for col in feature_cols}

                signal = AnomalySignal(
                    timestamp=timestamp,
                    symbol=symbol,
                    anomaly_score=float(score),
                    features=features,
                )
                signals.append(signal)

        logger.info(f"Detected {len(signals)} anomalies (threshold={threshold})")
        return signals

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Estimate feature importance based on average path length contribution.
        Note: This is an approximation for Isolation Forest.
        """
        if not self.is_fitted or not self.model:
            return {}

        # Use feature importances from tree averaging
        importances = np.zeros(len(self.feature_names))

        for tree in self.model.estimators_:
            importances += tree.feature_importances_

        importances /= len(self.model.estimators_)

        return dict(zip(self.feature_names, importances))

    def get_stats(self) -> Dict:
        """Get detector statistics."""
        return {
            "is_fitted": self.is_fitted,
            "fit_timestamp": self._fit_timestamp.isoformat() if self._fit_timestamp else None,
            "n_features": len(self.feature_names),
            "n_estimators": self.config.n_estimators,
            "contamination": self.config.contamination,
        }

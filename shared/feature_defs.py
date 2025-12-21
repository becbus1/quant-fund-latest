"""
Shared feature definitions used by both trader and edge_factory.
All features are microstructure-based.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List


class FeatureType(Enum):
    ORDER_FLOW = "order_flow"
    VOLUME = "volume"
    VOLATILITY = "volatility"
    SPREAD = "spread"
    INTENSITY = "intensity"


@dataclass
class FeatureDefinition:
    name: str
    feature_type: FeatureType
    window_seconds: int
    description: str


# Core microstructure features for anomaly detection
FEATURE_DEFINITIONS: List[FeatureDefinition] = [
    FeatureDefinition(
        name="order_flow_imbalance",
        feature_type=FeatureType.ORDER_FLOW,
        window_seconds=60,
        description="Buy volume minus sell volume normalized by total volume",
    ),
    FeatureDefinition(
        name="order_flow_imbalance_5m",
        feature_type=FeatureType.ORDER_FLOW,
        window_seconds=300,
        description="Order flow imbalance over 5 minute window",
    ),
    FeatureDefinition(
        name="volume_spike_ratio",
        feature_type=FeatureType.VOLUME,
        window_seconds=60,
        description="Current volume vs rolling average volume",
    ),
    FeatureDefinition(
        name="volume_acceleration",
        feature_type=FeatureType.VOLUME,
        window_seconds=120,
        description="Rate of change in volume",
    ),
    FeatureDefinition(
        name="realized_volatility",
        feature_type=FeatureType.VOLATILITY,
        window_seconds=60,
        description="Standard deviation of log returns",
    ),
    FeatureDefinition(
        name="volatility_expansion",
        feature_type=FeatureType.VOLATILITY,
        window_seconds=300,
        description="Current volatility vs baseline volatility",
    ),
    FeatureDefinition(
        name="trade_intensity",
        feature_type=FeatureType.INTENSITY,
        window_seconds=60,
        description="Number of trades per second",
    ),
    FeatureDefinition(
        name="trade_intensity_change",
        feature_type=FeatureType.INTENSITY,
        window_seconds=120,
        description="Rate of change in trade intensity",
    ),
    FeatureDefinition(
        name="avg_trade_size",
        feature_type=FeatureType.VOLUME,
        window_seconds=60,
        description="Average trade size in quote currency",
    ),
    FeatureDefinition(
        name="large_trade_ratio",
        feature_type=FeatureType.VOLUME,
        window_seconds=300,
        description="Ratio of large trades to total trades",
    ),
    FeatureDefinition(
        name="price_range_pct",
        feature_type=FeatureType.VOLATILITY,
        window_seconds=60,
        description="High-low range as percentage of mid price",
    ),
    FeatureDefinition(
        name="vwap_deviation",
        feature_type=FeatureType.ORDER_FLOW,
        window_seconds=60,
        description="Current price deviation from VWAP",
    ),
]


def get_feature_names() -> List[str]:
    """Return list of all feature names."""
    return [f.name for f in FEATURE_DEFINITIONS]


def get_features_by_type(feature_type: FeatureType) -> List[FeatureDefinition]:
    """Return features filtered by type."""
    return [f for f in FEATURE_DEFINITIONS if f.feature_type == feature_type]

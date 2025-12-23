"""
Rolling Order Flow Imbalance (OFI) Z-score.
"""

from collections import deque
from statistics import mean, stdev
from typing import Deque, Dict

from shared.schemas import Side

ROLLING_WINDOW = 120
MIN_OBSERVATIONS = 3

_ofi_buffers: Dict[str, Deque[float]] = {}

def compute_z_score(symbol: str, side: Side) -> float:
    buffer = _ofi_buffers.get(symbol)

    if buffer is None:
        buffer = deque(maxlen=ROLLING_WINDOW)
        _ofi_buffers[symbol] = buffer

    ofi_value = 1.0 if side == Side.BUY else -1.0
    buffer.append(ofi_value)

    if len(buffer) < MIN_OBSERVATIONS:
        return 0.0

    mu = mean(buffer)

    try:
        sigma = stdev(buffer)
    except Exception:
        return 0.0

    if sigma == 0:
        return 0.0

    z = (ofi_value - mu) / sigma
    return max(min(z, 6.0), -6.0)

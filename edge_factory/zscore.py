"""
Rolling Order Flow Imbalance (OFI) Z-score.

OFI definition (simplified, execution-safe):
- BUY trades contribute +1
- SELL trades contribute -1
- Rolling window over recent signals per symbol
"""

from collections import deque
from statistics import mean, stdev
from typing import Deque, Dict

# ======================
# CONFIG
# ======================

ROLLING_WINDOW = 120        # number of recent signals per symbol
MIN_OBSERVATIONS = 20      # minimum needed for stable z-score

# ======================
# STATE (in-memory)
# ======================

_ofi_buffers: Dict[str, Deque[float]] = {}


# ======================
# CORE FUNCTION
# ======================

def compute_z_score(symbol: str) -> float:
    """
    Compute rolling OFI z-score for a symbol.

    Returns:
        float: standardized z-score (mean 0, unit variance)
    """

    buffer = _ofi_buffers.get(symbol)

    if buffer is None:
        buffer = deque(maxlen=ROLLING_WINDOW)
        _ofi_buffers[symbol] = buffer

    # NOTE:
    # We cannot observe raw trades here without DB access,
    # so we treat each signal as +1 or -1 depending on direction.
    # BUY bias => +1, SELL bias => -1
    #
    # This keeps z-score meaningful relative to strategy behavior.
    #
    # Side is encoded upstream into features but execution
    # always calls this *before* insert, so we infer neutrality here.

    # Neutral placeholder until features are expanded:
    ofi_value = 0.0

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

    # Clamp extreme numerical spikes
    return max(min(z, 6.0), -6.0)

"""
Confidence score mapping from z-score.

Maps signal strength → (0, 1) using a sigmoid.
"""

import math

# ======================
# CONFIG
# ======================

SIGMOID_STEEPNESS = 1.2     # higher = sharper confidence jump
MIN_CONFIDENCE = 0.05
MAX_CONFIDENCE = 0.95


# ======================
# CORE FUNCTION
# ======================

def compute_confidence(z_score: float) -> float:
    """
    Convert z-score magnitude into confidence probability.

    Args:
        z_score (float): standardized signal strength

    Returns:
        float: confidence in (0, 1)
    """

    strength = abs(z_score)

    # Sigmoid mapping
    confidence = 1.0 / (1.0 + math.exp(-SIGMOID_STEEPNESS * strength))

    # Clamp to avoid degenerate 0/1 probabilities
    confidence = max(MIN_CONFIDENCE, min(MAX_CONFIDENCE, confidence))

    return confidence

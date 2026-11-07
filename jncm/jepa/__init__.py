"""JEPA joint-mode modules (SPEC section 8): EMA target encoder."""

from jncm.jepa.ema import EMAModule, cosine_momentum

__all__ = [
    "EMAModule",
    "cosine_momentum",
]

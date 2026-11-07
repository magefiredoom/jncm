"""JEPA joint-mode modules (SPEC section 8): EMA target encoder and VICReg."""

from jncm.jepa.ema import EMAModule, cosine_momentum
from jncm.jepa.vicreg import covariance_term, variance_term, vicreg_loss

__all__ = [
    "EMAModule",
    "cosine_momentum",
    "covariance_term",
    "variance_term",
    "vicreg_loss",
]

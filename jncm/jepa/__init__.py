"""JEPA joint-mode modules (SPEC section 8): EMA target, VICReg, joint loss."""

from jncm.jepa.diagnostics import (
    COLLAPSE_PATIENCE,
    COLLAPSE_STD,
    CollapseAlarm,
    code_diagnostics,
    effective_rank,
    explained_variance,
    std_summary,
)
from jncm.jepa.ema import EMAModule, cosine_momentum
from jncm.jepa.joint import (
    JEPA_HEADS,
    JointStep,
    encode_clean,
    integrate_flow,
    joint_loss,
    predicted_codes,
)
from jncm.jepa.vicreg import covariance_term, variance_term, vicreg_loss

__all__ = [
    "COLLAPSE_PATIENCE",
    "COLLAPSE_STD",
    "JEPA_HEADS",
    "CollapseAlarm",
    "EMAModule",
    "JointStep",
    "code_diagnostics",
    "cosine_momentum",
    "covariance_term",
    "effective_rank",
    "encode_clean",
    "explained_variance",
    "integrate_flow",
    "joint_loss",
    "predicted_codes",
    "std_summary",
    "variance_term",
    "vicreg_loss",
]

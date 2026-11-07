"""VICReg variance and covariance terms for JEPA joint training (SPEC 8, step 6).

Both terms take one level's codes as ``[N, d]`` (N = B * l_k vectors of that
level) and are differentiable. The combined per-level loss is

    L_vic = lambda_var * var + lambda_cov * cov

with the SPEC defaults ``lambda_var = 1.0`` and ``lambda_cov = 0.04``.
"""

from __future__ import annotations

import torch

GAMMA = 1.0
LAMBDA_VAR = 1.0
LAMBDA_COV = 0.04
EPS = 1e-8


def _check_input(x: torch.Tensor) -> None:
    if x.ndim != 2:
        raise ValueError(
            f"expected a 2-D [N, d] tensor of level codes, got {tuple(x.shape)}"
        )


def variance_term(
    x: torch.Tensor, gamma: float = GAMMA, eps: float = EPS
) -> torch.Tensor:
    """Variance term ``mean_d relu(gamma - std_d)`` (SPEC 8, step 6).

    Args:
        x: level codes, shape ``[N, d]``; N vectors of one level.
        gamma: target standard deviation per dimension (default 1).
        eps: numerical stabilizer under the square root, so a collapsed
            dimension still has a finite gradient.

    Returns:
        Scalar tensor; 0 when every dimension has std >= gamma, up to gamma
        when dimensions are collapsed.
    """
    _check_input(x)
    std = torch.sqrt(x.var(dim=0, unbiased=False) + eps)
    return torch.relu(gamma - std).mean()


def covariance_term(x: torch.Tensor) -> torch.Tensor:
    """Covariance term ``sum_{i != j} Cov_ij^2 / d`` (SPEC 8, step 6).

    Args:
        x: level codes, shape ``[N, d]``.

    Returns:
        Scalar tensor; the mean squared off-diagonal covariance scaled by ``d``.
    """
    _check_input(x)
    n, d = x.shape
    centered = x - x.mean(dim=0, keepdim=True)
    covariance = centered.t() @ centered / max(n - 1, 1)
    off_diagonal = covariance - torch.diag_embed(torch.diagonal(covariance))
    return off_diagonal.pow(2).sum() / d


def vicreg_loss(
    x: torch.Tensor,
    gamma: float = GAMMA,
    lambda_var: float = LAMBDA_VAR,
    lambda_cov: float = LAMBDA_COV,
    eps: float = EPS,
) -> torch.Tensor:
    """Combined per-level VICReg loss ``lambda_var * var + lambda_cov * cov``.

    Args:
        x: level codes, shape ``[N, d]``.
        gamma: variance-term target std.
        lambda_var: variance-term weight (SPEC default 1.0).
        lambda_cov: covariance-term weight (SPEC default 0.04).
        eps: numerical stabilizer for the standard deviation.

    Returns:
        Scalar tensor.
    """
    return lambda_var * variance_term(x, gamma=gamma, eps=eps) + lambda_cov * covariance_term(x)

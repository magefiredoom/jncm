"""
Continuous noisy channel quantizer with LayerNorm and Gaussian channel noise (SPEC 4.3).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class NoisyChannelQuantizer(nn.Module):
    """
    Continuous code quantizer bounded by power constraint and channel noise.

    u' = Linear_k(u) [B, l_k, d_k]
    c_k = LayerNorm(u', elementwise_affine=False) (clean code, unit power per dim)
    code_k = c_k + sigma_k * eps, eps ~ N(0, I) (transmission code, train & eval)
    """

    def __init__(
        self,
        d_in: int,
        d_k: int = 8,
        sigma_k: float = 0.4,
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_k = d_k
        self.sigma_k = sigma_k

        self.proj = nn.Linear(d_in, d_k)
        # Power constraint: unit variance across code dimension, eps=1e-8 ensures exact unit power
        self.norm = nn.LayerNorm(d_k, eps=1e-8, elementwise_affine=False)

    def forward(
        self,
        u: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process residual pooled representations through the noisy channel.

        Args:
            u: Input residual representation [B, l_k, d_in]
            generator: Optional torch.Generator for deterministic noise sampling

        Returns:
            targets: Clean code c_k of shape [B, l_k, d_k]
            codes: Noisy transmitted code c_k + sigma_k * eps [B, l_k, d_k]
            loss: Zero tensor (quant_loss is 0 for continuous variant)
        """
        u_prime = self.proj(u)  # [B, l_k, d_k]
        clean_code = self.norm(u_prime)  # [B, l_k, d_k]

        # Channel noise applied in both train and eval
        if generator is not None:
            eps = torch.randn(
                clean_code.shape,
                generator=generator,
                device=clean_code.device,
                dtype=clean_code.dtype,
            )
        else:
            eps = torch.randn_like(clean_code)

        noisy_code = clean_code + self.sigma_k * eps
        quant_loss = torch.tensor(0.0, device=u.device, dtype=u.dtype)

        return clean_code, noisy_code, quant_loss

    def compute_diagnostics(self, c_k: torch.Tensor) -> dict[str, float]:
        """
        Compute batch diagnostics: std over batch/time per dimension, effective rank,
        and nominal bits per position (SPEC 4.5).
        """
        # Reshape to [N, d_k]
        flat = c_k.detach().reshape(-1, self.d_k).float()
        dim_stds = flat.std(dim=0)
        min_std = float(dim_stds.min().item())
        mean_std = float(dim_stds.mean().item())

        # Effective rank: exp(spectral entropy of normalized covariance)
        cov = torch.cov(flat.t())
        evals = torch.linalg.eigvalsh(cov)
        evals = torch.clamp(evals, min=1e-8)
        norm_evals = evals / evals.sum()
        spectral_entropy = -torch.sum(norm_evals * torch.log(norm_evals)).item()
        eff_rank = float(math.exp(spectral_entropy))

        snr = 1.0 / (self.sigma_k**2)
        nominal_bits = float((self.d_k / 2.0) * math.log2(1.0 + snr))

        return {
            "std_min": min_std,
            "std_mean": mean_std,
            "eff_rank": eff_rank,
            "nominal_bits": nominal_bits,
        }
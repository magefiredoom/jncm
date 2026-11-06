"""
Discrete vector quantizer with straight-through estimator, EMA codebook updates,
and dead-code reinitialization (SPEC 4.2).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    Vector Quantizer with straight-through gradient and EMA updates.

    Projects input u [B, l_k, d] -> u' [B, l_k, d_code], quantizes against codebook
    E_k [V_k, d_code], computes commitment loss beta * MSE(u', sg(e)), and maintains
    codebook weights via exponential moving average (no gradients on codebook).
    """

    def __init__(
        self,
        d_in: int,
        d_code: int,
        V_k: int,
        beta: float = 0.25,
        ema_decay: float = 0.99,
        dead_code_threshold: float = 1e-4,
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_code = d_code
        self.V_k = V_k
        self.beta = beta
        self.ema_decay = ema_decay
        self.dead_code_threshold = dead_code_threshold

        self.proj = nn.Linear(d_in, d_code)

        # Codebook weights (not trained by autograd)
        self.embedding = nn.Embedding(V_k, d_code)
        # Initialize randomly from standard normal
        nn.init.normal_(self.embedding.weight, mean=0.0, std=1.0 / math.sqrt(d_code))
        self.embedding.weight.requires_grad = False

        # EMA accumulators
        self.register_buffer("cluster_size", torch.ones(V_k))
        self.register_buffer("embed_avg", self.embedding.weight.clone())
        self._last_flat_u: torch.Tensor | None = None

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        """Retrieve code vectors from discrete indices [B, l_k] -> [B, l_k, d_code]."""
        return self.embedding(indices)

    def forward(
        self,
        u: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize input u.

        Args:
            u: Tensor of shape [B, l_k, d_in]

        Returns:
            targets: Discrete code indices of shape [B, l_k] (torch.long)
            codes: Code vectors with straight-through estimator [B, l_k, d_code]
            loss: Scalar commitment loss beta * MSE(u', sg(e))
        """
        # SPEC 4.2: all VQ arithmetic runs in fp32 even under bf16 autocast.
        # bf16 breaks the EMA accumulator dtype match and degrades argmin.
        with torch.autocast(device_type=u.device.type, enabled=False):
            u = u.float()
            u_prime = self.proj(u)  # [B, l_k, d_code]
            b, l_k, d_c = u_prime.shape
            flat_u = u_prime.reshape(-1, d_c)

            if self.training:
                self._last_flat_u = flat_u.detach()

            # Distance ||flat_u - E||^2 = ||flat_u||^2 + ||E||^2 - 2 flat_u E^T
            embed_w = self.embedding.weight  # [V_k, d_code]
            dist = (
                torch.sum(flat_u**2, dim=1, keepdim=True)
                + torch.sum(embed_w**2, dim=1).unsqueeze(0)
                - 2.0 * torch.matmul(flat_u, embed_w.t())
            )

            indices = torch.argmin(dist, dim=1)  # [B * l_k]
            quantized = self.embedding(indices).view(b, l_k, d_c)
            targets = indices.view(b, l_k)

            # EMA codebook update during training
            if self.training:
                with torch.no_grad():
                    counts = torch.bincount(indices, minlength=self.V_k).float()
                    sums = torch.zeros(self.V_k, d_c, device=u.device, dtype=u.dtype)
                    sums.index_add_(0, indices, flat_u)

                    self.cluster_size.mul_(self.ema_decay).add_(
                        counts, alpha=1.0 - self.ema_decay
                    )
                    self.embed_avg.mul_(self.ema_decay).add_(
                        sums, alpha=1.0 - self.ema_decay
                    )

                    # Laplace smoothing to prevent division by zero
                    n = torch.sum(self.cluster_size)
                    smoothed_size = (
                        (self.cluster_size + 1e-5) / (n + self.V_k * 1e-5)
                    ) * n
                    new_w = self.embed_avg / smoothed_size.unsqueeze(1)
                    self.embedding.weight.copy_(new_w)

            # Commitment loss
            loss = self.beta * F.mse_loss(u_prime, quantized.detach())

            # Straight-through estimator
            codes = u_prime + (quantized - u_prime).detach()

        return targets, codes, loss

    def reinit_dead_codes(
        self,
        u_flat: torch.Tensor | None = None,
        threshold: float | None = None,
    ) -> int:
        """
        Reinitialize codes whose EMA usage is below threshold of mean usage (SPEC 4.2).

        Args:
            u_flat: Optional sample vectors of shape [N, d_code] (defaults to cached batch)
            threshold: Optional threshold fraction of mean usage (default: self.dead_code_threshold)

        Returns:
            Number of dead codes reinitialized.
        """
        if u_flat is None:
            u_flat = self._last_flat_u
        if u_flat is None or len(u_flat) == 0:
            return 0

        thresh = self.dead_code_threshold if threshold is None else threshold
        with torch.no_grad():
            mean_usage = self.cluster_size.mean()
            dead_mask = self.cluster_size < (thresh * mean_usage)
            n_dead = int(dead_mask.sum().item())
            if n_dead > 0:
                rand_idx = torch.randint(0, len(u_flat), (n_dead,), device=u_flat.device)
                replacement = u_flat[rand_idx].detach()

                self.embedding.weight.data[dead_mask] = replacement
                self.embed_avg.data[dead_mask] = replacement * mean_usage
                self.cluster_size.data[dead_mask] = mean_usage
            return n_dead

    def compute_diagnostics(self, targets: torch.Tensor) -> dict[str, float]:
        """Compute usage fraction and code perplexity over a batch of code indices."""
        flat = targets.view(-1)
        counts = torch.bincount(flat, minlength=self.V_k).float()
        used = (counts > 0).float().sum().item()
        usage_frac = used / float(self.V_k)

        probs = counts / counts.sum().clamp(min=1.0)
        nonzero_p = probs[probs > 0]
        entropy = -(nonzero_p * torch.log(nonzero_p)).sum().item()
        perplexity = float(math.exp(entropy))

        return {
            "usage_frac": usage_frac,
            "perplexity": perplexity,
        }
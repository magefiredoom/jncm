"""Generator output heads and losses (SPEC 5.3 and 5.5).

Each head maps the per-level backbone hidden states [B, l_k, H] to code
predictions: codebook logits (``ce``), a clean continuous code (``regress``),
or a flow-matching velocity field (``flow``). ``loss`` returns the sum of the
per-level losses together with the list of per-level losses; each per-level
value is a mean over that level's positions, so ``weighted_total`` reproduces
SPEC 5.5 exactly.

Sampling interfaces: ``sample_level(h, k, generator=...)`` returns one level's
codes ([B, l_k] integer indices for ``ce``, [B, l_k, d_k] vectors otherwise), and
``sample_codes(head, hidden, generator=...)`` loops over levels.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from jncm.config import GeneratorConfig


def _randn(
    shape: Sequence[int],
    generator: torch.Generator | None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Standard normal noise, reproducible when a generator is given."""
    if generator is not None:
        return torch.randn(shape, generator=generator, device=device, dtype=dtype)
    return torch.randn(shape, device=device, dtype=dtype)


def _rand(
    shape: Sequence[int],
    generator: torch.Generator | None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Uniform [0, 1) noise, reproducible when a generator is given."""
    if generator is not None:
        return torch.rand(shape, generator=generator, device=device, dtype=dtype)
    return torch.rand(shape, device=device, dtype=dtype)


def _check_level(
    h: torch.Tensor,
    k: int,
    H: int,
    target: torch.Tensor | None = None,
    code_dim: int | None = None,
) -> None:
    """Validate one level's hidden state (and optional target) shape."""
    if h.dim() != 3 or h.shape[-1] != H:
        raise ValueError(
            f"level {k}: hidden must be [B, l_k, {H}], got {tuple(h.shape)}"
        )
    if target is None:
        return
    if tuple(target.shape[:2]) != tuple(h.shape[:2]):
        raise ValueError(
            f"level {k}: target must be [B={h.shape[0]}, l_k={h.shape[1]}, ...], "
            f"got {tuple(target.shape)}"
        )
    if code_dim is not None and target.shape[-1] != code_dim:
        raise ValueError(
            f"level {k}: target code width must be {code_dim}, "
            f"got {target.shape[-1]}"
        )


def _check_levels(hidden: Sequence[torch.Tensor], targets: Sequence[torch.Tensor], K: int) -> None:
    """Validate the level count before touching individual levels."""
    if len(hidden) != K:
        raise ValueError(f"expected {K} hidden tensors, got {len(hidden)}")
    if len(targets) != K:
        raise ValueError(f"expected {K} target tensors, got {len(targets)}")


class CEHead(nn.Module):
    """Per-level Linear(H -> V_k) with cross-entropy on code indices (SPEC 5.3).

    Sample: categorical draw at temperature 1 from the per-position logits.
    """

    def __init__(self, H: int, V_k: int, K: int) -> None:
        super().__init__()
        if H < 1 or V_k < 1 or K < 1:
            raise ValueError(f"H, V_k, K must be positive, got {H}, {V_k}, {K}")
        self.H = H
        self.V_k = V_k
        self.K = K
        self.code_dim = V_k
        self.heads = nn.ModuleList([nn.Linear(H, V_k) for _ in range(K)])

    def loss(
        self,
        hidden: Sequence[torch.Tensor],
        targets: Sequence[torch.Tensor],
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Cross-entropy per level on integer code indices.

        Args:
            hidden: One tensor [B, l_k, H] per level.
            targets: One LongTensor [B, l_k] of codebook indices per level.
            generator: Unused (CE training is deterministic).

        Returns:
            (sum of per-level losses, list of per-level losses), each loss a
            mean over the level's positions.
        """
        _check_levels(hidden, targets, self.K)
        per_level: list[torch.Tensor] = []
        for k, (h, t) in enumerate(zip(hidden, targets), start=1):
            _check_level(h, k, self.H)
            if t.dtype not in (torch.long, torch.int64):
                raise ValueError(
                    f"level {k}: ce targets must be integer indices, got {t.dtype}"
                )
            if tuple(t.shape) != (h.shape[0], h.shape[1]):
                raise ValueError(
                    f"level {k}: ce targets must be [B, l_k]="
                    f"{(h.shape[0], h.shape[1])}, got {tuple(t.shape)}"
                )
            logits = self.heads[k - 1](h)  # [B, l_k, V_k]
            per_level.append(
                F.cross_entropy(logits.reshape(-1, self.V_k), t.reshape(-1))
            )
        return torch.stack(per_level).sum(), per_level

    def sample_level(
        self,
        h: torch.Tensor,
        k: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Draw one code index per position: [B, l_k, H] -> LongTensor [B, l_k]."""
        _check_level(h, k, self.H)
        logits = self.heads[k - 1](h)
        probs = F.softmax(logits.float(), dim=-1)  # temperature 1
        flat = probs.reshape(-1, self.V_k)
        idx = torch.multinomial(flat, 1, generator=generator)
        return idx.reshape(h.shape[0], h.shape[1])


class RegressHead(nn.Module):
    """Per-level Linear(H -> d_k) regression of the clean continuous code.

    Loss: MSE against the clean code c_k, or ``1 - cosine`` when ``cosine``.
    Sampling optionally renormalizes the prediction (``renorm``: LayerNorm with
    no affine, because a regression mean has shrunken norm) and adds fresh
    channel noise ``sigma_k * eps'`` (``add_channel_noise``), matching the
    noisy codes the decoder was trained on.
    """

    def __init__(
        self,
        H: int,
        d_k: int,
        K: int,
        sigma_k: float = 0.4,
        renorm: bool = True,
        add_channel_noise: bool = True,
        cosine: bool = False,
    ) -> None:
        super().__init__()
        if H < 1 or d_k < 1 or K < 1:
            raise ValueError(f"H, d_k, K must be positive, got {H}, {d_k}, {K}")
        if sigma_k < 0.0:
            raise ValueError(f"sigma_k must be >= 0, got {sigma_k}")
        self.H = H
        self.d_k = d_k
        self.K = K
        self.sigma_k = sigma_k
        self.renorm = renorm
        self.add_channel_noise = add_channel_noise
        self.cosine = cosine
        self.code_dim = d_k
        self.heads = nn.ModuleList([nn.Linear(H, d_k) for _ in range(K)])
        self.norm = nn.LayerNorm(d_k, elementwise_affine=False)

    def loss(
        self,
        hidden: Sequence[torch.Tensor],
        targets: Sequence[torch.Tensor],
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Regression loss per level against clean codes [B, l_k, d_k].

        Args:
            hidden: One tensor [B, l_k, H] per level.
            targets: One tensor [B, l_k, d_k] of clean codes per level.
            generator: Unused (regression training is deterministic).

        Returns:
            (sum of per-level losses, list of per-level losses).
        """
        _check_levels(hidden, targets, self.K)
        per_level: list[torch.Tensor] = []
        for k, (h, t) in enumerate(zip(hidden, targets), start=1):
            _check_level(h, k, self.H, target=t, code_dim=self.d_k)
            pred = self.heads[k - 1](h)  # [B, l_k, d_k]
            if self.cosine:
                per_level.append((1.0 - F.cosine_similarity(pred, t, dim=-1)).mean())
            else:
                per_level.append(F.mse_loss(pred, t))
        return torch.stack(per_level).sum(), per_level

    def sample_level(
        self,
        h: torch.Tensor,
        k: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Predict one level's codes: [B, l_k, H] -> [B, l_k, d_k]."""
        _check_level(h, k, self.H)
        code = self.heads[k - 1](h)
        if self.renorm:
            code = self.norm(code)
        if self.add_channel_noise:
            code = code + self.sigma_k * _randn(
                code.shape, generator, code.device, code.dtype
            )
        return code


class SinusoidalTimeEmbedding(nn.Module):
    """Fixed sinusoidal time features of width ``dim`` (SPEC 5.3).

    64 log-spaced frequencies from 1 to 1e-4 cycles for t in [0, 1], evaluated
    as sin(2*pi*t*f) and cos(2*pi*t*f) and concatenated.
    """

    def __init__(self, dim: int = 128, min_freq: float = 1e-4) -> None:
        super().__init__()
        if dim < 2 or dim % 2 != 0:
            raise ValueError(f"time embedding dim must be even and >= 2, got {dim}")
        if not 0.0 < min_freq <= 1.0:
            raise ValueError(f"min_freq must be in (0, 1], got {min_freq}")
        self.dim = dim
        freqs = torch.exp(
            torch.linspace(math.log(1.0), math.log(min_freq), dim // 2)
        )
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed times t [N] -> [N, dim]."""
        angles = 2.0 * math.pi * t.unsqueeze(-1) * self.freqs.to(t.dtype)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class FlowMLP(nn.Module):
    """Conditional velocity network for one level (SPEC 5.3).

    ``Linear(d_k + time_dim + H -> 512)``, SiLU, ``n_blocks`` residual blocks
    ``x + SiLU(Linear(x))`` of width 512, ``Linear(512 -> d_k)``.
    """

    def __init__(
        self,
        d_k: int,
        H: int,
        hidden: int = 512,
        n_blocks: int = 3,
        time_dim: int = 128,
    ) -> None:
        super().__init__()
        if n_blocks < 1:
            raise ValueError(f"n_blocks must be >= 1, got {n_blocks}")
        self.d_k = d_k
        self.H = H
        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        self.input = nn.Linear(d_k + time_dim + H, hidden)
        self.blocks = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_blocks)])
        self.act = nn.SiLU()
        self.output = nn.Linear(hidden, d_k)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        h: torch.Tensor,
    ) -> torch.Tensor:
        """Velocity prediction: x_t [N, d_k], t [N], h [N, H] -> [N, d_k]."""
        if x_t.dim() != 2 or x_t.shape[-1] != self.d_k:
            raise ValueError(f"x_t must be [N, {self.d_k}], got {tuple(x_t.shape)}")
        if h.dim() != 2 or h.shape[-1] != self.H or h.shape[0] != x_t.shape[0]:
            raise ValueError(f"h must be [N={x_t.shape[0]}, {self.H}], got {tuple(h.shape)}")
        z = self.act(self.input(torch.cat([x_t, self.time_emb(t), h], dim=-1)))
        for lin in self.blocks:
            z = z + self.act(lin(z))
        return self.output(z)


class FlowHead(nn.Module):
    """Per-level conditional flow matching on the clean continuous code (SPEC 5.3).

    Training: ``n_rep`` (t, eps) pairs per valid position, ``x_t = (1-t)*eps +
    t*c_k``, target velocity ``c_k - eps``, MSE. Sampling: Euler integration
    from t = 0 (start ``eps``) to t = 1 in ``flow_steps`` steps.
    """

    def __init__(
        self,
        H: int,
        d_k: int,
        K: int,
        sigma_k: float = 0.4,
        n_rep: int = 4,
        flow_steps: int = 8,
        add_channel_noise: bool = True,
        hidden: int = 512,
        n_blocks: int = 3,
        time_dim: int = 128,
    ) -> None:
        super().__init__()
        if H < 1 or d_k < 1 or K < 1:
            raise ValueError(f"H, d_k, K must be positive, got {H}, {d_k}, {K}")
        if n_rep < 1:
            raise ValueError(f"n_rep must be >= 1, got {n_rep}")
        if flow_steps < 1:
            raise ValueError(f"flow_steps must be >= 1, got {flow_steps}")
        if sigma_k < 0.0:
            raise ValueError(f"sigma_k must be >= 0, got {sigma_k}")
        self.H = H
        self.d_k = d_k
        self.K = K
        self.sigma_k = sigma_k
        self.n_rep = n_rep
        self.flow_steps = flow_steps
        self.add_channel_noise = add_channel_noise
        self.code_dim = d_k
        self.nets = nn.ModuleList(
            [
                FlowMLP(d_k, H, hidden=hidden, n_blocks=n_blocks, time_dim=time_dim)
                for _ in range(K)
            ]
        )

    def loss(
        self,
        hidden: Sequence[torch.Tensor],
        targets: Sequence[torch.Tensor],
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Flow-matching MSE per level against clean codes [B, l_k, d_k].

        Args:
            hidden: One tensor [B, l_k, H] per level.
            targets: One tensor [B, l_k, d_k] of clean codes per level.
            generator: Optional generator for the (t, eps) draws.

        Returns:
            (sum of per-level losses, list of per-level losses).
        """
        _check_levels(hidden, targets, self.K)
        per_level: list[torch.Tensor] = []
        for k, (h, tgt) in enumerate(zip(hidden, targets), start=1):
            _check_level(h, k, self.H, target=tgt, code_dim=self.d_k)
            c = tgt
            # n_rep independent (t, eps) pairs per valid position.
            h_rep = h.reshape(-1, self.H).repeat_interleave(self.n_rep, dim=0)
            c_rep = c.reshape(-1, self.d_k).repeat_interleave(self.n_rep, dim=0)
            n = c_rep.shape[0]
            t = _rand((n, 1), generator, c.device, c.dtype)
            eps = _randn((n, self.d_k), generator, c.device, c.dtype)
            x_t = (1.0 - t) * eps + t * c_rep
            v = c_rep - eps
            pred = self.nets[k - 1](x_t, t.squeeze(-1), h_rep)
            per_level.append(F.mse_loss(pred, v))
        return torch.stack(per_level).sum(), per_level

    def sample_level(
        self,
        h: torch.Tensor,
        k: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Euler-integrate one level's flow: [B, l_k, H] -> [B, l_k, d_k]."""
        _check_level(h, k, self.H)
        b, length, _ = h.shape
        h_flat = h.reshape(-1, self.H)
        n = b * length
        x = _randn((n, self.d_k), generator, h.device, h.dtype)
        net = self.nets[k - 1]
        dt = 1.0 / self.flow_steps
        for step in range(self.flow_steps):
            t = torch.full((n,), step * dt, device=h.device, dtype=h.dtype)
            x = x + dt * net(x, t, h_flat)
        code = x.reshape(b, length, self.d_k)
        if self.add_channel_noise:
            code = code + self.sigma_k * _randn(
                code.shape, generator, code.device, code.dtype
            )
        return code


LevelHead = CEHead | RegressHead | FlowHead


def build_head(
    cfg: GeneratorConfig,
    H: int,
    K: int,
    *,
    V_k: int | None = None,
    d_k: int | None = None,
    sigma_k: float | None = None,
) -> LevelHead:
    """Construct the configured head (SPEC 5.3).

    Args:
        cfg: Generator config; ``head`` selects ``ce``/``regress``/``flow``.
        H: Backbone width.
        K: Number of levels.
        V_k: Codebook size (required by ``ce``).
        d_k: Continuous code width (required by ``regress`` and ``flow``).
        sigma_k: Channel noise std (required by ``regress`` and ``flow``).

    Returns:
        The head module.
    """
    if cfg.head == "ce":
        if V_k is None:
            raise ValueError("head 'ce' requires V_k (tokenizer.V_k)")
        return CEHead(H=H, V_k=V_k, K=K)
    if cfg.head in ("regress", "flow"):
        if d_k is None or sigma_k is None:
            raise ValueError(
                f"head '{cfg.head}' requires d_k and sigma_k (tokenizer.d_k, tokenizer.sigma_k)"
            )
        if cfg.head == "regress":
            return RegressHead(
                H=H,
                d_k=d_k,
                K=K,
                sigma_k=sigma_k,
                renorm=cfg.renorm_pred,
                add_channel_noise=cfg.add_channel_noise,
                cosine=cfg.regress_cosine,
            )
        return FlowHead(
            H=H,
            d_k=d_k,
            K=K,
            sigma_k=sigma_k,
            n_rep=cfg.n_rep,
            flow_steps=cfg.flow_steps,
            add_channel_noise=cfg.add_channel_noise,
        )
    raise ValueError(f"Unknown head: {cfg.head} (expected 'ce', 'regress' or 'flow')")


def level_weights(lengths: Sequence[int], alpha: float = 0.0) -> list[float]:
    """Per-level loss weights ``w_k = l_k^(-alpha) / sum_j l_j^(1-alpha)`` (SPEC 5.5)."""
    if alpha < 0.0:
        raise ValueError(f"alpha must be >= 0, got {alpha}")
    if not lengths:
        raise ValueError("lengths must be non-empty")
    if any(length < 1 for length in lengths):
        raise ValueError(f"all level lengths must be >= 1, got {list(lengths)}")
    denom = sum(length ** (1.0 - alpha) for length in lengths)
    return [length ** (-alpha) / denom for length in lengths]


def weighted_total(
    per_level_losses: Sequence[torch.Tensor],
    lengths: Sequence[int],
    alpha: float = 0.0,
) -> torch.Tensor:
    """``L_gen = sum_k w_k * sum_i loss_{k,i}`` from per-position-mean losses.

    Args:
        per_level_losses: One scalar per level, each a mean over its positions.
        lengths: Level schedule l_k.
        alpha: SPEC 5.5 exponent (0 = equal weight per position).

    Returns:
        Scalar tensor.
    """
    if len(per_level_losses) != len(lengths):
        raise ValueError(
            f"got {len(per_level_losses)} losses for {len(lengths)} levels"
        )
    weights = level_weights(lengths, alpha)
    contributions = [
        weight * length * loss
        for weight, length, loss in zip(weights, lengths, per_level_losses)
    ]
    return torch.stack(contributions).sum()


def sample_codes(
    head: LevelHead,
    hidden: Sequence[torch.Tensor],
    generator: torch.Generator | None = None,
) -> list[torch.Tensor]:
    """Sample one code tensor per level with ``sample_level``."""
    if len(hidden) != head.K:
        raise ValueError(f"expected {head.K} hidden tensors, got {len(hidden)}")
    return [
        head.sample_level(h, k, generator=generator)
        for k, h in enumerate(hidden, start=1)
    ]

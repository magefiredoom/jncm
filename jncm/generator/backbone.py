"""Level-block generator backbone (SPEC 5.1 and 5.2).

The backbone embeds each level from the tokenizer reconstruction prefix, runs a
bidirectional pre-LN transformer within each level only, and returns one hidden
state tensor per level. Level k depends only on codes of levels < k.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from jncm.config import GeneratorConfig
from jncm.levels import down, get_level_schedule

if TYPE_CHECKING:
    from jncm.tokenizer import MultiscaleTokenizer


class SinusoidalPositionProjection(nn.Module):
    """Fixed log-spaced sin/cos features projected to H (SPEC 5.1).

    Positions are p_i = (i + 0.5) / length. The 64 frequencies are geometric
    from 1 down to 1e-4 cycles per normalized position; sin and cos are
    concatenated (128 features) and linearly projected to H.
    """

    def __init__(self, H: int, n_freq: int = 64, min_freq: float = 1e-4) -> None:
        super().__init__()
        if n_freq < 1:
            raise ValueError(f"n_freq must be >= 1, got {n_freq}")
        if not 0.0 < min_freq <= 1.0:
            raise ValueError(f"min_freq must be in (0, 1], got {min_freq}")
        self.n_freq = n_freq
        freqs = torch.exp(
            torch.linspace(math.log(1.0), math.log(min_freq), n_freq)
        )
        self.register_buffer("freqs", freqs, persistent=False)
        self.proj = nn.Linear(2 * n_freq, H)

    def _features(self, length: int) -> torch.Tensor:
        """Return fixed features [1, length, 2*n_freq]."""
        device = self.freqs.device
        p = (torch.arange(length, device=device, dtype=torch.float32) + 0.5) / length
        angles = (
            2.0
            * math.pi
            * p.unsqueeze(1)
            * self.freqs.to(dtype=torch.float32).unsqueeze(0)
        )
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1).unsqueeze(0)

    def forward(self, lengths: Sequence[int]) -> list[torch.Tensor]:
        """Project one position feature tensor [1, l_k, H] per requested length."""
        feats = [self._features(length) for length in lengths]
        # Stacking keeps one linear projection call while preserving per-level lengths.
        flat = torch.cat(feats, dim=1)
        projected = self.proj(flat)
        outputs: list[torch.Tensor] = []
        start = 0
        for length in lengths:
            outputs.append(projected[:, start : start + length, :])
            start += length
        return outputs


class LevelInputEmbedding(nn.Module):
    """Embed downsampled reconstruction prefixes for levels 1..K (SPEC 5.1)."""

    def __init__(self, d: int, H: int, K: int, n_freq: int = 64) -> None:
        super().__init__()
        self.d = d
        self.H = H
        self.K = K
        self.input_proj = nn.Linear(d, H)
        self.level_emb = nn.Embedding(K + 1, H)
        self.pos_proj = SinusoidalPositionProjection(H, n_freq=n_freq)

    def forward(
        self,
        level_reps: Sequence[torch.Tensor],
        levels: Sequence[int] | None = None,
    ) -> list[torch.Tensor]:
        """Add input projection, sinusoidal positions, and level embedding.

        Args:
            level_reps: One tensor [B, l_k, d] per level.
            levels: 1-indexed level numbers; defaults to 1..len(level_reps).

        Returns:
            One tensor [B, l_k, H] per level.
        """
        if levels is None:
            levels = list(range(1, len(level_reps) + 1))
        if len(levels) != len(level_reps):
            raise ValueError("levels and level_reps must have the same length")
        positions = self.pos_proj([rep.shape[1] for rep in level_reps])
        outputs: list[torch.Tensor] = []
        for rep, pos, k in zip(level_reps, positions, levels):
            if not 1 <= k <= self.K:
                raise ValueError(f"level must be in 1..{self.K}, got {k}")
            if rep.shape[-1] != self.d:
                raise ValueError(
                    f"level {k} representation width must be d={self.d}, "
                    f"got {rep.shape[-1]}"
                )
            outputs.append(self.input_proj(rep) + pos + self.level_emb.weight[k])
        return outputs


class LevelTransformerBlock(nn.Module):
    """Pre-LN transformer block with optional key-padding mask (SPEC 5.2)."""

    def __init__(
        self,
        H: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if H % n_heads != 0:
            raise ValueError(f"H={H} must be divisible by n_heads={n_heads}")
        self.H = H
        self.n_heads = n_heads
        self.head_dim = H // n_heads
        self.ln1 = nn.LayerNorm(H)
        self.ln2 = nn.LayerNorm(H)
        self.qkv = nn.Linear(H, 3 * H)
        self.out_proj = nn.Linear(H, H)
        hidden = int(H * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(H, hidden),
            nn.GELU(),
            nn.Linear(hidden, H),
        )
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the block on [B, T, H] with optional valid-key mask [B, T]."""
        b, t, _ = x.shape
        normed = self.ln1(x)
        qkv = self.qkv(normed).reshape(b, t, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attn_mask = None
        if key_mask is not None:
            attn_mask = key_mask[:, None, None, :]
        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        attn = attn.transpose(1, 2).reshape(b, t, self.H)
        x = x + self.drop1(self.out_proj(attn))
        x = x + self.drop2(self.mlp(self.ln2(x)))
        return x


class LevelBlockTransformer(nn.Module):
    """Stack of pre-LN blocks with block-diagonal attention across levels."""

    def __init__(
        self,
        H: int,
        n_layers: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        self.H = H
        self.blocks = nn.ModuleList(
            [
                LevelTransformerBlock(
                    H,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        key_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run all blocks on a packed [B*K, L, H] tensor."""
        for block in self.blocks:
            x = block(x, key_mask=key_mask)
        return x

    def forward_levels(
        self,
        level_inputs: Sequence[torch.Tensor],
        lengths: Sequence[int] | None = None,
    ) -> list[torch.Tensor]:
        """Run all levels in one packed pass and gather valid positions.

        Args:
            level_inputs: One tensor [B, l_k, H] per level, possibly left-padded
                to a common length.
            lengths: Valid length of each level; defaults to each tensor's width.

        Returns:
            One hidden tensor [B, lengths[k-1], H] per level.
        """
        if not level_inputs:
            return []
        if lengths is None:
            lengths = [tensor.shape[1] for tensor in level_inputs]
        if len(lengths) != len(level_inputs):
            raise ValueError("lengths and level_inputs must have the same length")

        b, _, h = level_inputs[0].shape
        if h != self.H:
            raise ValueError(f"level inputs must have width H={self.H}, got {h}")
        max_len = max(tensor.shape[1] for tensor in level_inputs)
        padded: list[torch.Tensor] = []
        for idx, (tensor, length) in enumerate(zip(level_inputs, lengths)):
            if tensor.shape[0] != b or tensor.shape[2] != h:
                raise ValueError("all level inputs must have shape [B, l_k, H]")
            if not 1 <= length <= tensor.shape[1]:
                raise ValueError(
                    f"length for level {idx + 1} must be in 1..{tensor.shape[1]}, "
                    f"got {length}"
                )
            if tensor.shape[1] < max_len:
                tensor = F.pad(tensor, (0, 0, 0, max_len - tensor.shape[1]))
            padded.append(tensor)

        k = len(padded)
        packed = torch.stack(padded, dim=1).reshape(b * k, max_len, h)
        valid = torch.arange(max_len, device=packed.device).unsqueeze(0) < (
            torch.as_tensor(lengths, device=packed.device).unsqueeze(1)
        )
        key_mask = valid.unsqueeze(0).expand(b, -1, -1).reshape(b * k, max_len)
        hidden = self.forward(packed, key_mask=key_mask)
        hidden = hidden.reshape(b, k, max_len, h)
        return [hidden[:, idx, : lengths[idx], :] for idx in range(k)]


class GeneratorBackbone(nn.Module):
    """Generator backbone: level inputs built from codes plus level-block attention.

    Sampling uses exactly one ``backbone(x_k)`` call per level (TASKS T3.4).
    Training builds all level inputs with ``embed_levels`` and runs the packed
    path in one call with ``forward_levels``.
    """

    def __init__(
        self,
        cfg: GeneratorConfig,
        L: int,
        K: int,
        d: int,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.L = L
        self.K = K
        self.d = d
        self.H = cfg.H
        self.schedule = get_level_schedule(L, K)
        if len(self.schedule) != K:
            raise ValueError(f"schedule length must equal K={K}, got {self.schedule}")
        self.input_embed = LevelInputEmbedding(d, cfg.H, K)
        self.trunk = LevelBlockTransformer(
            cfg.H,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            mlp_ratio=cfg.mlp_ratio,
            dropout=cfg.dropout,
        )

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer: "MultiscaleTokenizer",
        cfg: GeneratorConfig,
    ) -> "GeneratorBackbone":
        """Build a backbone matching a tokenizer's width, length, and level count."""
        return cls(cfg=cfg, L=tokenizer.L, K=tokenizer.K, d=tokenizer.d)

    def forward(
        self,
        x: torch.Tensor,
        key_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run one level block [B, T, H] -> [B, T, H]."""
        return self.trunk(x, key_mask=key_mask)

    def forward_levels(
        self,
        level_inputs: Sequence[torch.Tensor],
        lengths: Sequence[int] | None = None,
    ) -> list[torch.Tensor]:
        """Run all levels packed with block-diagonal attention."""
        return self.trunk.forward_levels(level_inputs, lengths=lengths)

    def embed_level(
        self,
        tokenizer: "MultiscaleTokenizer",
        zhat_prev: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """Embed level k from the reconstruction prefix zhat_{k-1} [B, L, d]."""
        if not 1 <= k <= self.K:
            raise ValueError(f"level must be in 1..{self.K}, got {k}")
        if zhat_prev.shape[1:] != (self.L, self.d):
            raise ValueError(
                f"zhat_prev must have shape [B, {self.L}, {self.d}], "
                f"got {tuple(zhat_prev.shape)}"
            )
        s_k = down(zhat_prev, self.schedule[k - 1])
        return self.input_embed([s_k], levels=[k])[0]

    def pooled_reconstructions(
        self,
        tokenizer: "MultiscaleTokenizer",
        codes: Sequence[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Pooled prefix reconstructions ``s_k = down(zhat_{k-1}, l_k)``.

        The chain ``zhat`` is built only from the tokenizer's frozen
        contribution modules, so callers keep this part under ``no_grad``
        (T5.5a F11). Shapes: one ``[B, l_k, d]`` tensor per level.
        """
        required = self.K - 1
        if len(codes) < required:
            raise ValueError(
                f"need at least {required} chain codes for K={self.K}, got {len(codes)}"
            )
        batch = codes[0].shape[0]
        zhat = tokenizer.tok_emb.weight.new_zeros((batch, self.L, self.d))
        reps: list[torch.Tensor] = []
        for k in range(1, self.K + 1):
            reps.append(down(zhat, self.schedule[k - 1]))
            if k < self.K:
                zhat = zhat + tokenizer.contrib(k, codes[k - 1])
        return reps

    def embed_levels(
        self,
        tokenizer: "MultiscaleTokenizer",
        codes: Sequence[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Build all level inputs; level k uses only codes j < k (SPEC 5.1).

        The pooled cumulative reconstructions are tokenizer-only and run under
        ``no_grad``; ``input_embed`` is applied outside so its parameters train
        (T5.5a F11). Both trainers call this function.
        """
        with torch.no_grad():
            reps = self.pooled_reconstructions(tokenizer, codes)
        return self.input_embed(reps, levels=list(range(1, self.K + 1)))

    def forward_codes(
        self,
        tokenizer: "MultiscaleTokenizer",
        codes: Sequence[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Build level inputs from chain codes and run all levels in one pass."""
        return self.forward_levels(self.embed_levels(tokenizer, codes))

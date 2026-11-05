"""
Trunk embedding, reconstruction operators, decoder, and full MultiscaleTokenizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from jncm.config import TokenizerConfig
from jncm.levels import down, get_level_schedule, up
from jncm.tokenizer.noisy import NoisyChannelQuantizer
from jncm.tokenizer.vq import VectorQuantizer


@dataclass
class Encoded:
    """Tokenizer encoding output across levels (SPEC 4.1)."""

    targets: list[torch.Tensor]
    codes: list[torch.Tensor]
    zhat: list[torch.Tensor]
    quant_loss: torch.Tensor


class TransformerBlock(nn.Module):
    """Bidirectional Pre-LN Transformer Block."""

    def __init__(self, d: int, n_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.d = d
        self.n_heads = n_heads
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, int(d * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(d * mlp_ratio), d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN Self-Attention
        normed = self.ln1(x)
        b, t, d = normed.shape
        head_dim = d // self.n_heads
        qkv = normed.reshape(b, t, self.n_heads, head_dim).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(qkv, qkv, qkv, is_causal=False)
        attn_out = attn_out.transpose(1, 2).reshape(b, t, d)
        x = x + attn_out

        # Pre-LN MLP
        x = x + self.mlp(self.ln2(x))
        return x


class ReconstructionOperator(nn.Module):
    """
    R_k operator (SPEC 4.1): Linear(code_width -> d) -> upsample(l_k -> L)
    -> depthwise Conv1d(kernel 3, padding 1).
    """

    def __init__(self, code_dim: int, d: int, L: int) -> None:
        super().__init__()
        self.d = d
        self.L = L
        self.proj = nn.Linear(code_dim, d)
        self.conv = nn.Conv1d(d, d, kernel_size=3, padding=1, groups=d)

        # Initialize Conv1d close to identity filter [0, 1, 0] with zero bias
        nn.init.zeros_(self.conv.weight)
        if self.conv.weight.shape[-1] == 3:
            self.conv.weight.data[:, 0, 1] = 1.0
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, code: torch.Tensor) -> torch.Tensor:
        """Apply R_k to code tensor [B, l_k, code_dim] -> [B, L, d]."""
        h = self.proj(code)  # [B, l_k, d]
        h_up = up(h, self.L)  # [B, L, d]
        h_conv = self.conv(h_up.transpose(1, 2)).transpose(1, 2)
        return h_conv.contiguous()


def sample_cutoff(K: int, p_drop: float, generator: torch.Generator | None = None) -> int:
    """
    Sample nested dropout cutoff index (SPEC 4.4).

    With probability p_drop pick cut ~ Uniform{1..K-1}, else cut = K.
    """
    if p_drop <= 0.0 or K <= 1:
        return K
    r = torch.rand(1, generator=generator).item()
    if r < p_drop:
        return int(torch.randint(1, K, (1,), generator=generator).item())
    return K


class MultiscaleTokenizer(nn.Module):
    """
    Multiscale Residual Tokenizer (SPEC Section 4).
    Supports both 'vq' (discrete) and 'noisy' (continuous) quantizers.
    """

    def __init__(
        self,
        cfg: TokenizerConfig | None = None,
        L: int = 32,
        V: int = 64,
    ) -> None:
        super().__init__()
        if cfg is None:
            cfg = TokenizerConfig()
        self.cfg = cfg
        self.L = L
        self.V = V
        self.d = cfg.d
        self.K = cfg.K
        self.quantizer_type = cfg.quantizer
        self.schedule = get_level_schedule(L, self.K)

        # Trunk
        self.tok_emb = nn.Embedding(V, self.d)
        self.pos_emb = nn.Parameter(torch.randn(L, self.d) * 0.02)
        self.enc_blocks = nn.ModuleList(
            [
                TransformerBlock(self.d, n_heads=cfg.decoder_heads)
                for _ in range(cfg.encoder_blocks)
            ]
        )

        # Quantizers and Reconstruction operators per level
        self.quantizers = nn.ModuleList()
        self.reconstructors = nn.ModuleList()

        for k in range(self.K):
            if self.quantizer_type == "vq":
                q = VectorQuantizer(
                    d_in=self.d,
                    d_code=cfg.d_code,
                    V_k=cfg.V_k,
                    beta=cfg.beta,
                    ema_decay=cfg.ema_decay,
                    dead_code_threshold=cfg.dead_code_threshold,
                )
                r = ReconstructionOperator(code_dim=cfg.d_code, d=self.d, L=self.L)
            elif self.quantizer_type == "noisy":
                q = NoisyChannelQuantizer(
                    d_in=self.d,
                    d_k=cfg.d_k,
                    sigma_k=cfg.sigma_k,
                )
                r = ReconstructionOperator(code_dim=cfg.d_k, d=self.d, L=self.L)
            else:
                raise ValueError(f"Unknown quantizer: {self.quantizer_type}")
            self.quantizers.append(q)
            self.reconstructors.append(r)

        # Decoder
        self.dec_pos_emb = nn.Parameter(torch.randn(L, self.d) * 0.02)
        self.dec_blocks = nn.ModuleList(
            [
                TransformerBlock(self.d, n_heads=cfg.decoder_heads)
                for _ in range(cfg.decoder_blocks)
            ]
        )
        self.dec_norm = nn.LayerNorm(self.d)
        self.head = nn.Linear(self.d, V)

    def encode_trunk(self, x: torch.Tensor) -> torch.Tensor:
        """Compute initial representation z [B, L, d]."""
        _, t_len = x.shape
        z = self.tok_emb(x) + self.pos_emb[:t_len, :]
        for blk in self.enc_blocks:
            z = blk(z)
        return z

    def contrib(self, k: int, code: torch.Tensor) -> torch.Tensor:
        """
        Apply R_k to a code tensor [B, l_k, code_dim] or discrete indices [B, l_k].

        Args:
            k: 1-indexed level index (1..K).
            code: Tensor of code vectors or integer codebook indices.

        Returns:
            Level contribution tensor of shape [B, L, d].
        """
        idx = k - 1
        if code.dtype in (torch.long, torch.int32, torch.int64):
            code_vecs = self.quantizers[idx].lookup(code)
        else:
            code_vecs = code
        return self.reconstructors[idx](code_vecs)

    def reconstruct(self, codes: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Accumulate zhat reconstructions from a list of codes for levels 1..len(codes).

        Returns:
            List of len(codes) + 1 tensors of shape [B, L, d] starting with zhat[0] = 0.
        """
        b = codes[0].shape[0]
        device = codes[0].device
        dtype = self.tok_emb.weight.dtype

        zhat = [torch.zeros(b, self.L, self.d, device=device, dtype=dtype)]
        for k_idx, code in enumerate(codes, start=1):
            contrib_k = self.contrib(k_idx, code)
            zhat.append(zhat[-1] + contrib_k)
        return zhat

    def encode(
        self,
        x: torch.Tensor,
        cutoff: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Encoded:
        """
        Encode sequence x into multiscale residual hierarchy up to cutoff.

        Args:
            x: Token tensor of shape [B, L].
            cutoff: Optional integer cutoff in 1..K (default None = all K levels).
            generator: Optional torch.Generator for channel noise reproducibility.

        Returns:
            Encoded dataclass containing targets, codes, zhat, and quant_loss.
        """
        z = self.encode_trunk(x)
        max_k = self.K if cutoff is None else min(cutoff, self.K)

        zhat: list[torch.Tensor] = [torch.zeros_like(z)]
        targets: list[torch.Tensor] = []
        codes: list[torch.Tensor] = []
        total_quant_loss = torch.tensor(0.0, device=x.device)

        for k in range(1, max_k + 1):
            r = z - zhat[-1]  # Residual [B, L, d]
            u = down(r, self.schedule[k - 1])  # [B, l_k, d]

            quantizer = self.quantizers[k - 1]
            if self.quantizer_type == "noisy":
                target_k, code_k, q_loss = quantizer(u, generator=generator)
            else:
                target_k, code_k, q_loss = quantizer(u)

            targets.append(target_k)
            codes.append(code_k)
            total_quant_loss = total_quant_loss + q_loss

            contrib_k = self.contrib(k, code_k)
            zhat.append(zhat[-1] + contrib_k)

        return Encoded(
            targets=targets,
            codes=codes,
            zhat=zhat,
            quant_loss=total_quant_loss,
        )

    def decode(self, zhat_k: torch.Tensor) -> torch.Tensor:
        """
        Decode cumulative reconstruction tensor zhat_k [B, L, d] into token logits [B, L, V].
        """
        _, t_len, _ = zhat_k.shape
        h = zhat_k + self.dec_pos_emb[:t_len, :]
        for blk in self.dec_blocks:
            h = blk(h)
        h = self.dec_norm(h)
        logits = self.head(h)
        return logits

    def forward(
        self,
        x: torch.Tensor,
        cutoff: int | None = None,
        p_drop: float | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, Encoded, int]:
        """
        Forward pass with optional nested dropout (SPEC 4.4).

        Returns:
            logits: Decoded logits [B, L, V] from zhat[cut].
            enc: Encoded representations.
            cut: Cutoff level used.
        """
        if cutoff is None:
            if p_drop is None:
                p_drop = self.cfg.p_drop if self.training else 0.0
            cut = sample_cutoff(self.K, p_drop=p_drop, generator=generator)
        else:
            cut = cutoff

        enc = self.encode(x, cutoff=cut, generator=generator)
        logits = self.decode(enc.zhat[cut])
        return logits, enc, cut
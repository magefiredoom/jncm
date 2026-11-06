"""Level-by-level sampling for the generator (SPEC 5.6).

Sampling walks the hierarchy coarse to fine: level k is embedded from the
reconstruction prefix ``zhat_{k-1}``, run through the backbone (one forward pass
for that level only), sampled by the head, and added back through ``contrib``.
Exactly K backbone forward passes are made per batch, as the acceptance test
counts with a forward hook.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from jncm.generator.backbone import GeneratorBackbone
    from jncm.generator.heads import LevelHead
    from jncm.tokenizer import MultiscaleTokenizer


def sample_tokens(
    tokenizer: "MultiscaleTokenizer",
    backbone: "GeneratorBackbone",
    head: "LevelHead",
    batch_size: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate token ids coarse to fine (SPEC 5.6).

    ```
    zhat = 0
    for k in 1..K:
        s_k = down(zhat, l_k); h = backbone(level k only)
        code = head.sample(h)
        zhat = zhat + tok.contrib(k, code)
    tokens = argmax(tok.decode(zhat))
    ```

    The modules' train/eval modes are left untouched (the caller decides); the
    whole loop runs under ``torch.no_grad()``.

    Args:
        tokenizer: Frozen multiscale tokenizer; supplies ``contrib`` and
            ``decode`` and fixes ``L``, ``K`` and ``d``.
        backbone: Generator backbone whose K matches the tokenizer's K.
        head: Output head whose K matches the tokenizer's K.
        batch_size: Number of sequences to generate (>= 1).
        generator: Optional ``torch.Generator`` for every random draw (head
            sampling, flow integration start, channel noise), so a seed gives
            the same samples.

    Returns:
        LongTensor of token ids [batch_size, L], each in [0, V).
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if not (backbone.K == tokenizer.K == head.K):
        raise ValueError(
            "level counts disagree: "
            f"backbone.K={backbone.K}, tokenizer.K={tokenizer.K}, head.K={head.K}"
        )
    if (backbone.L, backbone.d) != (tokenizer.L, tokenizer.d):
        raise ValueError(
            "backbone must match the tokenizer: "
            f"backbone (L, d)=({backbone.L}, {backbone.d}) vs "
            f"tokenizer (L, d)=({tokenizer.L}, {tokenizer.d})"
        )

    k_levels = tokenizer.K
    with torch.no_grad():
        # zhat_0 = 0, on the tokenizer's device and dtype.
        zhat = tokenizer.tok_emb.weight.new_zeros((batch_size, tokenizer.L, tokenizer.d))
        for k in range(1, k_levels + 1):
            level_input = backbone.embed_level(tokenizer, zhat, k)  # [B, l_k, H]
            hidden = backbone(level_input)  # exactly one call per level
            code = head.sample_level(hidden, k, generator=generator)
            zhat = zhat + tokenizer.contrib(k, code)  # [B, L, d]
        logits = tokenizer.decode(zhat)  # [B, L, V]
    return logits.argmax(dim=-1)

"""Tests for the level-block generator backbone (SPEC 5.1/5.2, TASKS T3.1).

Acceptance:
(a) no leak: with eval() and a fixed seed, two inputs that differ only in codes
    of levels >= k give identical level-k outputs.
(b) padding invariance: changing values at padded positions does not change
    valid outputs.
(c) shapes.
"""

import pytest
import torch

from jncm.config import GeneratorConfig, TokenizerConfig
from jncm.generator.backbone import (
    GeneratorBackbone,
    LevelBlockTransformer,
    LevelInputEmbedding,
    SinusoidalPositionProjection,
)
from jncm.levels import get_level_schedule
from jncm.tokenizer import MultiscaleTokenizer
from jncm.utils import seed_everything

L = 32
V = 64
K = 4
SCHEDULE = get_level_schedule(L, K)  # [1, 11, 22, 32]


def _build(
    quantizer: str = "vq",
    H: int = 32,
    n_layers: int = 1,
    seed: int = 42,
) -> tuple[MultiscaleTokenizer, GeneratorBackbone]:
    """Build an eval-mode tokenizer and a small eval-mode generator backbone."""
    seed_everything(seed)
    tok_cfg = TokenizerConfig(
        d=64, K=K, quantizer=quantizer, V_k=64, d_code=16, d_k=8, sigma_k=0.4
    )
    tok = MultiscaleTokenizer(tok_cfg, L=L, V=V)
    tok.eval()
    gen_cfg = GeneratorConfig(
        H=H, n_layers=n_layers, n_heads=2, mlp_ratio=4.0, dropout=0.1
    )
    backbone = GeneratorBackbone.from_tokenizer(tok, gen_cfg)
    backbone.eval()
    return tok, backbone


def _code_width(quantizer: str) -> int:
    return 16 if quantizer == "vq" else 8


def _fake_codes(quantizer: str, batch: int, seed: int) -> list[torch.Tensor]:
    """Random chain codes, one [B, l_k, code_width] tensor per level."""
    gen = torch.Generator().manual_seed(seed)
    width = _code_width(quantizer)
    return [
        torch.randn(batch, length, width, generator=gen) for length in SCHEDULE
    ]


def test_backbone_shapes_vq() -> None:
    # Acceptance (c): one hidden tensor per level with width H.
    tok, backbone = _build("vq")
    assert backbone.schedule == SCHEDULE
    assert backbone.K == K
    assert backbone.L == L
    assert backbone.d == tok.d

    batch = 3
    x = torch.randint(0, V, (batch, L))
    enc = tok.encode(x)
    hidden = backbone.forward_codes(tok, enc.codes)

    assert len(hidden) == K
    for k, length in enumerate(SCHEDULE):
        assert hidden[k].shape == (batch, length, backbone.H)

    # Single-level path used by the sampler (SPEC 5.6).
    zhat_prev = torch.zeros(batch, L, tok.d)
    for k in range(1, K + 1):
        level_input = backbone.embed_level(tok, zhat_prev, k)
        assert level_input.shape == (batch, SCHEDULE[k - 1], backbone.H)
        assert backbone.forward(level_input).shape == (batch, SCHEDULE[k - 1], backbone.H)


def test_backbone_shapes_noisy() -> None:
    # Acceptance (c) for the continuous variant: chain codes carry d_k channels.
    tok, backbone = _build("noisy")
    batch = 2
    x = torch.randint(0, V, (batch, L))
    enc = tok.encode(x)
    hidden = backbone.forward_codes(tok, enc.codes)
    for k, length in enumerate(SCHEDULE):
        assert hidden[k].shape == (batch, length, backbone.H)


@pytest.mark.parametrize("quantizer", ["vq", "noisy"])
def test_no_leak_current_and_later_levels(quantizer: str) -> None:
    # Acceptance (a): changing codes at levels >= k must not change level-k outputs.
    tok, backbone = _build(quantizer)
    batch = 2
    codes = _fake_codes(quantizer, batch, seed=1)
    baseline = backbone.forward_codes(tok, codes)

    for k in range(1, K + 1):
        changed = list(codes)
        alt = _fake_codes(quantizer, batch, seed=100 + k)
        for level_idx in range(k - 1, K):
            changed[level_idx] = alt[level_idx]
        out = backbone.forward_codes(tok, changed)
        for level_idx in range(k):
            assert torch.equal(baseline[level_idx], out[level_idx]), (
                f"{quantizer}: level {level_idx + 1} output changed when only "
                f"codes of levels >= {k} changed"
            )


def test_earlier_levels_do_change_later_outputs() -> None:
    # Non-vacuity guard for the no-leak test: the conditioning prefix is used.
    tok, backbone = _build("vq")
    batch = 2
    codes = _fake_codes("vq", batch, seed=3)
    baseline = backbone.forward_codes(tok, codes)

    for k in range(2, K + 1):
        changed = list(codes)
        alt = _fake_codes("vq", batch, seed=200 + k)
        changed[k - 2] = alt[k - 2]  # level k-1 only
        out = backbone.forward_codes(tok, changed)
        assert not torch.equal(baseline[k - 1], out[k - 1]), (
            f"level {k} output ignored codes of level {k - 1}"
        )


def test_padding_invariance() -> None:
    # Acceptance (b): values at padded positions never reach valid outputs.
    tok, backbone = _build("vq", n_layers=2)
    batch = 2
    reps = backbone.embed_levels(tok, _fake_codes("vq", batch, seed=7))

    garbage: list[torch.Tensor] = []
    gen = torch.Generator().manual_seed(11)
    for rep, length in zip(reps, SCHEDULE):
        pad_len = L - length
        pad = torch.randn(batch, pad_len, backbone.H, generator=gen) * 100.0
        garbage.append(torch.cat([rep, pad], dim=1))

    out_garbage = backbone.forward_levels(garbage, lengths=SCHEDULE)
    out_clean = backbone.forward_levels(reps)

    for level_idx, (a, b) in enumerate(zip(out_garbage, out_clean)):
        assert a.shape == (batch, SCHEDULE[level_idx], backbone.H)
        assert torch.equal(a, b), f"padding leaked into level {level_idx + 1}"


def test_single_level_matches_packed_first_call() -> None:
    # The sampler's one-level forward must equal the trained packed path.
    tok, backbone = _build("vq", n_layers=2)
    batch = 2
    codes = _fake_codes("vq", batch, seed=99)
    packed = backbone.forward_codes(tok, codes)

    zhat = torch.zeros(batch, L, tok.d)
    for k in range(1, K + 1):
        single = backbone.forward(backbone.embed_level(tok, zhat, k))
        assert torch.allclose(single, packed[k - 1], atol=1e-5)
        if k < K:
            zhat = zhat + tok.contrib(k, codes[k - 1])


def test_embed_levels_requires_prefix_codes() -> None:
    tok, backbone = _build("vq")
    with pytest.raises(ValueError):
        backbone.embed_levels(tok, _fake_codes("vq", 1, seed=5)[: K - 2])


def test_position_features_are_normalized() -> None:
    # p_i = (i + 0.5) / length in (0, 1); sin and cos features of width 2*n_freq.
    proj = SinusoidalPositionProjection(H=32, n_freq=64)
    feats = proj._features(8)
    assert feats.shape == (1, 8, 128)
    projected = proj([8, 1])
    assert projected[0].shape == (1, 8, 32)
    assert projected[1].shape == (1, 1, 32)

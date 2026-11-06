"""Tests for conditioning corruption (SPEC 5.4, TASKS T3.3).

Acceptance:
- empirical swap rate matches p_j within 0.03;
- targets are untouched;
- ``none`` is the identity.
"""

import pytest
import torch

from jncm.config import CorruptionConfig, TokenizerConfig
from jncm.generator import (
    GeneratorBackbone,
    add_code_noise,
    corrupt_codes,
    linear_schedule,
    swap_codes,
)
from jncm.config import GeneratorConfig
from jncm.levels import get_level_schedule
from jncm.tokenizer import MultiscaleTokenizer
from jncm.utils import seed_everything

L = 32
V = 64
K = 4
SCHEDULE = get_level_schedule(L, K)  # [1, 11, 22, 32]


def _synthetic_codes(
    batch: int = 8, width: int = 8, seed: int = 0, dtype: torch.dtype = torch.float32
) -> list[torch.Tensor]:
    """Distinct random chain codes [B, l_k, width], one per level."""
    gen = torch.Generator().manual_seed(seed)
    return [
        torch.randn(batch, length, width, generator=gen).to(dtype)
        for length in SCHEDULE
    ]


def _changed_fraction(before: torch.Tensor, after: torch.Tensor) -> float:
    """Fraction of (batch, position) slots whose code tensor changed."""
    assert before.shape == after.shape
    changed = (before.reshape(before.shape[0], before.shape[1], -1) != after.reshape(
        after.shape[0], after.shape[1], -1
    )).any(dim=-1)
    return float(changed.float().mean())


def test_linear_schedule_endpoints_and_monotone() -> None:
    rates = linear_schedule(K, 0.8, 0.2)
    assert len(rates) == K
    assert rates[0] == pytest.approx(0.8)
    assert rates[-1] == pytest.approx(0.2)
    assert rates == pytest.approx([0.8, 0.6, 0.4, 0.2])
    assert all(a > b for a, b in zip(rates, rates[1:]))
    assert linear_schedule(1, 0.8, 0.2) == pytest.approx([0.8])
    with pytest.raises(ValueError):
        linear_schedule(0, 0.8, 0.2)


def test_none_is_identity() -> None:
    # Acceptance: mode "none" returns the codes unchanged.
    codes = _synthetic_codes(batch=4)
    cfg = CorruptionConfig(mode="none")
    out = corrupt_codes(codes, cfg, generator=torch.Generator().manual_seed(0))
    assert len(out) == len(codes)
    for got, want in zip(out, codes):
        assert got is want
        assert torch.equal(got, want)


def test_swap_rate_matches_schedule() -> None:
    # Acceptance: empirical swap rate matches p_j within 0.03 at every level.
    batch = 1024  # level 1 has l_1 = 1, so 1024 slots is the tightest estimate
    codes = _synthetic_codes(batch=batch, width=4, seed=3)
    cfg = CorruptionConfig(mode="swap", p_first=0.8, p_last=0.2)
    rates = linear_schedule(K, cfg.p_first, cfg.p_last)
    out = corrupt_codes(codes, cfg, generator=torch.Generator().manual_seed(11))

    for level, (before, after, rate) in enumerate(zip(codes, out, rates), start=1):
        assert after.shape == before.shape
        observed = _changed_fraction(before, after)
        assert abs(observed - rate) < 0.03, (
            f"level {level}: swap rate {observed:.3f} vs p_j={rate:.3f}"
        )

    # A single batch/position slot cannot be swapped with "another" position.
    single = swap_codes([torch.randn(1, 1, 4)], [0.8], generator=torch.Generator().manual_seed(1))
    assert single[0].shape == (1, 1, 4)


def test_swap_uses_other_positions_of_the_same_level() -> None:
    # Every swapped code is another position's code from its own level.
    batch = 16
    width = 3
    codes = _synthetic_codes(batch=batch, width=width, seed=5)
    rates = [1.0] * K
    out = swap_codes(codes, rates, generator=torch.Generator().manual_seed(2))

    for level, (before, after, length) in enumerate(zip(codes, out, SCHEDULE), start=1):
        pool = {tuple(row) for row in before.reshape(-1, width).tolist()}
        for row in after.reshape(-1, width).tolist():
            assert tuple(row) in pool, f"level {level}: swapped in a foreign code"
        # Distinct random codes: a swap never leaves the position untouched.
        if batch * length > 1:
            assert _changed_fraction(before, after) == pytest.approx(1.0)


def test_swap_is_seeded_and_out_of_place() -> None:
    codes = _synthetic_codes(batch=8, seed=7)
    snapshot = [c.clone() for c in codes]
    cfg = CorruptionConfig(mode="swap")

    a = corrupt_codes(codes, cfg, generator=torch.Generator().manual_seed(4))
    b = corrupt_codes(codes, cfg, generator=torch.Generator().manual_seed(4))
    c = corrupt_codes(codes, cfg, generator=torch.Generator().manual_seed(5))

    for level, (x, y) in enumerate(zip(a, b), start=1):
        assert torch.equal(x, y), f"level {level} is not reproducible"
    assert any(not torch.equal(x, y) for x, y in zip(a, c))
    for before, after in zip(codes, snapshot):
        assert torch.equal(before, after), "corruption modified its input"


def test_swap_leaves_targets_untouched_through_tokenizer() -> None:
    # Acceptance: the tokenizer targets (and the encoder codes) stay clean.
    seed_everything(0)
    tok = MultiscaleTokenizer(
        TokenizerConfig(d=64, K=K, quantizer="vq", V_k=64, d_code=16, d_k=8),
        L=L,
        V=V,
    )
    tok.eval()
    backbone = GeneratorBackbone.from_tokenizer(
        tok, GeneratorConfig(H=32, n_layers=1, n_heads=2, dropout=0.0)
    )
    backbone.eval()

    x = torch.randint(0, V, (4, L), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        enc = tok.encode(x)
        reference = list(enc.codes)
        targets_before = [t.clone() for t in enc.targets]
        clean_hidden = backbone.forward_codes(tok, enc.codes)

        corrupted = corrupt_codes(
            enc.codes,
            CorruptionConfig(mode="swap", p_first=1.0, p_last=1.0),
            generator=torch.Generator().manual_seed(3),
        )
        dirty_hidden = backbone.forward_codes(tok, corrupted)

    for level, (before, after) in enumerate(zip(enc.targets, targets_before), start=1):
        assert torch.equal(before, after), f"level {level} targets changed"
        assert before.dtype in (torch.long, torch.int64)
    for level, (before, after) in enumerate(zip(enc.codes, reference), start=1):
        assert torch.equal(before, after), f"level {level} encoder codes changed"
    # Corruption is applied to the conditioning, so level outputs do change.
    assert any(not torch.equal(a, b) for a, b in zip(clean_hidden, dirty_hidden))


def test_noise_mode_matches_sigma_schedule() -> None:
    codes = [torch.zeros(512, length, 8) for length in SCHEDULE]
    cfg = CorruptionConfig(mode="noise", s_first=0.5, s_last=0.1)
    sigmas = linear_schedule(K, cfg.s_first, cfg.s_last)
    out = corrupt_codes(
        codes,
        cfg,
        generator=torch.Generator().manual_seed(6),
        quantizer="noisy",
    )

    for level, (after, sigma) in enumerate(zip(out, sigmas), start=1):
        assert after.shape == (512, SCHEDULE[level - 1], 8)
        assert torch.isfinite(after).all()
        observed_std = float(after.std())
        observed_mean = float(after.mean())
        assert abs(observed_std - sigma) < 0.02, (
            f"level {level}: noise std {observed_std:.3f} vs s_j={sigma:.3f}"
        )
        assert abs(observed_mean) < 0.02, f"level {level}: noise mean {observed_mean:.3f}"


def test_noise_requires_noisy_variant() -> None:
    codes = _synthetic_codes(batch=4)
    cfg = CorruptionConfig(mode="noise")
    with pytest.raises(ValueError):
        corrupt_codes(codes, cfg, quantizer="vq")
    with pytest.raises(ValueError):
        corrupt_codes(codes, cfg)
    with pytest.raises(ValueError):
        add_code_noise([torch.zeros(2, 3, dtype=torch.long)], [0.5])


def test_invalid_arguments_raise() -> None:
    codes = _synthetic_codes(batch=4)
    with pytest.raises(ValueError):
        corrupt_codes(codes, CorruptionConfig(mode="nope"))
    with pytest.raises(ValueError):
        swap_codes(codes, [0.5])  # wrong number of rates
    with pytest.raises(ValueError):
        swap_codes(codes, [1.5, 0.5, 0.5, 0.5])
    with pytest.raises(ValueError):
        add_code_noise(codes, [0.5, 0.5, -0.1, 0.5])

"""Tests for generator heads and losses (SPEC 5.3/5.5, TASKS T3.2).

Acceptance:
- ce and regress overfit a fixed batch;
- flow learns a 2-D toy (two Gaussians at +-1, std 0.1) and its samples are
  bimodal (at least 35% in each mode);
- seeded determinism.

The overfit batches use batch size 1: at level 1 the backbone input is built
from zhat_0 = 0, so every row of a level-1 batch sees the same input and a
per-position head can only ever emit one prediction. A perfect batch fit at
level 1 is therefore possible only for one sequence, and the same argument
holds for the regression floor (variance of the level-1 targets).
"""

import pytest
import torch
import torch.nn.functional as F

from jncm.config import GeneratorConfig, TokenizerConfig
from jncm.generator import (
    CEHead,
    FlowHead,
    GeneratorBackbone,
    RegressHead,
    build_head,
    level_weights,
    sample_codes,
    weighted_total,
)
from jncm.levels import get_level_schedule
from jncm.tokenizer import MultiscaleTokenizer
from jncm.utils import seed_everything

L = 32
V = 64
K = 4
SCHEDULE = get_level_schedule(L, K)  # [1, 11, 22, 32]


def _pipeline(
    quantizer: str = "vq",
    head: str = "ce",
    batch: int = 1,
    H: int = 32,
    n_layers: int = 1,
    seed: int = 42,
) -> tuple[
    GeneratorBackbone,
    CEHead | RegressHead | FlowHead,
    list[torch.Tensor],
    list[torch.Tensor],
]:
    """Frozen tokenizer -> level inputs -> backbone -> head, on a fixed batch.

    Returns (backbone, head, level_inputs, targets) where targets are codebook
    indices for ``ce`` and clean continuous codes for ``noisy`` heads.
    """
    seed_everything(seed)
    tok_cfg = TokenizerConfig(
        d=64, K=K, quantizer=quantizer, V_k=64, d_code=16, d_k=8, sigma_k=0.4
    )
    tok = MultiscaleTokenizer(tok_cfg, L=L, V=V)
    tok.eval()
    for param in tok.parameters():
        param.requires_grad_(False)

    gen_cfg = GeneratorConfig(
        H=H, n_layers=n_layers, n_heads=2, dropout=0.0, head=head
    )
    backbone = GeneratorBackbone.from_tokenizer(tok, gen_cfg)
    head_module = build_head(
        gen_cfg,
        H=gen_cfg.H,
        K=K,
        V_k=tok_cfg.V_k,
        d_k=tok_cfg.d_k,
        sigma_k=tok_cfg.sigma_k,
    )

    x = torch.randint(0, V, (batch, L), generator=torch.Generator().manual_seed(seed))
    with torch.no_grad():
        enc = tok.encode(x, generator=torch.Generator().manual_seed(seed + 7))
        level_inputs = backbone.embed_levels(tok, enc.codes)
    return backbone, head_module, level_inputs, enc.targets


def _overfit(
    backbone: GeneratorBackbone,
    head: CEHead | RegressHead | FlowHead,
    level_inputs: list[torch.Tensor],
    targets: list[torch.Tensor],
    steps: int,
    lr: float,
) -> tuple[list[torch.Tensor], float, list[float]]:
    """Train backbone + head on one fixed batch; return final hidden states, loss."""
    optimizer = torch.optim.Adam(
        list(backbone.parameters()) + list(head.parameters()), lr=lr
    )
    backbone.train()
    head.train()
    noise = torch.Generator().manual_seed(1234)
    for _ in range(steps):
        hidden = backbone.forward_levels(level_inputs)
        loss, _ = head.loss(hidden, targets, generator=noise)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    backbone.eval()
    head.eval()
    with torch.no_grad():
        hidden = backbone.forward_levels(level_inputs)
        loss, per_level = head.loss(
            hidden, targets, generator=torch.Generator().manual_seed(99)
        )
    return hidden, float(loss.item()), [float(v) for v in per_level]


def test_ce_overfits_fixed_batch() -> None:
    # Acceptance: ce fits a fixed batch of code indices exactly.
    backbone, head, level_inputs, targets = _pipeline("vq", "ce", batch=1)
    assert isinstance(head, CEHead)
    assert head.code_dim == 64

    hidden, loss, per_level = _overfit(backbone, head, level_inputs, targets, 500, 5e-3)
    assert loss < 1e-2, f"ce did not overfit: loss={loss}, per_level={per_level}"
    assert max(per_level) < 1e-2, f"per-level ce losses: {per_level}"

    with torch.no_grad():
        correct = 0
        total = 0
        for k, (h, t) in enumerate(zip(hidden, targets), start=1):
            assert h.shape[:2] == t.shape
            pred = head.heads[k - 1](h).argmax(dim=-1)
            correct += int((pred == t).sum())
            total += t.numel()
    assert correct == total, f"ce memorised {correct}/{total} code indices"


def test_regress_overfits_fixed_batch() -> None:
    # Acceptance: regress fits a fixed batch of clean continuous codes.
    backbone, head, level_inputs, targets = _pipeline("noisy", "regress", batch=1)
    assert isinstance(head, RegressHead)
    assert head.code_dim == 8

    hidden, loss, per_level = _overfit(backbone, head, level_inputs, targets, 1200, 5e-3)
    assert loss < 2e-3, f"regress did not overfit: loss={loss}, per_level={per_level}"
    assert max(per_level) < 1e-3, f"per-level regress losses: {per_level}"

    with torch.no_grad():
        for k, (h, t) in enumerate(zip(hidden, targets), start=1):
            pred = head.heads[k - 1](h)
            cos = F.cosine_similarity(pred, t, dim=-1)
            assert float(cos.min()) > 0.99, f"level {k} cosine {float(cos.min())}"


def test_flow_learns_bimodal_toy() -> None:
    # Acceptance: flow learns a 2-D two-Gaussian toy and samples are bimodal.
    seed_everything(0)
    n = 512
    std = 0.1
    d_k = 2
    h_dim = 8
    head = FlowHead(
        H=h_dim,
        d_k=d_k,
        K=1,
        sigma_k=0.4,
        n_rep=4,
        flow_steps=8,
        add_channel_noise=False,
        hidden=64,
        n_blocks=3,
    )

    gen = torch.Generator().manual_seed(1)
    side = torch.where(
        torch.randint(0, 2, (n, 1), generator=gen) == 1, 1.0, -1.0
    )  # [n, 1]
    data = side + std * torch.randn(n, d_k, generator=gen)
    hidden = torch.zeros(1, n, h_dim)
    targets = [data.unsqueeze(0)]  # [1, n, d_k]

    optimizer = torch.optim.Adam(head.parameters(), lr=2e-3)
    noise = torch.Generator().manual_seed(2)
    head.train()
    first_loss: float | None = None
    for _ in range(600):
        loss, _ = head.loss([hidden], targets, generator=noise)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if first_loss is None:
            first_loss = float(loss.item())
    # The two-mode target keeps an irreducible ambiguity floor (about 0.86 here),
    # so acceptance is the bimodality below, not a small absolute loss.
    final_loss = float(loss.detach())
    assert final_loss < 0.6 * first_loss, (
        f"toy flow loss did not fall: {first_loss:.3f} -> {final_loss:.3f}"
    )

    head.eval()
    with torch.no_grad():
        samples = head.sample_level(
            hidden, 1, generator=torch.Generator().manual_seed(3)
        )[0]  # [n, d_k]
    assert samples.shape == (n, d_k)

    pos = torch.tensor([1.0, 1.0]).expand(n, d_k)
    neg = torch.tensor([-1.0, -1.0]).expand(n, d_k)
    d_pos = (samples - pos).norm(dim=-1)
    d_neg = (samples - neg).norm(dim=-1)
    nearest = torch.minimum(d_pos, d_neg)
    frac_pos = float((d_pos < d_neg).float().mean())
    frac_neg = 1.0 - frac_pos

    assert frac_pos >= 0.35, f"only {frac_pos:.2f} of samples in the +1 mode"
    assert frac_neg >= 0.35, f"only {frac_neg:.2f} of samples in the -1 mode"
    # Guards against a unimodal blob at the origin passing the 35/35 split.
    near_mode = float((nearest < 0.5).float().mean())
    assert near_mode >= 0.8, f"only {near_mode:.2f} of samples are near a mode"


@pytest.mark.parametrize(
    "quantizer,head_name",
    [("vq", "ce"), ("noisy", "regress"), ("noisy", "flow")],
)
def test_sampling_is_reproducible_and_seeded(quantizer: str, head_name: str) -> None:
    # Acceptance: seeded determinism for every stochastic sampler.
    _, head, level_inputs, _ = _pipeline(quantizer, head_name, batch=2, H=16)
    with torch.no_grad():
        hidden = [torch.randn(2, length, head.H) for length in SCHEDULE]
        first = sample_codes(head, hidden, generator=torch.Generator().manual_seed(7))
        second = sample_codes(head, hidden, generator=torch.Generator().manual_seed(7))
        other = sample_codes(head, hidden, generator=torch.Generator().manual_seed(8))

    for k, length in enumerate(SCHEDULE):
        assert len(first) == K
        assert first[k].shape == second[k].shape
        assert torch.equal(first[k], second[k]), f"level {k + 1} is not reproducible"
        assert not torch.equal(first[k], other[k]), f"level {k + 1} ignores the seed"
        if head_name == "ce":
            assert first[k].dtype in (torch.long, torch.int64)
            assert first[k].shape == (2, length)
            assert int(first[k].min()) >= 0 and int(first[k].max()) < head.V_k
        else:
            assert first[k].shape == (2, length, head.code_dim)


def test_training_loss_is_seeded_for_flow() -> None:
    _, head, _, _ = _pipeline("noisy", "flow", batch=1, H=16)
    hidden = [torch.randn(2, length, head.H) for length in SCHEDULE]
    targets = [
        torch.randn(2, length, head.code_dim) for length in SCHEDULE
    ]
    loss_a, per_a = head.loss(hidden, targets, generator=torch.Generator().manual_seed(5))
    loss_b, per_b = head.loss(hidden, targets, generator=torch.Generator().manual_seed(5))
    assert torch.equal(loss_a, loss_b)
    assert float(loss_a.item()) > 0.0
    for a, b in zip(per_a, per_b):
        assert torch.equal(a, b)


def test_regress_sample_renorm_and_channel_noise() -> None:
    head = RegressHead(
        H=16, d_k=8, K=2, sigma_k=0.4, renorm=True, add_channel_noise=False
    )
    h = torch.randn(3, 5, 16)
    code = head.sample_level(h, 1)
    assert code.shape == (3, 5, 8)
    # LayerNorm without affine gives unit variance across the code dimension.
    var = code.var(dim=-1, unbiased=False)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-3)

    raw = RegressHead(
        H=16, d_k=8, K=2, sigma_k=0.4, renorm=False, add_channel_noise=False
    )
    raw.heads[0].load_state_dict(head.heads[0].state_dict())
    assert not torch.allclose(raw.sample_level(h, 1), code)

    noisy = RegressHead(
        H=16, d_k=8, K=2, sigma_k=0.4, renorm=False, add_channel_noise=True
    )
    noisy.heads[0].load_state_dict(raw.heads[0].state_dict())
    a = noisy.sample_level(h, 1, generator=torch.Generator().manual_seed(1))
    b = noisy.sample_level(h, 1, generator=torch.Generator().manual_seed(1))
    c = noisy.sample_level(h, 1, generator=torch.Generator().manual_seed(2))
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_flow_sample_level_shape_and_noise_flag() -> None:
    clean = FlowHead(
        H=8, d_k=8, K=2, n_rep=2, flow_steps=4, add_channel_noise=False, hidden=16
    )
    h = torch.randn(2, 7, 8)
    code = clean.sample_level(h, 2, generator=torch.Generator().manual_seed(0))
    assert code.shape == (2, 7, 8)
    assert torch.isfinite(code).all()


def test_ce_loss_matches_cross_entropy() -> None:
    head = CEHead(H=8, V_k=6, K=2)
    hidden = [torch.randn(2, 3, 8), torch.randn(2, 5, 8)]
    targets = [
        torch.randint(0, 6, (2, 3)),
        torch.randint(0, 6, (2, 5)),
    ]
    total, per_level = head.loss(hidden, targets)
    manual = [
        F.cross_entropy(head.heads[k](hidden[k]).reshape(-1, 6), targets[k].reshape(-1))
        for k in range(2)
    ]
    assert torch.allclose(total, torch.stack(manual).sum())
    for got, want in zip(per_level, manual):
        assert torch.allclose(got, want)


def test_regress_cosine_option() -> None:
    head = RegressHead(H=8, d_k=4, K=1, cosine=True)
    hidden = [torch.randn(2, 3, 8)]
    targets = [torch.randn(2, 3, 4)]
    total, per_level = head.loss(hidden, targets)
    pred = head.heads[0](hidden[0])
    manual = (1.0 - F.cosine_similarity(pred, targets[0], dim=-1)).mean()
    assert torch.allclose(total, manual)
    assert torch.allclose(per_level[0], manual)


def test_level_weights_and_weighted_total() -> None:
    w = level_weights(SCHEDULE, alpha=0.0)
    assert len(w) == K
    assert all(abs(wk - 1.0 / sum(SCHEDULE)) < 1e-12 for wk in w)
    # alpha = 1: equal weight per level, and sum_k w_k * l_k == 1.
    w1 = level_weights(SCHEDULE, alpha=1.0)
    assert all(abs(wk * length - 1.0 / K) < 1e-12 for wk, length in zip(w1, SCHEDULE))
    assert abs(sum(wk * length for wk, length in zip(w1, SCHEDULE)) - 1.0) < 1e-12

    losses = [torch.tensor(0.5) for _ in SCHEDULE]
    assert abs(float(weighted_total(losses, SCHEDULE, alpha=0.0)) - 0.5) < 1e-6
    assert abs(float(weighted_total(losses, SCHEDULE, alpha=1.0)) - 0.5) < 1e-6

    # Different per-level means: alpha=0 is the position-weighted mean.
    level_means = [float(length) for length in SCHEDULE]
    expected = sum(l * l for l in SCHEDULE) / sum(SCHEDULE)
    got = float(weighted_total([torch.tensor(m) for m in level_means], SCHEDULE, 0.0))
    assert abs(got - expected) < 1e-5


def test_build_head_validates_arguments() -> None:
    with pytest.raises(ValueError):
        build_head(GeneratorConfig(head="nope"), H=8, K=1, V_k=4)
    with pytest.raises(ValueError):
        build_head(GeneratorConfig(head="ce"), H=8, K=1)
    with pytest.raises(ValueError):
        build_head(GeneratorConfig(head="regress"), H=8, K=1, d_k=4)
    head = build_head(
        GeneratorConfig(head="flow", flow_steps=5, n_rep=2), H=8, K=1, d_k=4, sigma_k=0.4
    )
    assert isinstance(head, FlowHead)
    assert head.flow_steps == 5 and head.n_rep == 2
    reg = build_head(GeneratorConfig(head="regress", regress_cosine=True), H=8, K=1, d_k=4, sigma_k=0.4)
    assert isinstance(reg, RegressHead) and reg.cosine is True

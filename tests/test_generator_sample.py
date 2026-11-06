"""Tests for the level-by-level sampler (SPEC 5.6, TASKS T3.4).

Acceptance:
- exactly K backbone forward passes per batch (counted with a forward hook);
- correct shapes and token ids in [0, V);
- the same seed gives the same samples.
"""

import pytest
import torch

from jncm.config import GeneratorConfig, TokenizerConfig
from jncm.generator import (
    FlowHead,
    GeneratorBackbone,
    LevelHead,
    RegressHead,
    build_head,
    sample_tokens,
)
from jncm.levels import get_level_schedule
from jncm.tokenizer import MultiscaleTokenizer
from jncm.utils import seed_everything

L = 32
V = 64
K = 4
H = 16
SCHEDULE = get_level_schedule(L, K)  # [1, 11, 22, 32]


def _pipeline(
    quantizer: str = "vq",
    head_name: str = "ce",
    seed: int = 42,
    K_override: int | None = None,
) -> tuple[MultiscaleTokenizer, GeneratorBackbone, LevelHead]:
    """Frozen eval-mode tokenizer, backbone and head, all consistent."""
    seed_everything(seed)
    k_levels = K if K_override is None else K_override
    tok_cfg = TokenizerConfig(
        d=64, K=k_levels, quantizer=quantizer, V_k=64, d_code=16, d_k=8, sigma_k=0.4
    )
    tok = MultiscaleTokenizer(tok_cfg, L=L, V=V)
    tok.eval()
    for param in tok.parameters():
        param.requires_grad_(False)

    gen_cfg = GeneratorConfig(H=H, n_layers=1, n_heads=2, dropout=0.0, head=head_name)
    backbone = GeneratorBackbone.from_tokenizer(tok, gen_cfg)
    backbone.eval()
    head = build_head(
        gen_cfg,
        H=gen_cfg.H,
        K=k_levels,
        V_k=tok_cfg.V_k,
        d_k=tok_cfg.d_k,
        sigma_k=tok_cfg.sigma_k,
    )
    head.eval()
    return tok, backbone, head


def _record_backbone_calls(
    backbone: GeneratorBackbone,
) -> tuple[list[tuple[int, ...]], torch.utils.hooks.RemovableHandle]:
    """Record the input shape of every backbone forward pass."""
    shapes: list[tuple[int, ...]] = []

    def hook(module, args, kwargs, output) -> None:
        x = args[0] if args else kwargs["x"]
        shapes.append(tuple(x.shape))

    handle = backbone.register_forward_hook(hook, with_kwargs=True)
    return shapes, handle


@pytest.mark.parametrize("batch", [1, 3])
def test_exactly_k_backbone_calls(batch: int) -> None:
    # Acceptance: exactly K backbone forward passes per batch.
    tok, backbone, head = _pipeline("vq", "ce")
    shapes, handle = _record_backbone_calls(backbone)
    try:
        tokens = sample_tokens(
            tok, backbone, head, batch, generator=torch.Generator().manual_seed(0)
        )
    finally:
        handle.remove()

    assert len(shapes) == K, f"expected {K} backbone calls, saw {len(shapes)}"
    for k, (shape, length) in enumerate(zip(shapes, SCHEDULE), start=1):
        assert shape == (batch, length, H), f"level {k} backbone input {shape}"
    assert tokens.shape == (batch, L)


@pytest.mark.parametrize("k_levels", [1, K])
def test_call_count_for_other_level_counts(k_levels: int) -> None:
    tok, backbone, head = _pipeline("vq", "ce", K_override=k_levels)
    shapes, handle = _record_backbone_calls(backbone)
    try:
        sample_tokens(tok, backbone, head, 2, generator=torch.Generator().manual_seed(1))
    finally:
        handle.remove()
    assert len(shapes) == k_levels


@pytest.mark.parametrize(
    "quantizer,head_name",
    [("vq", "ce"), ("noisy", "regress"), ("noisy", "flow")],
)
def test_shapes_and_token_range(quantizer: str, head_name: str) -> None:
    # Acceptance: correct shapes, token ids in [0, V).
    tok, backbone, head = _pipeline(quantizer, head_name)
    tokens = sample_tokens(
        tok, backbone, head, 4, generator=torch.Generator().manual_seed(3)
    )
    assert tokens.shape == (4, L)
    assert tokens.dtype in (torch.long, torch.int64)
    assert int(tokens.min()) >= 0
    assert int(tokens.max()) < V


@pytest.mark.parametrize(
    "quantizer,head_name",
    [("vq", "ce"), ("noisy", "regress"), ("noisy", "flow")],
)
def test_same_seed_same_samples(quantizer: str, head_name: str) -> None:
    # Acceptance: same seed gives the same samples.
    tok, backbone, head = _pipeline(quantizer, head_name)
    first = sample_tokens(
        tok, backbone, head, 2, generator=torch.Generator().manual_seed(7)
    )
    second = sample_tokens(
        tok, backbone, head, 2, generator=torch.Generator().manual_seed(7)
    )
    other = sample_tokens(
        tok, backbone, head, 2, generator=torch.Generator().manual_seed(8)
    )
    assert torch.equal(first, second), "same seed produced different samples"
    assert not torch.equal(first, other), "different seed produced identical samples"


def test_matches_manual_spec_loop() -> None:
    # The sampler must reproduce the SPEC 5.6 pseudocode step by step.
    tok, backbone, head = _pipeline("noisy", "flow")
    assert isinstance(head, FlowHead)

    tokens = sample_tokens(
        tok, backbone, head, 2, generator=torch.Generator().manual_seed(11)
    )

    gen = torch.Generator().manual_seed(11)
    with torch.no_grad():
        zhat = tok.tok_emb.weight.new_zeros((2, tok.L, tok.d))
        for k in range(1, tok.K + 1):
            h = backbone(backbone.embed_level(tok, zhat, k))
            code = head.sample_level(h, k, generator=gen)
            zhat = zhat + tok.contrib(k, code)
        manual = tok.decode(zhat).argmax(dim=-1)
    assert torch.equal(tokens, manual)


def test_rejects_inconsistent_modules() -> None:
    tok, backbone, _ = _pipeline("vq", "ce")
    mismatched = build_head(
        GeneratorConfig(H=H, n_layers=1, n_heads=2, dropout=0.0, head="ce"),
        H=H,
        K=K + 1,
        V_k=64,
    )
    with pytest.raises(ValueError):
        sample_tokens(tok, backbone, mismatched, 1)
    _, _, head = _pipeline("vq", "ce")
    with pytest.raises(ValueError):
        sample_tokens(tok, backbone, head, 0)


def test_sampler_does_not_touch_module_modes() -> None:
    # ASSUMPTION: the sampler never flips train/eval; the caller decides.
    tok, backbone, head = _pipeline("vq", "ce")
    tok.train()
    sample_tokens(tok, backbone, head, 1, generator=torch.Generator().manual_seed(0))
    assert tok.training is True
    assert backbone.training is False and head.training is False


def test_heads_bind_to_their_own_sampler() -> None:
    _, _, regress = _pipeline("noisy", "regress")
    _, _, flow = _pipeline("noisy", "flow")
    assert isinstance(regress, RegressHead)
    assert isinstance(flow, FlowHead)

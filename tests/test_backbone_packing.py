"""T4.7 / SPEC 5.2 (v1.6): complementary levels share one packed sequence.

`forward_levels` used to pad every level to max(l_k) and run B*K sequences; at the dev
size only 774 of 1536 slots were real, so about half of a training step was padding
(docs/PLAN_2026-11-20.md). Level k and level K+1-k now share one sequence with a
two-block attention mask. Attention stays inside a level, so with dropout off the packed
layout must reproduce the padded one (kept as `layout="padded"`) to float precision,
for outputs and for every gradient, through the backbone alone and through the real
staged and joint losses.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.flop_counter import FlopCounterMode

from jncm.config import load_config
from jncm.generator.backbone import LevelBlockTransformer
from jncm.generator.packing import pair_groups as _pair_groups
from jncm.jepa.joint import joint_loss
from jncm.levels import get_level_schedule
from jncm.tokenizer import MultiscaleTokenizer
from jncm.train_generator import build_generator, compute_generator_loss

SMOKE = get_level_schedule(32, 4)  # [1, 11, 22, 32]
DEV = get_level_schedule(128, 12)  # [1, 13, ..., 128]
ODD = get_level_schedule(32, 5)  # [1, 9, 17, 24, 32]: the middle level has no partner
UNEVEN = [1, 3, 5, 10]  # pairs of 11 and 8: padding inside a packed sequence


def _transformer(H: int = 32, n_layers: int = 2, seed: int = 0) -> LevelBlockTransformer:
    torch.manual_seed(seed)
    return LevelBlockTransformer(H, n_layers=n_layers, n_heads=2, dropout=0.1).eval()


def _inputs(lengths, H: int, batch: int = 3, seed: int = 1) -> list[torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return [torch.randn(batch, length, H, generator=gen) for length in lengths]


# --------------------------------------------------------------------------- #
# Grouping
# --------------------------------------------------------------------------- #


def test_dev_schedule_packs_into_six_sequences_of_129() -> None:
    groups = _pair_groups(DEV)

    assert groups == [(0, 11), (1, 10), (2, 9), (3, 8), (4, 7), (5, 6)]
    assert {sum(DEV[i] for i in group) for group in groups} == {129}


def test_smoke_schedule_packs_into_two_sequences_of_33() -> None:
    assert _pair_groups(SMOKE) == [(0, 3), (1, 2)]
    assert [sum(SMOKE[i] for i in g) for g in _pair_groups(SMOKE)] == [33, 33]


def test_an_odd_level_count_leaves_the_middle_level_alone() -> None:
    groups = _pair_groups(ODD)

    assert sorted(groups) == [(0, 4), (1, 3), (2,)]


def test_no_grouping_when_pairing_saves_nothing() -> None:
    assert _pair_groups([10, 10]) is None
    assert _pair_groups([7]) is None


# --------------------------------------------------------------------------- #
# Equivalence with the padded layout (dropout off)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("lengths", [SMOKE, DEV, ODD, UNEVEN], ids=["smoke", "dev", "odd", "uneven"])
def test_packed_outputs_equal_the_padded_layout(lengths) -> None:
    model = _transformer()
    inputs = _inputs(lengths, model.H)

    packed = model.forward_levels(inputs, layout="pairs")
    padded = model.forward_levels(inputs, layout="padded")

    assert [tuple(t.shape) for t in packed] == [(3, length, model.H) for length in lengths]
    for level, (a, b) in enumerate(zip(packed, padded)):
        assert torch.allclose(a, b, atol=1e-5), (level, float((a - b).abs().max()))


@pytest.mark.parametrize("lengths", [SMOKE, ODD, UNEVEN], ids=["smoke", "odd", "uneven"])
def test_packed_gradients_equal_the_padded_layout(lengths) -> None:
    model = _transformer()
    gen = torch.Generator().manual_seed(5)
    weights = [torch.randn(3, length, model.H, generator=gen) for length in lengths]

    def grads(layout: str) -> list[torch.Tensor]:
        inputs = [t.requires_grad_(True) for t in _inputs(lengths, model.H)]
        outputs = model.forward_levels(inputs, layout=layout)
        loss = sum((out * w).sum() for out, w in zip(outputs, weights))
        return list(torch.autograd.grad(loss, list(model.parameters()) + inputs))

    for index, (a, b) in enumerate(zip(grads("pairs"), grads("padded"))):
        assert torch.allclose(a, b, atol=1e-5), (index, float((a - b).abs().max()))


def test_values_past_a_levels_length_never_enter_the_packed_pass() -> None:
    model = _transformer()
    clean = _inputs(SMOKE, model.H)
    gen = torch.Generator().manual_seed(9)
    garbage = [
        torch.cat([t, 100.0 * torch.randn(3, 32 - t.shape[1], model.H, generator=gen)], dim=1)
        for t in clean
    ]

    with_garbage = model.forward_levels(garbage, lengths=SMOKE, layout="pairs")
    reference = model.forward_levels(clean, layout="padded")

    for a, b in zip(with_garbage, reference):
        assert torch.allclose(a, b, atol=1e-5)


def test_padding_inside_a_packed_sequence_stays_finite_in_both_directions() -> None:
    """A padding row that could attend to nothing would be NaN, and 0 * NaN poisons the
    weight gradients even though the row is discarded."""
    model = _transformer().train()
    inputs = [t.requires_grad_(True) for t in _inputs(UNEVEN, model.H)]

    outputs = model.forward_levels(inputs, layout="pairs")
    sum(out.sum() for out in outputs).backward()

    assert all(torch.isfinite(out).all() for out in outputs)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())


def test_dropout_on_gives_the_right_shapes() -> None:
    """With dropout on the two layouts draw different masks, so only shapes and
    finiteness are comparable."""
    model = _transformer().train()

    outputs = model.forward_levels(_inputs(DEV, model.H), layout="pairs")

    assert [tuple(t.shape) for t in outputs] == [(3, length, model.H) for length in DEV]
    assert all(torch.isfinite(t).all() for t in outputs)


def test_an_unknown_layout_is_rejected() -> None:
    model = _transformer()

    with pytest.raises(ValueError, match="layout"):
        model.forward_levels(_inputs(SMOKE, model.H), layout="diagonal")


# --------------------------------------------------------------------------- #
# Cost
# --------------------------------------------------------------------------- #


def test_packing_about_halves_the_flops_at_the_smoke_size() -> None:
    model = _transformer().train()

    def flops(layout: str) -> int:
        inputs = [t.requires_grad_(True) for t in _inputs(SMOKE, model.H)]
        with FlopCounterMode(display=False) as counter:
            outputs = model.forward_levels(inputs, layout=layout)
            sum(out.sum() for out in outputs).backward()
        return counter.get_total_flops()

    assert flops("pairs") <= 0.55 * flops("padded")


# --------------------------------------------------------------------------- #
# Through the real losses (dropout 0)
# --------------------------------------------------------------------------- #


def _staged(quantizer: str, head: str):
    cfg = load_config("configs/smoke.yaml")
    cfg.device = "cpu"
    cfg.tokenizer.quantizer = quantizer
    cfg.generator.head = head
    cfg.generator.dropout = 0.0
    torch.manual_seed(0)
    tokenizer = MultiscaleTokenizer(cfg.tokenizer, L=cfg.L, V=cfg.V).eval()
    for parameter in tokenizer.parameters():
        parameter.requires_grad_(False)
    backbone, head_module = build_generator(cfg, tokenizer)
    return cfg, tokenizer, backbone, head_module


@pytest.mark.parametrize("quantizer,head", [("vq", "ce"), ("noisy", "flow"), ("noisy", "regress")])
def test_staged_loss_and_gradients_do_not_depend_on_the_layout(quantizer: str, head: str) -> None:
    cfg, tokenizer, backbone, head_module = _staged(quantizer, head)
    x = torch.randint(0, cfg.V, (4, cfg.L), generator=torch.Generator().manual_seed(3))
    params = list(backbone.parameters()) + list(head_module.parameters())

    def run(layout: str):
        backbone.level_layout = layout
        loss, _ = compute_generator_loss(
            tokenizer, backbone, head_module, x, cfg, generator=torch.Generator().manual_seed(7)
        )
        return loss, torch.autograd.grad(loss, params, allow_unused=True)

    loss_pairs, grads_pairs = run("pairs")
    loss_padded, grads_padded = run("padded")

    assert torch.allclose(loss_pairs, loss_padded, atol=1e-5)
    for a, b in zip(grads_pairs, grads_padded):
        if a is None or b is None:
            assert a is None and b is None
            continue
        assert torch.allclose(a, b, atol=1e-5), float((a - b).abs().max())


def test_joint_loss_and_gradients_do_not_depend_on_the_layout() -> None:
    cfg = load_config("configs/smoke.yaml")
    cfg.device = "cpu"
    cfg.tokenizer.quantizer = "noisy"
    cfg.generator.head = "flow"
    cfg.generator.dropout = 0.0
    torch.manual_seed(0)
    tokenizer = MultiscaleTokenizer(cfg.tokenizer, L=cfg.L, V=cfg.V)
    backbone, head_module = build_generator(cfg, tokenizer)
    x = torch.randint(0, cfg.V, (4, cfg.L), generator=torch.Generator().manual_seed(3))
    params = (
        list(tokenizer.parameters()) + list(backbone.parameters()) + list(head_module.parameters())
    )

    def run(layout: str):
        backbone.level_layout = layout
        step = joint_loss(
            cfg, tokenizer, backbone, head_module, x,
            generator=torch.Generator().manual_seed(7), cut=tokenizer.K,
        )
        return step.loss, torch.autograd.grad(step.loss, params, allow_unused=True)

    loss_pairs, grads_pairs = run("pairs")
    loss_padded, grads_padded = run("padded")

    assert torch.allclose(loss_pairs, loss_padded, atol=1e-5)
    for a, b in zip(grads_pairs, grads_padded):
        if a is None or b is None:
            assert a is None and b is None
            continue
        assert torch.allclose(a, b, atol=1e-5), float((a - b).abs().max())


def test_the_backbone_defaults_to_the_packed_layout() -> None:
    _, _, backbone, _ = _staged("vq", "ce")

    assert backbone.level_layout == "pairs"

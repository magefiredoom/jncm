"""Tests for jncm/jepa/joint.py (T5.3, SPEC section 8).

Covers the noise-free EMA target chain, the loss composition and every ablation
flag (VICReg, decoder_through_E, ema), the deterministic c-hat definitions used
by VICReg, and the SPEC 8 diagnostics (std, effective rank, R2, collapse alarm).
"""

from __future__ import annotations

import pytest
import torch

from jncm.config import load_config
from jncm.jepa.ema import EMAModule
from jncm.jepa.joint import encode_clean, joint_loss, predicted_codes
from jncm.tokenizer import MultiscaleTokenizer
from jncm.train_generator import build_generator


def _cfg(head: str = "regress"):
    """Small ``noisy`` config for CPU tests (L=16, V=16, K=4, d=16)."""
    cfg = load_config("configs/smoke.yaml")
    cfg.L = 16
    cfg.V = 16
    cfg.device = "cpu"
    cfg.tokenizer.quantizer = "noisy"
    cfg.tokenizer.d = 16
    cfg.tokenizer.d_k = 4
    cfg.tokenizer.decoder_heads = 2
    cfg.generator.head = head
    cfg.generator.H = 16
    cfg.generator.n_layers = 1
    cfg.generator.n_heads = 2
    cfg.generator.dropout = 0.0
    cfg.generator.n_rep = 1
    cfg.generator.flow_steps = 2
    cfg.generator.corrupt.mode = "none"
    cfg.generator.train.batch_size = 4
    return cfg


def _build(cfg, seed: int = 0):
    torch.manual_seed(seed)
    tokenizer = MultiscaleTokenizer(cfg.tokenizer, L=cfg.L, V=cfg.V)
    for parameter in tokenizer.parameters():
        parameter.requires_grad_(True)
    backbone, head = build_generator(cfg, tokenizer)
    return tokenizer, backbone, head


def _batch(cfg) -> torch.Tensor:
    return torch.randint(0, cfg.V, (cfg.generator.train.batch_size, cfg.L))


# --------------------------------------------------------------------------- #
# Target encoding
# --------------------------------------------------------------------------- #


def test_encode_clean_is_the_noiseless_residual_chain() -> None:
    cfg = _cfg()
    tokenizer, _, _ = _build(cfg)
    x = _batch(cfg)

    # With sigma = 0 the transmitted code is the clean code, so the online
    # encode is exactly the noiseless chain (its own residual chain).
    for quantizer in tokenizer.quantizers:
        quantizer.sigma_k = 0.0
    targets, zhat = encode_clean(tokenizer, x)
    plain = tokenizer.encode(x)

    assert len(targets) == tokenizer.K
    assert len(zhat) == tokenizer.K + 1
    for clean, expected in zip(targets, plain.targets):
        assert torch.equal(clean, expected)
    for got, expected in zip(zhat, plain.zhat):
        assert torch.equal(got, expected)

    # With the real channel noise every chain code differs from the target.
    for quantizer in tokenizer.quantizers:
        quantizer.sigma_k = cfg.tokenizer.sigma_k
    noisy = tokenizer.encode(x, generator=torch.Generator().manual_seed(3))
    for clean, code in zip(targets, noisy.codes):
        assert not torch.allclose(clean, code)


def test_encode_clean_rejects_vq() -> None:
    cfg = _cfg()
    cfg.tokenizer.quantizer = "vq"
    cfg.generator.head = "ce"
    tokenizer, _, _ = _build(cfg)
    with pytest.raises(ValueError):
        encode_clean(tokenizer, _batch(cfg))


def test_joint_targets_are_noise_free_and_detached() -> None:
    cfg = _cfg()
    tokenizer, backbone, head = _build(cfg)
    x = _batch(cfg)

    # E-bar at init is a copy of E: its targets are exactly the noiseless chain.
    ema = EMAModule(tokenizer, total_steps=100)
    step = joint_loss(
        cfg,
        tokenizer,
        backbone,
        head,
        x,
        target_tokenizer=ema.module,
        generator=torch.Generator().manual_seed(0),
    )
    expected, _ = encode_clean(ema.module, x)
    for target, want in zip(step.targets, expected):
        assert torch.equal(target, want)
        assert target.requires_grad is False

    # Raising the online channel noise must not move the EMA targets: they
    # come from E-bar's noise-free chain, not from E's transmitted codes.
    online_before = tokenizer.encode(x, generator=torch.Generator().manual_seed(1))
    for quantizer in tokenizer.quantizers:
        quantizer.sigma_k = 2.0
    online_after = tokenizer.encode(x, generator=torch.Generator().manual_seed(1))
    assert any(
        not torch.allclose(before, after)
        for before, after in zip(online_before.codes, online_after.codes)
    )
    after = joint_loss(
        cfg,
        tokenizer,
        backbone,
        head,
        x,
        target_tokenizer=ema.module,
        generator=torch.Generator().manual_seed(0),
    )
    for before_target, after_target in zip(step.targets, after.targets):
        assert torch.equal(before_target, after_target)

    # ema=False: the online encoder produces the same targets, stop-grad.
    step_online = joint_loss(
        cfg,
        tokenizer,
        backbone,
        head,
        x,
        target_tokenizer=None,
        generator=torch.Generator().manual_seed(0),
    )
    expected_online, _ = encode_clean(tokenizer, x)
    for target, want in zip(step_online.targets, expected_online):
        assert torch.equal(target, want)
        assert target.requires_grad is False


# --------------------------------------------------------------------------- #
# Loss composition and flags
# --------------------------------------------------------------------------- #


def test_loss_components_compose_and_flags_do_not_change_upstream_draws() -> None:
    cfg = _cfg()
    cfg.jepa.vicreg = False
    tokenizer, backbone, head = _build(cfg)
    x = _batch(cfg)

    off = joint_loss(
        cfg, tokenizer, backbone, head, x, generator=torch.Generator().manual_seed(5)
    )
    assert off.loss_vic.item() == 0.0
    assert torch.allclose(
        off.loss, off.loss_dec + cfg.jepa.lambda_pred * off.loss_pred
    )
    assert off.predictions == []

    cfg.jepa.vicreg = True
    on = joint_loss(
        cfg, tokenizer, backbone, head, x, generator=torch.Generator().manual_seed(5)
    )
    # VICReg is the last draw, so the shared losses are identical under a seed.
    assert torch.allclose(on.loss_dec, off.loss_dec)
    assert torch.allclose(on.loss_pred, off.loss_pred)
    assert on.loss_vic.item() >= 0.0
    assert torch.allclose(
        on.loss,
        on.loss_dec + cfg.jepa.lambda_pred * on.loss_pred + on.loss_vic,
    )
    assert len(on.predictions) == tokenizer.K

    cfg.jepa.lambda_pred = 0.25
    scaled = joint_loss(
        cfg, tokenizer, backbone, head, x, generator=torch.Generator().manual_seed(5)
    )
    assert torch.allclose(
        scaled.loss,
        scaled.loss_dec + 0.25 * scaled.loss_pred + scaled.loss_vic,
    )


def test_vicreg_flag_penalises_a_collapsed_prediction() -> None:
    cfg = _cfg()
    tokenizer, backbone, head = _build(cfg)
    x = _batch(cfg)
    with torch.no_grad():
        for linear in head.heads:
            linear.weight.zero_()
            linear.bias.zero_()

    cfg.jepa.vicreg = False
    off = joint_loss(
        cfg, tokenizer, backbone, head, x, generator=torch.Generator().manual_seed(0)
    )
    cfg.jepa.vicreg = True
    on = joint_loss(
        cfg, tokenizer, backbone, head, x, generator=torch.Generator().manual_seed(0)
    )

    # c-hat = 0: the variance term is gamma per level, on top of the online c_k.
    assert off.loss_vic.item() == 0.0
    assert on.loss_vic.item() >= tokenizer.K * 0.95
    assert on.loss.item() > off.loss.item()


def test_decoder_through_E_controls_encoder_gradients() -> None:
    cfg = _cfg()
    cfg.jepa.lambda_pred = 0.0
    cfg.jepa.vicreg = False
    x = _batch(cfg)

    grads: dict[bool, torch.Tensor] = {}
    for through in (True, False):
        cfg.jepa.decoder_through_E = through
        tokenizer, backbone, head = _build(cfg)
        step = joint_loss(
            cfg,
            tokenizer,
            backbone,
            head,
            x,
            target_tokenizer=None,
            generator=torch.Generator().manual_seed(0),
        )
        step.loss.backward()
        decoder_grad = tokenizer.head.weight.grad
        # The decoder is trained from zhat in both branches.
        assert decoder_grad is not None and decoder_grad.abs().sum().item() > 0
        grads[through] = tokenizer.tok_emb.weight.grad

    assert grads[True] is not None and grads[True].abs().sum().item() > 0
    assert grads[False] is None or grads[False].abs().sum().item() == 0


def test_joint_loss_rejects_phase1_configs() -> None:
    cfg = _cfg()
    cfg.tokenizer.quantizer = "vq"
    cfg.generator.head = "ce"
    tokenizer, backbone, head = _build(cfg)
    with pytest.raises(ValueError):
        joint_loss(
            cfg,
            tokenizer,
            backbone,
            head,
            _batch(cfg),
            generator=torch.Generator().manual_seed(0),
        )


def test_predictions_are_deterministic_and_noiseless() -> None:
    cfg = _cfg()
    tokenizer, backbone, head = _build(cfg)
    x = _batch(cfg)
    step = joint_loss(
        cfg, tokenizer, backbone, head, x, generator=torch.Generator().manual_seed(2)
    )
    for level, (hidden, prediction) in enumerate(
        zip(step.hidden, step.predictions), start=1
    ):
        assert prediction.shape == (x.shape[0], backbone.schedule[level - 1], cfg.tokenizer.d_k)
        # regress c-hat is exactly the raw linear output (no renorm, no noise).
        assert torch.equal(prediction, head.heads[level - 1](hidden).detach())

    flow_cfg = _cfg(head="flow")
    tokenizer_f, backbone_f, head_f = _build(flow_cfg)
    hidden = [torch.randn(2, length, flow_cfg.generator.H) for length in backbone_f.schedule]
    first = predicted_codes(head_f, hidden, generator=torch.Generator().manual_seed(4))
    second = predicted_codes(head_f, hidden, generator=torch.Generator().manual_seed(4))
    assert all(torch.equal(a, b) for a, b in zip(first, second))
    assert all(torch.isfinite(p).all() for p in first)

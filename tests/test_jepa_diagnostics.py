"""Tests for jncm/jepa/diagnostics.py (T5.3, SPEC section 8).

Covers the effective-rank and R2 definitions, the per-level std/rank diagnostics
record shapes, and the collapse-alarm patience/reset semantics.
"""

from __future__ import annotations

import math

import pytest
import torch

from jncm.config import load_config
from jncm.jepa.diagnostics import (
    CollapseAlarm,
    code_diagnostics,
    effective_rank,
    explained_variance,
)
from jncm.jepa.joint import joint_loss
from jncm.tokenizer import MultiscaleTokenizer
from jncm.train_generator import build_generator


def _cfg():
    """Small ``noisy`` config for CPU tests (L=16, V=16, K=4, d=16)."""
    cfg = load_config("configs/smoke.yaml")
    cfg.L = 16
    cfg.V = 16
    cfg.device = "cpu"
    cfg.tokenizer.quantizer = "noisy"
    cfg.tokenizer.d = 16
    cfg.tokenizer.d_k = 4
    cfg.tokenizer.decoder_heads = 2
    cfg.generator.head = "regress"
    cfg.generator.H = 16
    cfg.generator.n_layers = 1
    cfg.generator.n_heads = 2
    cfg.generator.dropout = 0.0
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


def test_effective_rank_endpoints() -> None:
    torch.manual_seed(0)
    isotropic = torch.randn(4096, 8)
    assert 7.0 < effective_rank(isotropic) <= 8.0
    assert effective_rank(torch.full((64, 5), 2.0)) == pytest.approx(1.0, abs=1e-6)

    torch.manual_seed(1)
    direction = torch.randn(2048, 1)
    rank_one = direction * torch.randn(1, 6)
    assert effective_rank(rank_one) < 1.2


def test_explained_variance_definition() -> None:
    torch.manual_seed(2)
    target = torch.randn(8, 4, 3)
    assert explained_variance(target, target) == pytest.approx(1.0)
    zeros = torch.zeros_like(target)
    # Zero mean, unit variance target: MSE = Var, so R2 = 0.
    assert explained_variance(zeros, target) == pytest.approx(0.0, abs=0.15)
    constant = torch.full((4, 2, 3), 1.5)
    assert explained_variance(torch.zeros_like(constant), constant) == 0.0
    worse = explained_variance(target + 1.0, target)
    assert worse < 1.0


def test_code_diagnostics_shapes() -> None:
    cfg = _cfg()
    tokenizer, backbone, head = _build(cfg)
    batch = torch.randint(0, cfg.V, (cfg.generator.train.batch_size, cfg.L))
    step = joint_loss(
        cfg, tokenizer, backbone, head, batch,
        generator=torch.Generator().manual_seed(0),
    )

    diag = code_diagnostics(step.codes, step.targets, step.predictions)

    assert set(diag) == {
        "std_min",
        "std_mean",
        "target_std_min",
        "target_std_mean",
        "eff_rank",
        "target_eff_rank",
        "r2",
    }
    for values in diag.values():
        assert len(values) == tokenizer.K
        assert all(math.isfinite(value) for value in values)
    # Effective rank lives in [1, d_k]; LayerNorm codes have per-dim std ~1.
    assert all(1.0 - 1e-6 <= value <= cfg.tokenizer.d_k for value in diag["eff_rank"])
    assert all(0.0 <= value for value in diag["std_min"])
    assert all(0.0 < value < 1.5 for value in diag["std_mean"])


def test_collapse_alarm_patience_and_reset() -> None:
    alarm = CollapseAlarm(patience=3)
    assert alarm.update([0.05, 1.0]) == [False, False]
    assert alarm.update([0.05, 1.0]) == [False, False]
    assert alarm.update([0.05, 1.0]) == [True, False]
    # A single healthy step resets the counter.
    assert alarm.update([0.5, 1.0]) == [False, False]
    assert alarm.counts == [0, 0]
    assert alarm.update([0.05, 0.02]) == [False, False]
    assert alarm.counts == [1, 1]

    state = alarm.state_dict()
    restored = CollapseAlarm(patience=3)
    restored.load_state_dict(state)
    assert restored.counts == [1, 1]

    assert CollapseAlarm().patience == 200

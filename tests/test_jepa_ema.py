"""Tests for jncm/jepa/ema.py (T5.1, SPEC 8 step 8)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from jncm.jepa.ema import EMAModule, cosine_momentum


class Tiny(nn.Module):
    """One 1x2 parameter plus an integer buffer, for exact hand arithmetic."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[1.0, -2.0]]))
        self.register_buffer("count", torch.zeros((), dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.t()


def test_update_matches_a_hand_computed_value() -> None:
    online = Tiny()
    ema = EMAModule(online, total_steps=1000)
    with torch.no_grad():
        online.weight.copy_(torch.tensor([[2.0, 3.0]]))

    momentum = ema.update(online, step=0)

    expected = 0.996 * torch.tensor([[1.0, -2.0]]) + 0.004 * torch.tensor(
        [[2.0, 3.0]]
    )
    assert momentum == pytest.approx(0.996)
    assert torch.allclose(ema.module.weight, expected, atol=1e-6)


def test_target_has_no_grad_and_is_independent_of_the_online_module() -> None:
    online = Tiny()
    ema = EMAModule(online, total_steps=10)

    assert all(not param.requires_grad for param in ema.module.parameters())
    assert ema.module.weight.data_ptr() != online.weight.data_ptr()
    before = ema.module.weight.detach().clone()
    with torch.no_grad():
        online.weight.add_(10.0)
    assert torch.equal(ema.module.weight, before)

    loss = ema.module(torch.ones(1, 2)).sum()
    assert loss.requires_grad is False


def test_target_is_eval_mode_and_stays_eval_after_parent_train() -> None:
    class Parent(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.online = nn.Sequential(
                nn.Linear(4, 4), nn.Dropout(0.5), nn.BatchNorm1d(4)
            )
            self.ema = EMAModule(self.online, total_steps=10)

    parent = Parent()

    parent.train()

    assert parent.online.training is True
    assert parent.ema.module.training is False
    assert all(not module.training for module in parent.ema.module.modules())
    parent.ema.train(True)
    assert parent.ema.module.training is False
    assert all(not module.training for module in parent.ema.module.modules())


def test_schedule_endpoints_and_monotonicity() -> None:
    assert cosine_momentum(0, 1000) == pytest.approx(0.996)
    assert cosine_momentum(1000, 1000) == pytest.approx(1.0)
    assert cosine_momentum(5000, 1000) == pytest.approx(1.0)
    assert cosine_momentum(500, 1000) == pytest.approx(0.998)

    values = [cosine_momentum(step, 1000) for step in range(1001)]
    assert all(0.996 <= value <= 1.0 for value in values)
    assert all(
        later >= earlier for earlier, later in zip(values, values[1:])
    )

    with pytest.raises(ValueError):
        cosine_momentum(0, 0)


def test_update_uses_the_scheduled_momentum() -> None:
    online = Tiny()
    ema = EMAModule(online, total_steps=1000)
    with torch.no_grad():
        online.weight.copy_(torch.tensor([[2.0, 3.0]]))
    momentum = cosine_momentum(500, 1000)

    returned = ema.update(online, step=500)

    expected = momentum * torch.tensor([[1.0, -2.0]]) + (1.0 - momentum) * (
        torch.tensor([[2.0, 3.0]])
    )
    assert returned == pytest.approx(momentum)
    assert torch.allclose(ema.module.weight, expected, atol=1e-6)

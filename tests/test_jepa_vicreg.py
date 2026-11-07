"""Tests for jncm/jepa/vicreg.py (T5.2, SPEC 8 step 6)."""

from __future__ import annotations

import pytest
import torch

from jncm.jepa.vicreg import covariance_term, variance_term, vicreg_loss


def test_iid_standard_normal_gives_near_zero_var_and_small_cov() -> None:
    torch.manual_seed(0)
    x = torch.randn(4096, 8)

    var = variance_term(x)
    cov = covariance_term(x)

    assert var.item() < 0.02
    assert cov.item() < 0.01


def test_a_constant_gives_var_equal_to_gamma() -> None:
    x = torch.full((64, 5), 2.5)

    # sqrt(eps) with the default eps leaves 1 - 1e-4; eps=0 is exactly gamma.
    assert variance_term(x).item() == pytest.approx(1.0, abs=1e-2)
    assert variance_term(x, eps=0.0).item() == pytest.approx(1.0, abs=1e-6)
    assert covariance_term(x).item() == pytest.approx(0.0, abs=1e-12)


def test_correlated_dimensions_give_cov_greater_than_zero() -> None:
    torch.manual_seed(1)
    first = torch.randn(2048, 1)
    x = torch.cat(
        [
            first,
            first + 0.01 * torch.randn(2048, 1),
            torch.randn(2048, 3),
        ],
        dim=1,
    )

    assert covariance_term(x).item() > 0.1


def test_gradients_flow_through_both_terms() -> None:
    torch.manual_seed(2)
    x = torch.randn(256, 4, requires_grad=True)

    loss = variance_term(x) + covariance_term(x)

    assert loss.requires_grad
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum().item() > 0


def test_a_collapsed_dimension_keeps_a_finite_gradient() -> None:
    x = torch.full((32, 4), 1.5, requires_grad=True)

    variance_term(x).backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_vicreg_loss_weights_the_two_terms() -> None:
    torch.manual_seed(3)
    x = torch.randn(512, 6)

    expected = 1.0 * variance_term(x) + 0.04 * covariance_term(x)

    assert vicreg_loss(x).item() == pytest.approx(expected.item(), rel=1e-6)

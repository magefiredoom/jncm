"""Tests for level schedule calculation, downsampling, and upsampling."""

import pytest
import torch
from jncm.levels import down, get_level_schedule, up


def test_level_schedule_smoke() -> None:
    # L=32, K=4 gives [1, 11, 22, 32] (sum 66)
    sched = get_level_schedule(32, 4)
    assert sched == [1, 11, 22, 32]
    assert sum(sched) == 66
    assert sched[0] == 1
    assert sched[-1] == 32


def test_level_schedule_dev() -> None:
    # L=128, K=12 gives [1, 13, 24, 36, 47, 59, 70, 82, 93, 105, 116, 128] (sum 774)
    sched = get_level_schedule(128, 12)
    assert sched == [1, 13, 24, 36, 47, 59, 70, 82, 93, 105, 116, 128]
    assert sum(sched) == 774
    assert sched[0] == 1
    assert sched[-1] == 128


def test_level_schedule_various_configurations() -> None:
    for L, K in [(16, 2), (64, 5), (100, 10), (256, 16)]:
        sched = get_level_schedule(L, K)
        assert len(sched) == K
        assert sched[0] == 1
        assert sched[-1] == L
        # Strictly increasing
        assert all(sched[i] < sched[i + 1] for i in range(K - 1))


def test_down_identities_and_properties() -> None:
    B, L, C = 4, 32, 16
    x = torch.randn(B, L, C)

    # down(x, L) is identity
    x_id = down(x, L)
    assert torch.equal(x, x_id)

    # down(x, 1) equals mean over position dimension
    x_mean = down(x, 1)
    expected_mean = x.mean(dim=1, keepdim=True)
    assert torch.allclose(x_mean, expected_mean, atol=1e-6)
    assert x_mean.shape == (B, 1, C)

    # intermediate lengths have correct shapes
    for l_k in [1, 11, 22, 32]:
        out = down(x, l_k)
        assert out.shape == (B, l_k, C)


def test_up_identities_and_properties() -> None:
    B, L, C = 4, 32, 16

    # up(x, L) when l == L is identity
    x_L = torch.randn(B, L, C)
    assert torch.equal(up(x_L, L), x_L)

    # up(x, L) when l == 1 broadcasts across length L
    x_1 = torch.randn(B, 1, C)
    x_up_1 = up(x_1, L)
    assert x_up_1.shape == (B, L, C)
    for pos in range(L):
        assert torch.equal(x_up_1[:, pos, :], x_1[:, 0, :])

    # intermediate upsampling shape
    for l_k in [11, 22]:
        x_k = torch.randn(B, l_k, C)
        out = up(x_k, L)
        assert out.shape == (B, L, C)
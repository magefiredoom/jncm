"""
Multiscale level schedule, downsampling pooling, and spatial interpolation operators.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def get_level_schedule(L: int, K: int) -> list[int]:
    """
    Compute level schedule l_k for k = 1..K (SPEC Section 2).

    Formula: l_k = round(1 + (k - 1) * (L - 1) / (K - 1))
    Must satisfy l_1 = 1, l_K = L, and be strictly increasing.
    """
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    if L < 1:
        raise ValueError(f"L must be >= 1, got {L}")
    if K == 1:
        return [L]

    schedule: list[int] = []
    for k in range(1, K + 1):
        l_k = int(round(1.0 + (k - 1) * (L - 1) / (K - 1)))
        schedule.append(l_k)

    if schedule[0] != 1:
        raise ValueError(f"Level schedule must start at 1, got {schedule[0]}")
    if schedule[-1] != L:
        raise ValueError(f"Level schedule must end at L={L}, got {schedule[-1]}")
    for i in range(len(schedule) - 1):
        if schedule[i] >= schedule[i + 1]:
            raise ValueError(
                f"Level schedule must be strictly increasing, got {schedule}"
            )

    return schedule


def down(x: torch.Tensor, l: int) -> torch.Tensor:
    """
    Downsample representations from length T to l using 1D adaptive average pooling.

    Args:
        x: Input tensor of shape [B, T, C].
        l: Target sequence length.

    Returns:
        Tensor of shape [B, l, C].
    """
    _, T, _ = x.shape
    if T == l:
        return x
    if l == 1:
        return x.mean(dim=1, keepdim=True)

    # Adaptive pool operates on [B, C, T]
    x_trans = x.transpose(1, 2)
    pooled = F.adaptive_avg_pool1d(x_trans, l)
    return pooled.transpose(1, 2).contiguous()


def up(x: torch.Tensor, L: int) -> torch.Tensor:
    """
    Upsample representations from length l to target length L.

    For l == L: identity.
    For l == 1: broadcast along time dimension.
    For 1 < l < L: linear interpolation (align_corners=False).

    Args:
        x: Input tensor of shape [B, l, C].
        L: Target sequence length.

    Returns:
        Tensor of shape [B, L, C].
    """
    _, l, _ = x.shape
    if l == L:
        return x
    if l == 1:
        return x.expand(-1, L, -1).contiguous()

    x_trans = x.transpose(1, 2)
    interpolated = F.interpolate(x_trans, size=L, mode="linear", align_corners=False)
    return interpolated.transpose(1, 2).contiguous()
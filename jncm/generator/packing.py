"""Packing of the K levels for one backbone pass (SPEC 5.2 v1.6, TASKS T4.7).

Attention is restricted to each level, so the levels can share sequences as long as a
mask keeps them apart. Two layouts:

- ``pairs``: level k and level K+1-k share one sequence with a two-block mask. For the
  level schedule l_k + l_{K+1-k} = L + 1, so at L = 128, K = 12 six sequences of 129
  hold every position and none is padding.
- ``padded``: every level is padded to the longest and runs as its own sequence with a
  key-padding mask; at the dev size 762 of 1536 slots were padding. It is the reference
  the packed layout is tested against, and the fallback when pairing saves nothing.

Shapes: level inputs are one ``[B, >= l_k, H]`` tensor per level (values past
``lengths[k]`` are ignored); packed tensors are ``[B * groups, T, H]``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def pair_groups(lengths: Sequence[int]) -> list[tuple[int, ...]] | None:
    """Groups of level indices that share one packed sequence.

    The shortest level goes with the longest, the second shortest with the second
    longest, and so on; an odd count leaves the middle level alone. Returns ``None``
    when the groups would not use fewer slots than padding every level to the longest.
    """
    k = len(lengths)
    if k < 2:
        return None
    order = sorted(range(k), key=lambda i: (lengths[i], i))
    groups: list[tuple[int, ...]] = [
        tuple(sorted((order[j], order[k - 1 - j]))) for j in range(k // 2)
    ]
    if k % 2:
        groups.append((order[k // 2],))
    seq_len = max(sum(lengths[i] for i in group) for group in groups)
    if len(groups) * seq_len >= k * max(lengths):
        return None
    return groups


def pack_pairs(
    level_inputs: Sequence[torch.Tensor],
    lengths: Sequence[int],
    groups: Sequence[tuple[int, ...]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Packed ``[B*G, T, H]`` input and boolean attention mask ``[B*G, 1, T, T]``.

    A query attends to the valid keys of its own level. A padding slot attends only to
    itself: a row that may attend to nothing is NaN, and 0 * NaN would reach the weight
    gradients even though the row is discarded.
    """
    b, _, h = level_inputs[0].shape
    device = level_inputs[0].device
    n_groups = len(groups)
    seq_len = max(sum(lengths[i] for i in group) for group in groups)
    rows: list[torch.Tensor] = []
    segments = torch.full((n_groups, seq_len), -1, dtype=torch.long, device=device)
    for g, group in enumerate(groups):
        # Slice first, so values past a level's valid length never enter.
        parts = [level_inputs[i][:, : lengths[i], :] for i in group]
        row = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        if row.shape[1] < seq_len:
            row = F.pad(row, (0, 0, 0, seq_len - row.shape[1]))
        rows.append(row)
        start = 0
        for i in group:
            segments[g, start : start + lengths[i]] = i
            start += lengths[i]
    packed = torch.stack(rows, dim=1).reshape(b * n_groups, seq_len, h)
    same_level = segments[:, :, None] == segments[:, None, :]
    valid_key = (segments >= 0)[:, None, :]
    own_slot = torch.eye(seq_len, dtype=torch.bool, device=device)
    allowed = (same_level & valid_key) | own_slot  # [G, T, T]
    attn_mask = (
        allowed.unsqueeze(0).expand(b, -1, -1, -1).reshape(b * n_groups, 1, seq_len, seq_len)
    )
    return packed, attn_mask


def unpack_pairs(
    hidden: torch.Tensor,
    batch: int,
    lengths: Sequence[int],
    groups: Sequence[tuple[int, ...]],
) -> list[torch.Tensor]:
    """One ``[B, l_k, H]`` tensor per level, in level order, from ``pack_pairs`` output."""
    n_groups = len(groups)
    hidden = hidden.reshape(batch, n_groups, hidden.shape[1], hidden.shape[2])
    outputs: list[torch.Tensor | None] = [None] * len(lengths)
    for g, group in enumerate(groups):
        start = 0
        for i in group:
            outputs[i] = hidden[:, g, start : start + lengths[i], :]
            start += lengths[i]
    return outputs  # type: ignore[return-value]


def pack_padded(
    level_inputs: Sequence[torch.Tensor],
    lengths: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference layout: ``[B*K, T, H]`` with T the widest input, key mask ``[B*K, T]``."""
    b, _, h = level_inputs[0].shape
    max_len = max(tensor.shape[1] for tensor in level_inputs)
    padded: list[torch.Tensor] = []
    for tensor in level_inputs:
        if tensor.shape[1] < max_len:
            tensor = F.pad(tensor, (0, 0, 0, max_len - tensor.shape[1]))
        padded.append(tensor)
    k = len(padded)
    packed = torch.stack(padded, dim=1).reshape(b * k, max_len, h)
    valid = torch.arange(max_len, device=packed.device).unsqueeze(0) < (
        torch.as_tensor(lengths, device=packed.device).unsqueeze(1)
    )
    key_mask = valid.unsqueeze(0).expand(b, -1, -1).reshape(b * k, max_len)
    return packed, key_mask


def unpack_padded(
    hidden: torch.Tensor, batch: int, lengths: Sequence[int]
) -> list[torch.Tensor]:
    """One ``[B, l_k, H]`` tensor per level from ``pack_padded`` output."""
    k = len(lengths)
    hidden = hidden.reshape(batch, k, hidden.shape[1], hidden.shape[2])
    return [hidden[:, idx, : lengths[idx], :] for idx in range(k)]

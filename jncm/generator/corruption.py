"""Conditioning corruption for generator training (SPEC 5.4, TASKS T3.3).

Corruption is applied once to the whole hierarchy of chain codes before the
backbone runs; the tokenizer targets stay clean (SPEC 5.1 steps 1-2).

Modes (``generator.corrupt.mode``):
- ``none``: identity.
- ``swap``: for level j at rate p_j, replace the code at (b, i) with the code at
  a uniformly random other (batch, position) of the same level. p_j decreases
  linearly from ``p_first`` (level 1) to ``p_last`` (level K).
- ``noise``: add N(0, s_j^2) to the chain codes of level j (``noisy`` variant
  only). s_j decreases linearly from ``s_first`` to ``s_last``.

All functions are out-of-place: the input code tensors are never modified.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from jncm.config import CorruptionConfig

MODES: tuple[str, ...] = ("none", "swap", "noise")


def linear_schedule(K: int, first: float, last: float) -> list[float]:
    """Linear schedule with ``first`` at level 1 and ``last`` at level K.

    Args:
        K: Number of levels (>= 1).
        first: Value at level 1.
        last: Value at level K.

    Returns:
        K floats. For K = 1 the single level is the first level, so ``[first]``.
    """
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    if K == 1:
        return [float(first)]
    step = (float(last) - float(first)) / (K - 1)
    return [float(first) + i * step for i in range(K)]


def _check_lengths(values: Sequence[float], codes: Sequence[torch.Tensor], what: str) -> None:
    """Validate one rate/sigma per code tensor."""
    if len(values) != len(codes):
        raise ValueError(f"got {len(values)} {what} for {len(codes)} levels")
    if not codes:
        raise ValueError("codes must be non-empty")


def _rand(
    n: int,
    generator: torch.Generator | None,
    device: torch.device,
) -> torch.Tensor:
    """Uniform [0, 1) draws of shape [n], reproducible when a generator is given."""
    return torch.rand(n, generator=generator, device=device)


def swap_codes(
    codes: Sequence[torch.Tensor],
    rates: Sequence[float],
    generator: torch.Generator | None = None,
) -> list[torch.Tensor]:
    """Replace each level's codes with codes from other positions of that level.

    For every (batch, position) of level j, with probability p_j the code is
    replaced by the code of a uniformly drawn *other* (batch, position) of the
    same level, taken from the uncorrupted input.

    Args:
        codes: One tensor [B, l_j, ...] per level (indices or code vectors).
        rates: One probability p_j per level, in [0, 1].
        generator: Optional RNG for reproducible draws.

    Returns:
        New tensors with the same shapes and dtypes; a level with fewer than two
        (batch, position) slots has no other position to draw from and is
        returned unchanged.
    """
    _check_lengths(rates, codes, "rates")
    out: list[torch.Tensor] = []
    for level, (code, rate) in enumerate(zip(codes, rates), start=1):
        if not 0.0 <= float(rate) <= 1.0:
            raise ValueError(f"level {level}: swap rate must be in [0, 1], got {rate}")
        if code.dim() < 1:
            raise ValueError(f"level {level}: codes must have at least 1 dim")
        if code.shape[0] < 1:
            raise ValueError(f"level {level}: batch size must be >= 1")
        n = int(code.shape[0])
        if code.dim() >= 2:
            n *= int(code.shape[1])
        if n < 2 or float(rate) == 0.0:
            out.append(code)
            continue

        flat = code.reshape(n, *code.shape[2:])
        device = code.device
        mask = _rand(n, generator, device) < float(rate)
        own = torch.arange(n, device=device)
        # Uniform over the n-1 *other* positions: draw in [0, n-1) and skip self.
        pick = torch.randint(0, n - 1, (n,), generator=generator, device=device)
        pick = pick + (pick >= own).to(pick.dtype)
        replacement = flat[pick]
        mask_shape = (n,) + (1,) * (code.dim() - 2)
        swapped = torch.where(mask.reshape(mask_shape), replacement, flat)
        out.append(swapped.reshape(code.shape))
    return out


def add_code_noise(
    codes: Sequence[torch.Tensor],
    sigmas: Sequence[float],
    generator: torch.Generator | None = None,
) -> list[torch.Tensor]:
    """Add N(0, s_j^2) to each level's chain codes (SPEC 5.4, ``noisy`` only).

    Args:
        codes: One float tensor [B, l_j, d_k] per level.
        sigmas: One noise std s_j >= 0 per level (linear from s_first to s_last).
        generator: Optional RNG for reproducible draws.

    Returns:
        New float tensors with the same shapes and dtypes.
    """
    _check_lengths(sigmas, codes, "sigmas")
    out: list[torch.Tensor] = []
    for level, (code, sigma) in enumerate(zip(codes, sigmas), start=1):
        if float(sigma) < 0.0:
            raise ValueError(f"level {level}: noise std must be >= 0, got {sigma}")
        if not code.is_floating_point():
            raise ValueError(
                f"level {level}: noise corruption needs float chain codes, "
                f"got dtype {code.dtype}"
            )
        if float(sigma) == 0.0:
            out.append(code)
            continue
        noise = torch.randn(
            code.shape,
            generator=generator,
            device=code.device,
            dtype=code.dtype,
        )
        out.append(code + float(sigma) * noise)
    return out


def corrupt_codes(
    codes: Sequence[torch.Tensor],
    cfg: CorruptionConfig,
    generator: torch.Generator | None = None,
    quantizer: str | None = None,
) -> list[torch.Tensor]:
    """Apply the configured corruption to the whole hierarchy (SPEC 5.4).

    Args:
        codes: Chain codes, one tensor [B, l_j, ...] per level.
        cfg: ``generator.corrupt`` config (mode and schedule endpoints).
        generator: Optional RNG for reproducible draws.
        quantizer: Tokenizer variant; ``noise`` requires ``"noisy"``.

    Returns:
        One corrupted tensor per level, shapes unchanged. ``mode="none"``
        returns the input tensors unchanged.
    """
    if not codes:
        raise ValueError("codes must be non-empty")
    if cfg.mode == "none":
        return list(codes)
    if cfg.mode == "swap":
        rates = linear_schedule(len(codes), cfg.p_first, cfg.p_last)
        return swap_codes(codes, rates, generator=generator)
    if cfg.mode == "noise":
        if quantizer != "noisy":
            raise ValueError(
                "corrupt.mode='noise' is for the noisy variant only "
                f"(SPEC 5.4), got quantizer={quantizer!r}"
            )
        sigmas = linear_schedule(len(codes), cfg.s_first, cfg.s_last)
        return add_code_noise(codes, sigmas, generator=generator)
    raise ValueError(f"Unknown corrupt.mode: {cfg.mode!r} (expected one of {MODES})")

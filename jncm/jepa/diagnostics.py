"""JEPA collapse diagnostics (SPEC section 8, TASKS T5.3).

Per level, on the online clean codes ``c_k``, the EMA targets ``t_k`` and the
predictions ``c-hat_k``:

- min and mean per-dimension std of ``c_k`` and ``t_k``,
- effective rank ``exp(H(normalized covariance eigenvalues))``,
- explained variance ``R2_k = 1 - MSE_k / Var(t_k)``,
- a collapse alarm that fires after 200 consecutive steps with mean std < 0.1.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

COLLAPSE_STD = 0.1
COLLAPSE_PATIENCE = 200


def _flat_codes(codes: torch.Tensor) -> torch.Tensor:
    """Codes [B, l_k, d] (or already flat [N, d]) as [N, d] float32 vectors."""
    if codes.dim() == 2:
        return codes.detach().float()
    if codes.dim() == 3:
        return codes.detach().reshape(-1, codes.shape[-1]).float()
    raise ValueError(
        f"codes must be [B, l_k, d] or [N, d], got {tuple(codes.shape)}"
    )


def std_summary(codes: torch.Tensor) -> tuple[float, float]:
    """Min and mean per-dimension std over a level's [B, l_k, d] codes."""
    flat = _flat_codes(codes)
    stds = flat.std(dim=0, unbiased=False)
    return float(stds.min().item()), float(stds.mean().item())


def effective_rank(codes: torch.Tensor) -> float:
    """Effective rank ``exp(H(normalized covariance eigenvalues))`` (SPEC 4.5/8).

    Args:
        codes: One level's codes [B, l_k, d] (or [N, d]).

    Returns:
        A float in ``[1, d]``; a constant (zero-variance) code returns 1.0.
    """
    flat = _flat_codes(codes)
    d = flat.shape[1]
    if flat.shape[0] < 2 or d < 2:
        return 1.0
    covariance = torch.cov(flat.t())
    eigenvalues = torch.linalg.eigvalsh(covariance)
    total = float(eigenvalues.sum().item())
    if total <= 1e-12:
        return 1.0
    eigenvalues = eigenvalues.clamp(min=1e-8)
    probs = eigenvalues / eigenvalues.sum()
    entropy = -(probs * probs.log()).sum()
    return float(entropy.exp().item())


def explained_variance(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """``R2 = 1 - MSE / Var(t)`` over all elements of one level (SPEC 8).

    Args:
        prediction: ``c-hat_k`` [B, l_k, d].
        target: ``t_k`` [B, l_k, d].

    Returns:
        R2 as a float; 0.0 when the target has no variance (undefined ratio).
    """
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target shapes differ: {tuple(prediction.shape)} "
            f"vs {tuple(target.shape)}"
        )
    pred = prediction.detach().float()
    tgt = target.detach().float()
    mse = float(torch.mean((pred - tgt) ** 2).item())
    variance = float(tgt.var(unbiased=False).item())
    if variance <= 0.0:
        return 0.0
    return 1.0 - mse / variance


def code_diagnostics(
    codes: Sequence[torch.Tensor],
    targets: Sequence[torch.Tensor],
    predictions: Sequence[torch.Tensor],
) -> dict[str, list[float]]:
    """Per-level SPEC 8 diagnostics for c_k, t_k and c-hat_k.

    Args:
        codes: Online clean codes ``c_k`` [B, l_k, d_k] per level.
        targets: EMA targets ``t_k`` [B, l_k, d_k] per level.
        predictions: Generator predictions ``c-hat_k`` [B, l_k, d_k] per level.

    Returns:
        Lists of K floats: ``std_min``/``std_mean`` (c_k),
        ``target_std_min``/``target_std_mean`` (t_k), ``eff_rank`` (c_k),
        ``target_eff_rank`` (t_k) and ``r2`` (c-hat_k vs t_k).
    """
    if not (len(codes) == len(targets) == len(predictions)):
        raise ValueError(
            "codes, targets and predictions must have one tensor per level, "
            f"got {len(codes)}, {len(targets)}, {len(predictions)}"
        )
    out: dict[str, list[float]] = {
        "std_min": [],
        "std_mean": [],
        "target_std_min": [],
        "target_std_mean": [],
        "eff_rank": [],
        "target_eff_rank": [],
        "r2": [],
    }
    for code, target, prediction in zip(codes, targets, predictions):
        std_min, std_mean = std_summary(code)
        target_min, target_mean = std_summary(target)
        out["std_min"].append(std_min)
        out["std_mean"].append(std_mean)
        out["target_std_min"].append(target_min)
        out["target_std_mean"].append(target_mean)
        out["eff_rank"].append(effective_rank(code))
        out["target_eff_rank"].append(effective_rank(target))
        out["r2"].append(explained_variance(prediction, target))
    return out


class CollapseAlarm:
    """Per-level collapse alarm: mean std < 0.1 for 200 consecutive steps.

    Args:
        threshold: Mean-std threshold (SPEC 8 default 0.1).
        patience: Consecutive steps below the threshold (SPEC 8 default 200).
    """

    def __init__(
        self,
        threshold: float = COLLAPSE_STD,
        patience: int = COLLAPSE_PATIENCE,
    ) -> None:
        if threshold < 0.0:
            raise ValueError(f"threshold must be >= 0, got {threshold}")
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        self.threshold = float(threshold)
        self.patience = int(patience)
        self.counts: list[int] = []

    def update(self, mean_stds: Sequence[float]) -> list[bool]:
        """Advance every level's counter and return the per-level alarm flags."""
        if len(mean_stds) > len(self.counts):
            self.counts.extend([0] * (len(mean_stds) - len(self.counts)))
        alarmed: list[bool] = []
        for level, mean_std in enumerate(mean_stds):
            if float(mean_std) < self.threshold:
                self.counts[level] += 1
            else:
                self.counts[level] = 0
            alarmed.append(self.counts[level] >= self.patience)
        return alarmed

    def state_dict(self) -> dict[str, list[int]]:
        """Counter state for checkpointing."""
        return {"counts": list(self.counts)}

    def load_state_dict(self, state: dict[str, list[int]]) -> None:
        """Restore counters saved by :meth:`state_dict`."""
        self.counts = [int(value) for value in state.get("counts", [])]

"""EMA target encoder for JEPA joint training (SPEC 8, step 8).

The target encoder is a deep copy of the online module: ``requires_grad`` is
False, it is always in eval mode (a parent ``.train()`` must not switch it
back), and after every optimizer step it is updated in place with

    theta_bar <- m * theta_bar + (1 - m) * theta

where ``m`` is cosine-scheduled from 0.996 to 1.0 over the run's total steps.
"""

from __future__ import annotations

import copy
import math

import torch
from torch import nn

DEFAULT_MOMENTUM = 0.996
FINAL_MOMENTUM = 1.0


def cosine_momentum(
    step: int,
    total_steps: int,
    momentum: float = DEFAULT_MOMENTUM,
    final_momentum: float = FINAL_MOMENTUM,
) -> float:
    """Cosine schedule ``m(t) = m0 + (m1 - m0) * (1 - cos(pi t / T)) / 2``.

    ``t`` is clamped to ``[0, T]``, so step 0 gives ``m0`` and any step >=
    ``total_steps`` gives ``m1``. The schedule is non-decreasing.

    Args:
        step: optimizer step index (0-based).
        total_steps: training budget T; must be positive.
        momentum: starting momentum m0.
        final_momentum: ending momentum m1.
    """
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    if not 0.0 <= momentum <= 1.0:
        raise ValueError(f"momentum must be in [0, 1], got {momentum}")
    if not 0.0 <= final_momentum <= 1.0:
        raise ValueError(f"final_momentum must be in [0, 1], got {final_momentum}")
    progress = min(max(step, 0) / total_steps, 1.0)
    return momentum + (final_momentum - momentum) * 0.5 * (
        1.0 - math.cos(math.pi * progress)
    )


class EMAModule(nn.Module):
    """Deep-copied, gradient-free target module updated by exponential moving average.

    Args:
        module: online module to track. Its parameters and buffers are copied.
        total_steps: training budget used by the momentum schedule.
        momentum: starting momentum (SPEC default 0.996).
        final_momentum: ending momentum (SPEC default 1.0).

    After construction the copy has ``requires_grad=False`` on every parameter
    and is in eval mode. :meth:`train` always keeps it in eval mode, so calling
    ``.train()`` on a parent module cannot switch the target back on.
    """

    def __init__(
        self,
        module: nn.Module,
        total_steps: int,
        momentum: float = DEFAULT_MOMENTUM,
        final_momentum: float = FINAL_MOMENTUM,
    ) -> None:
        super().__init__()
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")
        self.module = copy.deepcopy(module)
        self.total_steps = int(total_steps)
        self.momentum = float(momentum)
        self.final_momentum = float(final_momentum)
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)
        self.module.eval()

    def train(self, mode: bool = True) -> "EMAModule":
        """Keep the target in eval mode regardless of the requested mode."""
        super().train(False)
        return self

    @torch.no_grad()
    def update(self, online: nn.Module, step: int) -> float:
        """Move the target toward ``online`` using the momentum at ``step``.

        Parameters and floating-point buffers are averaged in place; integer
        buffers (for example ``num_batches_tracked``) are copied. The online
        module must have the same named parameters and buffers.

        Args:
            online: the module whose parameters were just updated.
            step: optimizer step index, used for the cosine momentum schedule.

        Returns:
            The momentum used for this update.
        """
        momentum = cosine_momentum(
            step, self.total_steps, self.momentum, self.final_momentum
        )
        alpha = 1.0 - momentum

        online_params = dict(online.named_parameters())
        target_params = dict(self.module.named_parameters())
        if set(online_params) != set(target_params):
            raise ValueError(
                "online and target modules have different parameters: "
                f"missing in online {sorted(set(target_params) - set(online_params))}, "
                f"missing in target {sorted(set(online_params) - set(target_params))}"
            )
        for name, target in target_params.items():
            source = online_params[name]
            if source.shape != target.shape:
                raise ValueError(
                    f"parameter {name!r} shape mismatch: "
                    f"{tuple(source.shape)} vs {tuple(target.shape)}"
                )
            target.mul_(momentum).add_(source.detach(), alpha=alpha)

        online_buffers = dict(online.named_buffers())
        for name, target in self.module.named_buffers():
            source = online_buffers.get(name)
            if source is None:
                raise ValueError(f"online module is missing buffer {name!r}")
            if target.is_floating_point():
                target.mul_(momentum).add_(source.detach(), alpha=alpha)
            else:
                target.copy_(source)
        return momentum

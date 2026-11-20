"""T5.8 (REVIEW M2): global RNG state can be captured, saved and restored.

The state goes into ``latest.pt`` and comes back through ``torch.load``, whose
default ``weights_only`` unpickler accepts only plain types and tensors, so the
round trip is tested through a real file with ``weights_only=True``.
"""

from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import pytest
import torch

from jncm.utils import capture_rng_state, restore_rng_state


def _draw() -> tuple:
    return (
        random.random(),
        float(np.random.rand()),
        torch.rand(3).tolist(),
        torch.randn(2).tolist(),
    )


def test_restore_replays_the_same_draws() -> None:
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    state = capture_rng_state()
    first = _draw()
    second = _draw()
    assert first != second  # the streams do advance

    restore_rng_state(state)

    assert _draw() == first
    assert _draw() == second


def test_state_round_trips_through_a_weights_only_checkpoint(tmp_path: Path) -> None:
    random.seed(5)
    np.random.seed(5)
    torch.manual_seed(5)
    _draw()
    state = capture_rng_state()
    expected = _draw()
    torch.save({"rng_state": state}, tmp_path / "s.pt")
    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)

    loaded = torch.load(tmp_path / "s.pt", map_location="cpu", weights_only=True)
    restore_rng_state(loaded["rng_state"])

    assert _draw() == expected


def test_restore_accepts_states_that_were_mapped_to_another_device(tmp_path: Path) -> None:
    """The trainers call torch.load(..., map_location=device); on a GPU node the
    saved ByteTensors would arrive on CUDA, where set_rng_state rejects them."""
    torch.manual_seed(3)
    state = capture_rng_state()
    expected = torch.rand(2).tolist()

    class _FakeDeviceTensor:
        """Mimics a tensor that must be moved with .cpu() before use."""

        def __init__(self, tensor: torch.Tensor) -> None:
            self.tensor = tensor
            self.moved = False

        def cpu(self) -> torch.Tensor:
            self.moved = True
            return self.tensor

    wrapped = _FakeDeviceTensor(state["torch"])
    restore_rng_state(dict(state, torch=wrapped))

    assert wrapped.moved
    assert torch.rand(2).tolist() == expected


def test_a_state_without_every_stream_is_rejected() -> None:
    state = capture_rng_state()
    for key in ("python", "numpy", "torch"):
        broken = {k: v for k, v in state.items() if k != key}
        with pytest.raises(ValueError, match=key):
            restore_rng_state(broken)

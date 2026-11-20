"""T5.7 (REVIEW B1): resume checkpoints are written atomically.

With ``latest.pt`` rewritten at every milestone, a Slurm TIMEOUT is likely to
land inside a write. A truncated ``latest.pt`` would crash every later chunk of
the chain, so the file must be replaced as a whole or not at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from jncm.utils import atomic_torch_save


def test_atomic_save_writes_a_loadable_file_and_leaves_no_temporary(tmp_path: Path) -> None:
    target = tmp_path / "latest.pt"

    written = atomic_torch_save({"step": 3, "w": torch.arange(4)}, target)

    assert written == target
    state = torch.load(target, map_location="cpu")
    assert state["step"] == 3 and torch.equal(state["w"], torch.arange(4))
    assert [path.name for path in tmp_path.iterdir()] == ["latest.pt"]


def test_atomic_save_replaces_an_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "latest.pt"
    atomic_torch_save({"step": 1}, target)

    atomic_torch_save({"step": 2}, target)

    assert torch.load(target, map_location="cpu")["step"] == 2


def test_failed_save_keeps_the_previous_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "latest.pt"
    atomic_torch_save({"step": 1}, target)

    def dying_save(obj: object, destination: object) -> None:
        Path(str(destination)).write_bytes(b"partial")  # what a kill mid-write leaves
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", dying_save)
    with pytest.raises(OSError, match="disk full"):
        atomic_torch_save({"step": 2}, target)
    monkeypatch.undo()

    assert torch.load(target, map_location="cpu")["step"] == 1
    assert [path.name for path in tmp_path.iterdir()] == ["latest.pt"]

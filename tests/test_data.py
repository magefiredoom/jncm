"""Tests for MemmapDataset, InfiniteBatchIterator, determinism, and disjointness."""

from pathlib import Path
import numpy as np
import pytest
import torch
from jncm.data import InfiniteBatchIterator, MemmapDataset, count_positions


def _create_synthetic_bin(path: Path, num_windows: int, seq_len: int, offset: int = 0) -> None:
    total_tokens = num_windows * seq_len
    data = (np.arange(total_tokens, dtype=np.uint16) + offset).reshape(num_windows, seq_len)
    data.tofile(path)


def test_memmap_dataset_reading(tmp_path: Path) -> None:
    bin_file = tmp_path / "data.bin"
    _create_synthetic_bin(bin_file, num_windows=10, seq_len=32, offset=5)

    ds = MemmapDataset(bin_file, seq_len=32)
    assert len(ds) == 10
    item0 = ds[0]
    assert item0.shape == (32,)
    assert item0.dtype == torch.long
    assert item0[0].item() == 5


def test_batch_iterator_shapes_and_positions(tmp_path: Path) -> None:
    bin_file = tmp_path / "data.bin"
    _create_synthetic_bin(bin_file, num_windows=50, seq_len=32)
    ds = MemmapDataset(bin_file, seq_len=32)

    iterator = InfiniteBatchIterator(ds, batch_size=16, seed=42, shuffle=True)
    batch = next(iterator)

    assert batch.shape == (16, 32)
    assert batch.dtype == torch.long
    assert iterator.positions_seen == 16 * 32

    # Draw next 4 batches
    for _ in range(4):
        _ = next(iterator)
    assert iterator.positions_seen == 5 * 16 * 32
    assert count_positions(5, 16, 32) == iterator.positions_seen


def test_batch_iterator_determinism(tmp_path: Path) -> None:
    bin_file = tmp_path / "data.bin"
    _create_synthetic_bin(bin_file, num_windows=100, seq_len=32)
    ds = MemmapDataset(bin_file, seq_len=32)

    it1 = InfiniteBatchIterator(ds, batch_size=16, seed=777, shuffle=True)
    it2 = InfiniteBatchIterator(ds, batch_size=16, seed=777, shuffle=True)

    for _ in range(5):
        b1 = next(it1)
        b2 = next(it2)
        assert torch.equal(b1, b2)

    # Different seed gives different batch order
    it3 = InfiniteBatchIterator(ds, batch_size=16, seed=888, shuffle=True)
    b3 = next(it3)
    b1_first = next(InfiniteBatchIterator(ds, batch_size=16, seed=777, shuffle=True))
    assert not torch.equal(b3, b1_first)


def test_train_val_disjointness(tmp_path: Path) -> None:
    train_bin = tmp_path / "train.bin"
    val_bin = tmp_path / "val.bin"

    # Train tokens in range [0, 1000), Val tokens in range [5000, 6000)
    _create_synthetic_bin(train_bin, num_windows=20, seq_len=16, offset=0)
    _create_synthetic_bin(val_bin, num_windows=10, seq_len=16, offset=5000)

    train_ds = MemmapDataset(train_bin, seq_len=16)
    val_ds = MemmapDataset(val_bin, seq_len=16)

    train_it = InfiniteBatchIterator(train_ds, batch_size=8, seed=1)
    val_it = InfiniteBatchIterator(val_ds, batch_size=8, seed=2)

    train_batch = next(train_it)
    val_batch = next(val_it)

    train_set = set(train_batch.view(-1).tolist())
    val_set = set(val_batch.view(-1).tolist())

    assert train_set.isdisjoint(val_set)

# --------------------------------------------------------------------------- #
# T5.8 (REVIEW M2): the iterator can be saved and restored mid-run
# --------------------------------------------------------------------------- #


def _iterator(tmp_path: Path, windows: int = 5, batch: int = 2, shuffle: bool = True):
    path = tmp_path / f"train_{windows}.bin"
    if not path.exists():  # a second iterator maps the same file (Windows forbids rewriting it)
        _create_synthetic_bin(path, windows, 4)
    return InfiniteBatchIterator(
        MemmapDataset(path, 4), batch_size=batch, seed=7, shuffle=shuffle
    )


@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("consumed", [0, 1, 2, 3, 7, 12])
def test_restored_iterator_continues_the_same_stream(
    tmp_path: Path, shuffle: bool, consumed: int
) -> None:
    """5 windows and batch 2: a batch straddles every epoch boundary, and 12
    batches reach epoch 4, so every kind of position is covered."""
    reference = _iterator(tmp_path, shuffle=shuffle)
    for _ in range(consumed):
        next(reference)
    state = reference.state_dict()
    expected = [next(reference) for _ in range(9)]

    restored = _iterator(tmp_path, shuffle=shuffle)
    restored.load_state_dict(state)
    got = [next(restored) for _ in range(9)]

    assert all(torch.equal(a, b) for a, b in zip(expected, got))
    assert restored.positions_seen == reference.positions_seen
    assert isinstance(state["epoch"], int) and isinstance(state["pos"], int)


def test_iterator_state_survives_a_weights_only_checkpoint(tmp_path: Path) -> None:
    iterator = _iterator(tmp_path)
    for _ in range(4):
        next(iterator)
    torch.save({"data_state": iterator.state_dict()}, tmp_path / "s.pt")

    loaded = torch.load(tmp_path / "s.pt", map_location="cpu", weights_only=True)

    other = _iterator(tmp_path)
    other.load_state_dict(loaded["data_state"])
    assert torch.equal(next(other), next(iterator))


def test_iterator_state_from_another_dataset_or_seed_is_rejected(tmp_path: Path) -> None:
    iterator = _iterator(tmp_path, windows=5)
    state = iterator.state_dict()

    with pytest.raises(ValueError, match="num_windows"):
        _iterator(tmp_path, windows=6).load_state_dict(state)
    with pytest.raises(ValueError, match="batch_size"):
        _iterator(tmp_path, batch=3).load_state_dict(state)
    wrong_seed = dict(state, seed=state["seed"] + 1)
    with pytest.raises(ValueError, match="seed"):
        _iterator(tmp_path).load_state_dict(wrong_seed)
    with pytest.raises(ValueError, match="pos"):
        _iterator(tmp_path).load_state_dict(dict(state, pos=99))

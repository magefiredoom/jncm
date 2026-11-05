"""Tests for MemmapDataset, InfiniteBatchIterator, determinism, and disjointness."""

from pathlib import Path
import numpy as np
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
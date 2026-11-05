"""Memmap dataset, deterministic seeded batch iterator, and position tracking."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import torch


class MemmapDataset:
    """
    Memory-mapped dataset of fixed-length token sequences.
    
    Reads uint16 binary files produced by prep_tinystories.py.
    """

    def __init__(self, bin_path: str | Path, seq_len: int) -> None:
        self.path = Path(bin_path)
        self.seq_len = seq_len
        if not self.path.exists():
            raise FileNotFoundError(f"Binary file not found: {self.path}")

        raw = np.memmap(self.path, dtype=np.uint16, mode="r")
        self.num_windows = len(raw) // seq_len
        if self.num_windows == 0:
            raise ValueError(
                f"File {self.path} has {len(raw)} tokens, less than seq_len {seq_len}"
            )
        # Reshape view into [num_windows, seq_len]
        self.data = raw[: self.num_windows * seq_len].reshape(self.num_windows, seq_len)

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Return a single sequence of shape [seq_len] as LongTensor."""
        return torch.from_numpy(self.data[idx].astype(np.int64))


class InfiniteBatchIterator:
    """
    Infinite, deterministic seeded batch iterator returning LongTensor [B, L].

    Shuffles sequences per epoch using a seeded NumPy RandomState and
    tracks total token positions seen across all generated batches.
    """

    def __init__(
        self,
        dataset: MemmapDataset,
        batch_size: int,
        seed: int = 42,
        shuffle: bool = True,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.seq_len = dataset.seq_len
        self.seed = seed
        self.shuffle = shuffle
        self.rng = np.random.RandomState(seed)
        self.num_windows = len(dataset)
        self._indices = np.arange(self.num_windows)
        self._pos = 0
        self._positions_seen = 0
        self._reset_epoch()

    def _reset_epoch(self) -> None:
        if self.shuffle:
            self._indices = self.rng.permutation(self.num_windows)
        else:
            self._indices = np.arange(self.num_windows)
        self._pos = 0

    @property
    def positions_seen(self) -> int:
        """Total token positions yielded so far."""
        return self._positions_seen

    def __iter__(self) -> InfiniteBatchIterator:
        return self

    def __next__(self) -> torch.Tensor:
        """Yield next batch of shape [B, L] as torch.LongTensor."""
        batch_indices: list[int] = []
        while len(batch_indices) < self.batch_size:
            remaining = self.batch_size - len(batch_indices)
            avail = self.num_windows - self._pos
            take = min(remaining, avail)
            batch_indices.extend(self._indices[self._pos : self._pos + take])
            self._pos += take
            if self._pos >= self.num_windows:
                self._reset_epoch()

        batch_arr = self.dataset.data[batch_indices]  # [B, L] uint16
        tensor = torch.from_numpy(batch_arr.astype(np.int64))  # [B, L] LongTensor
        self._positions_seen += tensor.numel()
        return tensor


def count_positions(steps: int, batch_size: int, seq_len: int) -> int:
    """Calculate total token positions: steps * batch_size * seq_len."""
    return steps * batch_size * seq_len
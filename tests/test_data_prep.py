"""Tests for dataset packing logic (pure functions, no network/download required)."""

from pathlib import Path
import numpy as np
from scripts.prep_tinystories import pack_stories


def test_pack_stories_window_shape_and_eos() -> None:
    # 3 stories, seq_len = 4, eos_id = 1
    # Story 1: [10, 11] -> with eos: [10, 11, 1]
    # Story 2: [20, 21, 22] -> with eos: [20, 21, 22, 1]
    # Story 3: [30] -> with eos: [30, 1]
    # Concatenated stream: [10, 11, 1, 20, 21, 22, 1, 30, 1] (total 9 tokens)
    # Windows of 4 tokens: num_windows = 9 // 4 = 2 (remainder 1 dropped)
    # Window 0: [10, 11, 1, 20]
    # Window 1: [21, 22, 1, 30]
    stories = [[10, 11], [20, 21, 22], [30]]
    windows = pack_stories(stories, eos_id=1, seq_len=4)

    assert windows.shape == (2, 4)
    assert windows.dtype == np.uint16
    expected = np.array(
        [
            [10, 11, 1, 20],
            [21, 22, 1, 30],
        ],
        dtype=np.uint16,
    )
    assert np.array_equal(windows, expected)


def test_pack_stories_existing_eos() -> None:
    # If a story already ends with eos_id, it should not add duplicate eos
    stories = [[10, 11, 1], [20, 1]]
    windows = pack_stories(stories, eos_id=1, seq_len=3)
    # Stream: [10, 11, 1, 20, 1] -> 5 tokens -> 1 window of 3: [10, 11, 1]
    assert windows.shape == (1, 3)
    assert np.array_equal(windows[0], [10, 11, 1])


def test_pack_stories_drops_remainder() -> None:
    stories = [[5] * 10]
    windows = pack_stories(stories, eos_id=1, seq_len=4)
    # Story: [5]*10 + [1] = 11 tokens -> 2 windows of 4 = 8 tokens, 3 dropped
    assert windows.shape == (2, 4)


def test_pack_stories_empty_or_too_short() -> None:
    stories: list[list[int]] = [[], []]
    windows = pack_stories(stories, eos_id=1, seq_len=32)
    assert windows.shape == (0, 32)

    stories2 = [[1, 2]]
    windows2 = pack_stories(stories2, eos_id=1, seq_len=32)
    assert windows2.shape == (0, 32)


def test_bpe_initial_alphabet_covers_all_bytes(tmp_path: Path) -> None:
    from tokenizers.pre_tokenizers import ByteLevel

    from scripts.prep_tinystories import train_bpe_tokenizer

    tok = train_bpe_tokenizer(["hello"], vocab_size=300, out_path=tmp_path / "tok.json")
    vocab = tok.get_vocab()
    missing = [ch for ch in ByteLevel.alphabet() if ch not in vocab]
    assert missing == []


def test_memmap_roundtrip_with_packed(tmp_path: Path) -> None:
    stories = [list(range(20)), list(range(20, 50))]
    windows = pack_stories(stories, eos_id=99, seq_len=16)

    bin_file = tmp_path / "test.bin"
    windows.tofile(bin_file)

    read_memmap = np.memmap(bin_file, dtype=np.uint16, mode="r")
    reshaped = read_memmap.reshape(-1, 16)
    assert np.array_equal(reshaped, windows)
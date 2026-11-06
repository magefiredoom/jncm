"""
Prepares TinyStories dataset: trains BPE tokenizer, packs stories into length-L
windows separated by <eos>, and writes uint16 memmaps and meta.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import numpy as np


def pack_stories(
    story_token_lists: list[list[int]],
    eos_id: int,
    seq_len: int,
) -> np.ndarray:
    """
    Concatenate tokenized stories separated by <eos> and pack into windows of seq_len.

    Trailing remainder tokens that do not fill a full window are dropped.
    Returns:
        np.ndarray of shape [N, seq_len] and dtype uint16.
    """
    stream: list[int] = []
    for story in story_token_lists:
        if not story:
            continue
        stream.extend(story)
        if stream[-1] != eos_id:
            stream.append(eos_id)

    total_tokens = len(stream)
    num_windows = total_tokens // seq_len
    if num_windows == 0:
        return np.empty((0, seq_len), dtype=np.uint16)

    usable = num_windows * seq_len
    flat = np.array(stream[:usable], dtype=np.uint16)
    return flat.reshape(num_windows, seq_len)


class WindowPacker:
    """Streaming equivalent of pack_stories(): memory is O(chunk), not O(dataset)."""

    def __init__(self, eos_id: int, seq_len: int) -> None:
        self.eos_id = eos_id
        self.seq_len = seq_len
        self._tail = np.empty(0, dtype=np.uint16)

    def add(self, stories: list[list[int]]) -> np.ndarray:
        """Append tokenized stories and return all newly completed windows."""
        parts = [self._tail]
        for story in stories:
            if len(story) == 0:
                continue
            arr = np.asarray(story, dtype=np.uint16)
            parts.append(arr)
            if arr[-1] != self.eos_id:
                parts.append(np.array([self.eos_id], dtype=np.uint16))
        flat = np.concatenate(parts)
        n = len(flat) // self.seq_len
        self._tail = flat[n * self.seq_len :].copy()
        return flat[: n * self.seq_len].reshape(n, self.seq_len)

    @property
    def dropped_tokens(self) -> int:
        """Tokens carried in the unfinished tail (dropped at end of split)."""
        return int(len(self._tail))


def write_packed_split(
    split: Any,
    tokenizer: Any,
    out_path: Path,
    eos_id: int,
    seq_len: int,
    chunk_size: int = 20_000,
) -> tuple[int, int]:
    """Tokenize and pack one split incrementally; return (windows, dropped_tokens)."""
    packer = WindowPacker(eos_id, seq_len)
    n_windows = 0
    with open(out_path, "wb") as f:
        for start in range(0, len(split), chunk_size):
            end = min(start + chunk_size, len(split))
            texts = [split[i]["text"] for i in range(start, end)]
            encodings = tokenizer.encode_batch(texts)
            windows = packer.add([enc.ids for enc in encodings])
            windows.tofile(f)
            n_windows += int(windows.shape[0])
    return n_windows, packer.dropped_tokens


def train_bpe_tokenizer(
    story_texts: list[str],
    vocab_size: int,
    out_path: Path,
) -> Any:
    """Train a ByteLevel BPE tokenizer with <pad>=0, <eos>=1 and save tokenizer.json."""
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.trainers import BpeTrainer

    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<eos>"],
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(story_texts, trainer=trainer)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_path))
    return tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare TinyStories dataset")
    parser.add_argument("--out", type=str, default="data/tinystories")
    parser.add_argument("--vocab_size", type=int, default=8192)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument(
        "--max_stories",
        type=int,
        default=None,
        help="Optional limit on stories processed (for testing)",
    )
    parser.add_argument(
        "--train_subset_for_bpe",
        type=int,
        default=50000,
        help="Number of stories to use for training the BPE tokenizer",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading TinyStories from Hugging Face datasets...")
    from datasets import load_dataset
    from tokenizers import Tokenizer

    dataset = load_dataset("roneneldan/TinyStories")
    train_split = dataset["train"]
    val_split = dataset["validation"]

    if args.max_stories is not None:
        train_count = min(args.max_stories, len(train_split))
        val_count = min(max(100, args.max_stories // 5), len(val_split))
        train_split = train_split.select(range(train_count))
        val_split = val_split.select(range(val_count))

    tok_file = out_dir / "tokenizer.json"
    if not tok_file.exists():
        print(f"Training BPE tokenizer (vocab {args.vocab_size})...")
        num_bpe = min(args.train_subset_for_bpe, len(train_split))
        bpe_corpus = [train_split[i]["text"] for i in range(num_bpe)]
        tokenizer = train_bpe_tokenizer(bpe_corpus, args.vocab_size, tok_file)
    else:
        print(f"Loading existing tokenizer from {tok_file}...")
        tokenizer = Tokenizer.from_file(str(tok_file))

    pad_id = tokenizer.token_to_id("<pad>")
    eos_id = tokenizer.token_to_id("<eos>")
    print(f"Tokenizer loaded: pad_id={pad_id}, eos_id={eos_id}")

    train_bin = out_dir / "train.bin"
    val_bin = out_dir / "val.bin"

    print("Tokenizing train split (streaming)...")
    train_windows, train_dropped = write_packed_split(
        train_split, tokenizer, train_bin, eos_id, args.seq_len
    )

    print("Tokenizing val split (streaming)...")
    val_windows, val_dropped = write_packed_split(
        val_split, tokenizer, val_bin, eos_id, args.seq_len
    )

    meta = {
        "train_windows": train_windows,
        "train_tokens": train_windows * args.seq_len,
        "train_dropped_tokens": train_dropped,
        "val_windows": val_windows,
        "val_tokens": val_windows * args.seq_len,
        "val_dropped_tokens": val_dropped,
        "V": int(tokenizer.get_vocab_size()),
        "L": args.seq_len,
        "pad_id": pad_id,
        "eos_id": eos_id,
    }
    meta_path = out_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Successfully prepared TinyStories at {out_dir}:")
    print(f"  Train: {meta['train_windows']} windows ({meta['train_tokens']} tokens)")
    print(f"  Val:   {meta['val_windows']} windows ({meta['val_tokens']} tokens)")


if __name__ == "__main__":
    main()
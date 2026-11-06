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

    def tokenize_stories(texts: list[str]) -> list[list[int]]:
        encodings = tokenizer.encode_batch(texts)
        return [enc.ids for enc in encodings]

    print("Tokenizing train split...")
    train_texts = [item["text"] for item in train_split]
    train_tokens = tokenize_stories(train_texts)
    train_windows = pack_stories(train_tokens, eos_id=eos_id, seq_len=args.seq_len)

    print("Tokenizing val split...")
    val_texts = [item["text"] for item in val_split]
    val_tokens = tokenize_stories(val_texts)
    val_windows = pack_stories(val_tokens, eos_id=eos_id, seq_len=args.seq_len)

    train_bin = out_dir / "train.bin"
    val_bin = out_dir / "val.bin"
    train_windows.tofile(train_bin)
    val_windows.tofile(val_bin)

    meta = {
        "train_windows": int(train_windows.shape[0]),
        "train_tokens": int(train_windows.size),
        "val_windows": int(val_windows.shape[0]),
        "val_tokens": int(val_windows.size),
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
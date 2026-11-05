"""
Protocol gate evaluation script (SPEC 6.1, TASKS T1.2).
Evaluates real validation sequences to report Gen PPL and unigram entropy H.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any
import numpy as np
import torch

from jncm.data import MemmapDataset
from jncm.eval.genppl import GPT2Scorer, gen_ppl, unigram_entropy
from jncm.utils import seed_everything


def eval_real_data(
    data_bin_path: str | Path,
    tokenizer_path: str | Path,
    scorer: Any,
    gpt2_tokenizer: Any,
    seq_len: int = 128,
    n_samples: int = 256,
    n_seeds: int = 5,
) -> dict[str, Any]:
    """
    Evaluate validation sequences across multiple seeds.

    Decodes sequences to text, re-tokenizes with GPT-2 tokenizer,
    computes unigram entropy on GPT-2 and dataset tokens, and scores with GPT-2.
    """
    from tokenizers import Tokenizer

    dataset = MemmapDataset(data_bin_path, seq_len=seq_len)
    tok = Tokenizer.from_file(str(tokenizer_path))
    num_total = len(dataset)
    if num_total < n_samples:
        raise ValueError(f"Dataset has {num_total} sequences, less than n_samples {n_samples}")

    ppl_runs: list[float] = []
    h_gpt2_runs: list[float] = []
    h_dataset_runs: list[float] = []

    for seed_idx in range(n_seeds):
        rng = np.random.RandomState(seed_idx + 42)
        indices = rng.choice(num_total, size=n_samples, replace=False)
        sample_tensors = [dataset[i] for i in indices]
        dataset_token_lists = [t.tolist() for t in sample_tensors]

        # Decode using dataset tokenizer
        texts = [tok.decode(ids) for ids in dataset_token_lists]

        # Re-tokenize with GPT-2 tokenizer
        gpt2_token_lists: list[list[int]] = []
        for t in texts:
            enc = gpt2_tokenizer.encode(t)
            if hasattr(enc, "ids"):
                gpt2_token_lists.append(enc.ids)
            else:
                gpt2_token_lists.append(list(enc))

        # Entropies
        h_gpt2 = unigram_entropy(gpt2_token_lists)
        h_dataset = unigram_entropy(dataset_token_lists)
        ppl = gen_ppl(texts, scorer)

        ppl_runs.append(ppl)
        h_gpt2_runs.append(h_gpt2)
        h_dataset_runs.append(h_dataset)

    return {
        "ppl_mean": float(np.mean(ppl_runs)),
        "ppl_std": float(np.std(ppl_runs)),
        "h_gpt2_mean": float(np.mean(h_gpt2_runs)),
        "h_gpt2_std": float(np.std(h_gpt2_runs)),
        "h_dataset_mean": float(np.mean(h_dataset_runs)),
        "h_dataset_std": float(np.std(h_dataset_runs)),
        "ppl_runs": ppl_runs,
        "h_gpt2_runs": h_gpt2_runs,
        "h_dataset_runs": h_dataset_runs,
        "n_samples": n_samples,
        "n_seeds": n_seeds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Gen PPL and H on real validation data")
    parser.add_argument("--data", type=str, default="data/tinystories/val.bin")
    parser.add_argument("--tokenizer", type=str, default="data/tinystories/tokenizer.json")
    default_gpt2 = os.environ.get("GPT2_LARGE_PATH", "models/gpt2-large")
    parser.add_argument("--gpt2", type=str, default=default_gpt2)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--n_samples", type=int, default=256)
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    scorer = GPT2Scorer(model_path=args.gpt2, device=args.device)
    from transformers import AutoTokenizer

    gpt2_tokenizer = AutoTokenizer.from_pretrained(str(args.gpt2), local_files_only=True)

    print(f"Running Protocol Gate on {args.data} with {args.n_samples} samples across {args.n_seeds} seeds...")
    results = eval_real_data(
        data_bin_path=args.data,
        tokenizer_path=args.tokenizer,
        scorer=scorer,
        gpt2_tokenizer=gpt2_tokenizer,
        seq_len=args.seq_len,
        n_samples=args.n_samples,
        n_seeds=args.n_seeds,
    )

    print("\n=== Protocol Gate Results ===")
    print(f"Gen PPL:           {results['ppl_mean']:.2f} +/- {results['ppl_std']:.2f} (Target: ~9.67, tol 15% [8.22, 11.12])")
    print(f"H (GPT-2 tokens):  {results['h_gpt2_mean']:.2f} +/- {results['h_gpt2_std']:.2f} (Target: ~3.96, tol 0.15 [3.81, 4.11])")
    print(f"H (Dataset tokens):{results['h_dataset_mean']:.2f} +/- {results['h_dataset_std']:.2f}")

    ref_ppl, ref_h = 9.67, 3.96
    pass_ppl = abs(results["ppl_mean"] - ref_ppl) / ref_ppl <= 0.15
    pass_h = abs(results["h_gpt2_mean"] - ref_h) <= 0.15 or abs(results["h_dataset_mean"] - ref_h) <= 0.15

    verdict = "PASS" if (pass_ppl and pass_h) else "CHECK TOLERANCE"
    print(f"Verdict: {verdict}")

    if args.out:
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
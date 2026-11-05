"""
Downloads and caches gpt2-large locally for offline generative perplexity scoring.
Run on a cluster login node or machine with internet access.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and cache gpt2-large locally")
    default_out = os.environ.get("GPT2_LARGE_PATH", "models/gpt2-large")
    parser.add_argument(
        "--out",
        type=str,
        default=default_out,
        help="Local directory to store gpt2-large model weights and tokenizer",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading gpt2-large to {out_path}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "gpt2-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(str(out_path))

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.save_pretrained(str(out_path))

    print(f"Successfully cached {model_name} to {out_path}")
    print(f"Set export GPT2_LARGE_PATH={out_path.resolve()} before running eval jobs.")


if __name__ == "__main__":
    main()
"""Generative perplexity (Gen PPL) and unigram entropy evaluation."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any
import torch


def unigram_entropy(token_id_lists: list[list[int]]) -> float:
    """
    Compute mean unigram entropy (in nats) across a list of token sequences.

    For each sequence, H = - sum_t p(t) * ln(p(t)) from empirical unigram counts.
    Returns the mean entropy over all sequences.
    """
    if not token_id_lists:
        return 0.0

    sample_entropies: list[float] = []
    for seq in token_id_lists:
        if not seq:
            sample_entropies.append(0.0)
            continue
        counts: dict[int, int] = {}
        for tok in seq:
            counts[tok] = counts.get(tok, 0) + 1
        n = float(len(seq))
        h = 0.0
        for cnt in counts.values():
            p = cnt / n
            h -= p * math.log(p)
        sample_entropies.append(h)

    return float(sum(sample_entropies) / len(sample_entropies))


class GPT2Scorer:
    """
    Autoregressive token scorer using offline pretrained GPT-2 (default gpt2-large).

    Reads from local path specified by model_path or $GPT2_LARGE_PATH.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str = "cpu",
    ) -> None:
        if model_path is None:
            model_path = os.environ.get("GPT2_LARGE_PATH", "models/gpt2-large")
        self.model_path = Path(model_path)
        self.device = torch.device(device)
        self._model: Any = None
        self._tokenizer: Any = None

    def _load(self) -> None:
        if self._model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"GPT-2 weights not found at {self.model_path}. "
                    "Run scripts/fetch_models.py first or set $GPT2_LARGE_PATH."
                )
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path), local_files_only=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path), local_files_only=True
            ).to(self.device)
            self._model.eval()

    def score_sample(self, text: str) -> tuple[float, int]:
        """
        Score a single text sequence.

        Returns:
            (total_nll, num_scored_tokens) where num_scored_tokens = len(tokens) - 1.
        """
        self._load()
        enc = self._tokenizer(text, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)  # [1, T]
        t_len = input_ids.shape[1]
        if t_len <= 1:
            return (0.0, 0)

        with torch.no_grad():
            outputs = self._model(input_ids)
            logits = outputs.logits  # [1, T, V]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            nll_sum = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="sum",
            )
            return (float(nll_sum.item()), t_len - 1)

    def score_texts(self, texts: list[str]) -> tuple[float, int]:
        """Score a list of texts and return cumulative (total_nll, total_tokens)."""
        tot_nll = 0.0
        tot_tokens = 0
        for t in texts:
            nll, count = self.score_sample(t)
            tot_nll += nll
            tot_tokens += count
        return (tot_nll, tot_tokens)


def gen_ppl(texts: list[str], scorer: Any) -> float:
    """
    Calculate generative perplexity: exp(mean token NLL over all scored tokens).

    Args:
        texts: list of decoded text strings
        scorer: GPT2Scorer instance or callable returning (total_nll, num_tokens)
    """
    if not texts:
        return float("nan")

    if hasattr(scorer, "score_texts"):
        tot_nll, tot_tokens = scorer.score_texts(texts)
    elif hasattr(scorer, "score_sample"):
        tot_nll = 0.0
        tot_tokens = 0
        for t in texts:
            nll, cnt = scorer.score_sample(t)
            tot_nll += nll
            tot_tokens += cnt
    elif callable(scorer):
        tot_nll, tot_tokens = scorer(texts)
    else:
        raise TypeError(f"Unsupported scorer: {type(scorer)}")

    if tot_tokens == 0:
        return float("nan")

    mean_nll = tot_nll / tot_tokens
    return float(math.exp(mean_nll))
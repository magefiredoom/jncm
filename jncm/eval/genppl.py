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

    def _encode(self, text: str) -> list[int]:
        """Token ids for one text, never truncated (SPEC 6.1)."""
        enc = self._tokenizer(text)
        ids = enc["input_ids"]
        if torch.is_tensor(ids):
            return [int(token) for token in ids.reshape(-1).tolist()]
        if ids and isinstance(ids[0], (list, tuple)):
            ids = ids[0]
        return [int(token) for token in ids]

    def _padding_id(self) -> int:
        """Pad id for batching; GPT-2 has no pad token, so fall back to EOS."""
        pad = getattr(self._tokenizer, "pad_token_id", None)
        if pad is None:
            pad = getattr(self._tokenizer, "eos_token_id", None)
        return 0 if pad is None else int(pad)

    def score_batch(self, texts: list[str], batch_size: int = 8) -> tuple[float, int]:
        """
        Score a list of texts in right-padded batches (SPEC 6.1).

        Padding is safe: the model receives an attention mask and both the NLL
        sum and the token count cover only non-pad target positions. Texts of
        equal length are scored together without padding. ``score_sample`` is
        the one-at-a-time reference path and must agree with this method.

        Args:
            texts: decoded sample strings.
            batch_size: maximum number of texts per forward pass (>= 1).

        Returns:
            ``(total_nll, num_scored_tokens)`` over all texts.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self._load()
        pad_id = self._padding_id()
        total_nll = 0.0
        total_tokens = 0
        for start in range(0, len(texts), batch_size):
            chunk = [self._encode(text) for text in texts[start : start + batch_size]]
            max_len = max((len(ids) for ids in chunk), default=0)
            if max_len <= 1:
                continue
            input_ids = torch.full(
                (len(chunk), max_len), pad_id, dtype=torch.long, device=self.device
            )
            attention = torch.zeros(
                (len(chunk), max_len), dtype=torch.long, device=self.device
            )
            for row, ids in enumerate(chunk):
                if ids:
                    input_ids[row, : len(ids)] = torch.tensor(
                        ids, dtype=torch.long, device=self.device
                    )
                    attention[row, : len(ids)] = 1
            with torch.no_grad():
                logits = self._model(input_ids, attention_mask=attention).logits  # [B, T, V]
                shift_logits = logits[:, :-1, :]
                shift_labels = input_ids[:, 1:]
                shift_mask = attention[:, 1:].bool()
                token_nll = torch.nn.functional.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.size(-1)),
                    shift_labels.reshape(-1),
                    reduction="none",
                ).view(shift_labels.shape)
                total_nll += float((token_nll * shift_mask).sum().item())
                total_tokens += int(shift_mask.sum().item())
        return (total_nll, total_tokens)

    def score_texts(
        self, texts: list[str], batch_size: int = 8
    ) -> tuple[float, int]:
        """Score a list of texts in batches; cumulative (total_nll, total_tokens)."""
        return self.score_batch(texts, batch_size=batch_size)

    @property
    def tokenizer(self) -> Any:
        """The loaded GPT-2 tokenizer (loads the model on first access)."""
        self._load()
        return self._tokenizer


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
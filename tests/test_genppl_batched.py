"""Batched GPT-2 scoring tests (T4.2).

The batched path must be padding-safe and reproduce the one-at-a-time
``score_sample`` path, which stays in ``GPT2Scorer`` as the reference. A tiny
hand-built causal language model and a character tokenizer stand in for GPT-2,
so these tests need no download and no GPU.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch
import torch.nn as nn

from jncm.eval.genppl import GPT2Scorer, gen_ppl

VOCAB = 32


class _CharTokenizer:
    """Deterministic character tokenizer: id = 1 + (ord(ch) % 31)."""

    def __init__(self) -> None:
        self.pad_token_id = 0
        self.eos_token_id = 0

    def __call__(self, text: str, return_tensors: str | None = None) -> dict:
        ids = [1 + (ord(ch) % (VOCAB - 1)) for ch in text]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return {"input_ids": ids}


class _TinyCausalLM(nn.Module):
    """One causal self-attention block plus a head, with random weights."""

    def __init__(self, d: int = 16, seed: int = 0) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.emb = nn.Embedding(VOCAB, d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, VOCAB)
        with torch.no_grad():
            for param in (
                self.emb.weight,
                self.qkv.weight,
                self.qkv.bias,
                self.out.weight,
                self.out.bias,
            ):
                param.copy_(torch.randn(param.shape, generator=generator) * 0.2)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> SimpleNamespace:
        x = self.emb(input_ids)  # [B, T, d]
        q, k, v = self.qkv(x).chunk(3, dim=-1)  # each [B, T, d]
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])  # [B, T, T]
        length = input_ids.shape[1]
        causal = torch.tril(torch.ones(length, length, dtype=torch.bool))
        scores = scores.masked_fill(~causal, float("-inf"))
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask[:, None, :] == 0, float("-inf"))
        x = x + torch.softmax(scores, dim=-1) @ v
        return SimpleNamespace(logits=self.out(x))  # [B, T, V]


def _scorer() -> GPT2Scorer:
    """A GPT2Scorer with fake model and tokenizer injected (no files read)."""
    scorer = GPT2Scorer(model_path="unused")
    scorer._model = _TinyCausalLM().eval()
    scorer._tokenizer = _CharTokenizer()
    return scorer


def _reference(texts: list[str]) -> tuple[float, int]:
    scorer = _scorer()
    total_nll = 0.0
    total_tokens = 0
    for text in texts:
        nll, tokens = scorer.score_sample(text)
        total_nll += nll
        total_tokens += tokens
    return total_nll, total_tokens


TEXTS = [
    "a",
    "abc def",
    "the quick brown fox jumps",
    "hello",
    "xy" * 20,
    "one two three four five six seven",
]


def test_score_batch_matches_one_at_a_time_path() -> None:
    ref_nll, ref_tokens = _reference(TEXTS)
    nll, tokens = _scorer().score_batch(TEXTS, batch_size=2)

    assert tokens == ref_tokens
    assert nll / tokens == pytest_approx(ref_nll / ref_tokens, abs=1e-4)


def test_score_batch_padding_does_not_change_a_short_sample() -> None:
    short = "the quick brown fox"
    long = "a" * 40

    alone_nll, alone_tokens = _scorer().score_batch([short])
    long_nll, _ = _scorer().score_batch([long])
    together_nll, together_tokens = _scorer().score_batch([short, long], batch_size=2)

    assert together_tokens == alone_tokens + _scorer().score_batch([long])[1]
    assert together_nll - long_nll == pytest_approx(alone_nll, abs=1e-4)


def test_score_batch_counts_only_real_targets() -> None:
    nll, tokens = _scorer().score_batch(["a", "abcdef"])

    assert tokens == 0 + 5
    assert nll > 0.0


def test_short_and_empty_texts_contribute_nothing() -> None:
    scorer = _scorer()

    assert scorer.score_batch([]) == (0.0, 0)
    assert scorer.score_batch(["", "x"]) == (0.0, 0)


def test_score_texts_delegates_to_batched_path() -> None:
    scorer = _scorer()
    calls: list[tuple[int, int]] = []
    original = scorer.score_batch

    def spy(texts: list[str], batch_size: int = 8) -> tuple[float, int]:
        calls.append((len(texts), batch_size))
        return original(texts, batch_size=batch_size)

    scorer.score_batch = spy  # type: ignore[method-assign]
    nll, tokens = scorer.score_texts(["abc", "defg"], batch_size=1)

    assert calls == [(2, 1)]
    assert tokens == (3 - 1) + (4 - 1)  # one scored target per position after the first
    assert nll > 0.0


def test_gen_ppl_matches_the_reference_path() -> None:
    ref_nll, ref_tokens = _reference(TEXTS)
    expected = math.exp(ref_nll / ref_tokens)

    observed = gen_ppl(TEXTS, _scorer())

    assert observed == pytest_approx(expected, rel=1e-5)


def pytest_approx(value: float, abs: float = 0.0, rel: float = 0.0):
    """Tiny local wrapper so the test file needs no pytest import for approx."""
    import pytest

    return pytest.approx(value, abs=abs, rel=rel)

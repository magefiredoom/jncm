"""Tests for unigram entropy and generative perplexity evaluation."""

import math
import pytest
from jncm.eval.genppl import gen_ppl, unigram_entropy


def test_unigram_entropy_uniform_distribution() -> None:
    # Uniform over n distinct tokens has entropy ln(n)
    for n in [2, 4, 16, 64]:
        seq = [list(range(n))]
        h = unigram_entropy(seq)
        expected = math.log(n)
        assert math.isclose(h, expected, rel_tol=1e-5)


def test_unigram_entropy_multiple_samples() -> None:
    # Sample 1: uniform over 4 tokens -> ln(4)
    # Sample 2: uniform over 16 tokens -> ln(16)
    # Sample 3: single token repeated 10 times -> entropy 0.0
    seqs = [
        [0, 1, 2, 3],
        list(range(16)),
        [7] * 10,
    ]
    h = unigram_entropy(seqs)
    expected = (math.log(4) + math.log(16) + 0.0) / 3.0
    assert math.isclose(h, expected, rel_tol=1e-5)


def test_unigram_entropy_empty() -> None:
    assert unigram_entropy([]) == 0.0
    assert unigram_entropy([[]]) == 0.0


class FakeScorerConstantNLL:
    """Mock scorer returning a constant per-token negative log-likelihood."""

    def __init__(self, token_nll: float, tokens_per_sample: int = 10) -> None:
        self.token_nll = token_nll
        self.tokens_per_sample = tokens_per_sample

    def score_texts(self, texts: list[str]) -> tuple[float, int]:
        total_tokens = len(texts) * self.tokens_per_sample
        total_nll = total_tokens * self.token_nll
        return (total_nll, total_tokens)


def test_gen_ppl_known_constant_nll() -> None:
    scorer = FakeScorerConstantNLL(token_nll=2.0, tokens_per_sample=10)
    texts = ["story one", "story two", "story three"]
    ppl = gen_ppl(texts, scorer)
    expected = math.exp(2.0)
    assert math.isclose(ppl, expected, rel_tol=1e-5)


class FakeScorerSampleSpecific:
    """Mock scorer returning varying lengths and losses."""

    def score_sample(self, text: str) -> tuple[float, int]:
        # Text "short" -> 10 tokens with loss 1.5
        # Text "long" -> 20 tokens with loss 3.0
        if "short" in text:
            return (15.0, 10)
        return (60.0, 20)


def test_gen_ppl_weighted_tokens() -> None:
    scorer = FakeScorerSampleSpecific()
    texts = ["short story", "long story"]
    # Total tokens = 10 + 20 = 30
    # Total nll = 15.0 + 60.0 = 75.0
    # Mean nll = 75.0 / 30 = 2.5
    # PPL = exp(2.5)
    ppl = gen_ppl(texts, scorer)
    expected = math.exp(2.5)
    assert math.isclose(ppl, expected, rel_tol=1e-5)


def test_gen_ppl_empty_input() -> None:
    scorer = FakeScorerConstantNLL(token_nll=1.0)
    assert math.isnan(gen_ppl([], scorer))
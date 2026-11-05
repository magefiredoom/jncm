"""Tests for protocol gate evaluation script using mock tokenizers and fake scorers."""

from pathlib import Path
import numpy as np
import pytest
from scripts.eval_data_row import eval_real_data


class MockTokenizer:
    """Mock dataset tokenizer mapping IDs to space-separated token strings."""

    def decode(self, ids: list[int]) -> str:
        return " ".join(f"w_{i}" for i in ids)


class MockGPT2Tokenizer:
    """Mock GPT-2 tokenizer mapping space-separated words to integer IDs."""

    def encode(self, text: str) -> list[int]:
        return [abs(hash(w)) % 1000 for w in text.split()]


class MockScorer:
    """Mock scorer returning a deterministic constant negative log likelihood."""

    def __init__(self, token_nll: float = 2.27) -> None:
        self.token_nll = token_nll

    def score_texts(self, texts: list[str]) -> tuple[float, int]:
        total_tokens = sum(len(t.split()) for t in texts)
        return (total_tokens * self.token_nll, total_tokens)


def test_eval_real_data_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Create synthetic validation bin: 30 sequences of length 16
    val_bin = tmp_path / "val.bin"
    num_seqs = 30
    seq_len = 16
    data = np.arange(num_seqs * seq_len, dtype=np.uint16) % 100
    data.tofile(val_bin)

    tok_file = tmp_path / "tokenizer.json"
    tok_file.write_text("{}", encoding="utf-8")

    # Monkeypatch Tokenizer.from_file to return MockTokenizer
    from tokenizers import Tokenizer

    monkeypatch.setattr(Tokenizer, "from_file", lambda path: MockTokenizer())

    scorer = MockScorer(token_nll=2.27)
    gpt2_tok = MockGPT2Tokenizer()

    results = eval_real_data(
        data_bin_path=val_bin,
        tokenizer_path=tok_file,
        scorer=scorer,
        gpt2_tokenizer=gpt2_tok,
        seq_len=seq_len,
        n_samples=10,
        n_seeds=2,
    )

    assert "ppl_mean" in results
    assert "h_gpt2_mean" in results
    assert "h_dataset_mean" in results
    assert len(results["ppl_runs"]) == 2
    assert results["n_samples"] == 10
    assert results["n_seeds"] == 2
    # Verify PPL exp(2.27) ~ 9.679
    assert abs(results["ppl_mean"] - np.exp(2.27)) < 1e-4
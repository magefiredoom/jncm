"""The scorer runs GPT-2 without importing `transformers` (jncm/eval/gpt2_lean.py).

2026-11-20: transformers 5.22.1 reads every .py file under `transformers/` when it is
imported (`create_import_structure_from_path`). On FIR's Lustre /home that import took
15 min 16 s cold and 7 min 26 s warm (probe jobs 4746172 and 4748974), on every job. The
scorer needs a tokenizer that returns ids and a model that maps ids to logits, so it now
uses `tokenizers` (1 s) and a 150-line torch GPT-2 that reads model.safetensors itself.
The Hugging Face classes stay in these tests as the reference.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from jncm.eval.gpt2_lean import (
    LeanGPT2Tokenizer,
    load_gpt2,
    read_safetensors,
)

REAL_GPT2 = Path("models/gpt2-large")
needs_real_weights = pytest.mark.skipif(
    not (REAL_GPT2 / "model.safetensors").exists(), reason="models/gpt2-large not present"
)


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A random two-layer GPT-2 written by transformers itself, in the real file layout."""
    from transformers import GPT2Config, GPT2LMHeadModel

    path = tmp_path_factory.mktemp("tiny_gpt2")
    torch.manual_seed(0)
    config = GPT2Config(
        vocab_size=50257, n_positions=64, n_embd=32, n_layer=2, n_head=4,
        activation_function="gelu_new",
    )
    model = GPT2LMHeadModel(config).eval()
    # a non-trivial LayerNorm so a wrong epsilon or affine order would show
    for name, param in model.named_parameters():
        if name.endswith("ln_f.weight") or ".ln_" in name and name.endswith(".weight"):
            torch.nn.init.uniform_(param, 0.5, 1.5)
        if name.endswith(".bias") and ".ln_" in name or name.endswith("ln_f.bias"):
            torch.nn.init.normal_(param, 0.0, 0.2)
    model.save_pretrained(path, safe_serialization=True)
    return path


def _hf(path: Path):
    from transformers import GPT2LMHeadModel

    return GPT2LMHeadModel.from_pretrained(str(path)).eval()


def test_safetensors_reader_matches_the_safetensors_package(tiny_dir: Path) -> None:
    from safetensors.torch import load_file

    expected = load_file(str(tiny_dir / "model.safetensors"))
    got = read_safetensors(tiny_dir / "model.safetensors")

    assert set(got) == set(expected)
    for name, tensor in expected.items():
        assert got[name].dtype == tensor.dtype and got[name].shape == tensor.shape, name
        assert torch.equal(got[name], tensor), name


def test_logits_match_the_hugging_face_model(tiny_dir: Path) -> None:
    torch.manual_seed(1)
    ids = torch.randint(0, 50257, (3, 17))

    with torch.no_grad():
        want = _hf(tiny_dir)(ids).logits
        got = load_gpt2(tiny_dir, device="cpu")(ids).logits

    assert got.shape == want.shape == (3, 17, 50257)
    assert torch.allclose(got, want, rtol=1e-4, atol=1e-4), float((got - want).abs().max())


def test_right_padded_batches_match_on_the_real_positions(tiny_dir: Path) -> None:
    """score_batch right-pads and passes an attention mask; the real tokens' logits must
    not depend on the padding, and an all-pad row must stay finite (the NLL is multiplied
    by a zero mask afterwards, and NaN * 0 is NaN)."""
    torch.manual_seed(2)
    ids = torch.randint(0, 50256, (3, 12))
    attention = torch.ones_like(ids)
    attention[1, 7:] = 0
    ids[1, 7:] = 50256
    attention[2, :] = 0
    ids[2, :] = 50256

    with torch.no_grad():
        want = _hf(tiny_dir)(ids, attention_mask=attention).logits
        got = load_gpt2(tiny_dir, device="cpu")(ids, attention_mask=attention).logits

    assert torch.isfinite(got).all()
    real = attention.bool()
    assert torch.allclose(got[real], want[real], rtol=1e-4, atol=1e-4)


def test_the_model_is_in_eval_mode_and_frozen(tiny_dir: Path) -> None:
    model = load_gpt2(tiny_dir, device="cpu")

    assert not model.training
    assert all(not p.requires_grad for p in model.parameters())


def test_too_long_an_input_raises(tiny_dir: Path) -> None:
    model = load_gpt2(tiny_dir, device="cpu")

    with pytest.raises(ValueError, match="positions"):
        model(torch.zeros(1, 65, dtype=torch.long))


@needs_real_weights
def test_tokenizer_matches_the_hugging_face_tokenizer() -> None:
    from transformers import AutoTokenizer

    reference = AutoTokenizer.from_pretrained(str(REAL_GPT2), local_files_only=True)
    lean = LeanGPT2Tokenizer(REAL_GPT2)
    texts = [
        "",
        "Once upon a time, there was a little girl named Lily.",
        "  leading and trailing spaces  ",
        "line one\nline two\n\nline four",
        "naïve café — 日本語 and emoji 😀",
        "<|endoftext|> inside text",
        "tabs\tand 123 numbers 4.56",
    ]

    for text in texts:
        assert lean(text)["input_ids"] == reference(text)["input_ids"], repr(text)
    as_tensor = lean("Hello there", return_tensors="pt")["input_ids"]
    assert as_tensor.shape == (1, len(reference("Hello there")["input_ids"]))
    assert lean.eos_token_id == reference.eos_token_id == 50256
    assert lean.pad_token_id is None


@pytest.mark.slow
@needs_real_weights
def test_real_gpt2_large_logits_and_nll_match_the_hugging_face_model() -> None:
    """VERIFIED on gpt2-large at the commit that introduced the module (see STATE)."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(REAL_GPT2), local_files_only=True)
    texts = [
        "Once upon a time, there was a little girl named Lily. She loved to play in the garden.",
        "The quick brown fox jumps over the lazy dog, and then it went home to sleep.",
    ]
    reference = _hf(REAL_GPT2)
    lean = load_gpt2(REAL_GPT2, device="cpu")

    for text in texts:
        ids = tokenizer(text, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            want = reference(ids).logits
            got = lean(ids).logits
        nll_want = torch.nn.functional.cross_entropy(want[0, :-1], ids[0, 1:], reduction="sum")
        nll_got = torch.nn.functional.cross_entropy(got[0, :-1], ids[0, 1:], reduction="sum")
        assert float((got - want).abs().max()) < 5e-3
        assert abs(float(nll_got - nll_want)) < 1e-2 * max(1.0, abs(float(nll_want)) / 100)


def test_the_scorer_never_imports_transformers(tiny_dir: Path, tmp_path: Path) -> None:
    """The point of the module, checked in a fresh interpreter: loading and scoring
    leaves `transformers` out of sys.modules."""
    if not (REAL_GPT2 / "tokenizer.json").exists():
        pytest.skip("needs a real tokenizer.json")
    model_dir = tmp_path / "gpt2"
    model_dir.mkdir()
    for name in ("model.safetensors", "config.json"):
        (model_dir / name).write_bytes((tiny_dir / name).read_bytes())
    (model_dir / "tokenizer.json").write_bytes((REAL_GPT2 / "tokenizer.json").read_bytes())
    code = (
        "import sys, json\n"
        "from jncm.eval.genppl import GPT2Scorer\n"
        "s = GPT2Scorer(model_path=sys.argv[1], device='cpu')\n"
        "nll, n = s.score_texts(['hello world', 'a second sample text'], batch_size=2)\n"
        "print(json.dumps({'tokens': n, 'finite': nll == nll, "
        "'transformers': 'transformers' in sys.modules}))\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", code, str(model_dir)], capture_output=True, text=True, timeout=300
    )

    assert result.returncode == 0, result.stderr[-600:]
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report == {"tokens": report["tokens"], "finite": True, "transformers": False}
    assert report["tokens"] > 0


@needs_real_weights
def test_encode_is_what_run_eval_and_eval_data_row_call() -> None:
    """`scripts/run_eval.py` and `scripts/eval_data_row.py` call `.encode(text)` on the
    scorer's tokenizer; it was missing on FIR (canary 3, 2026-11-20)."""
    from transformers import AutoTokenizer

    reference = AutoTokenizer.from_pretrained(str(REAL_GPT2), local_files_only=True)
    lean = LeanGPT2Tokenizer(REAL_GPT2)

    for text in ["", "Once upon a time.", "a  b\nc", "café 日本語"]:
        assert lean.encode(text) == reference.encode(text), repr(text)
        assert lean.encode(text) == lean(text)["input_ids"]

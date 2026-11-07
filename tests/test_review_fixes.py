"""Regression tests for docs/REVIEW_M0-M2.md findings F1 to F7 and F10.

Drafted without a torch runtime. Each test should FAIL before its fix and PASS after it.
If a test is wrong for a reason unrelated to its finding, fix the test and say so in docs/QUESTIONS.md.
Imports of not-yet-existing names are done inside the tests so one missing name cannot hide the others.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from jncm.config import TokenizerConfig, load_config
from jncm.tokenizer import MultiscaleTokenizer


# ---------------------------------------------------------------- F1
def test_f1_transformer_block_has_learned_attention_projections() -> None:
    from jncm.tokenizer.trunk import TransformerBlock

    d = 64
    block = TransformerBlock(d, n_heads=4)
    n_params = sum(p.numel() for p in block.parameters())
    # qkv (3d^2+3d) + out proj (d^2+d) + mlp (8d^2+5d) + two LayerNorms (4d) = 12d^2 + 13d
    assert n_params == 12 * d * d + 13 * d


# ---------------------------------------------------------------- F2
@pytest.mark.parametrize("quantizer", ["vq", "noisy"])
def test_f2_forward_backward_under_bf16_autocast(quantizer: str) -> None:
    torch.manual_seed(0)
    V = 64
    tok = MultiscaleTokenizer(TokenizerConfig(d=64, K=4, quantizer=quantizer), L=32, V=V)
    tok.train()
    x = torch.randint(0, V, (4, 32))
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        logits, enc, _ = tok(x, cutoff=4)
        loss = F.cross_entropy(logits.float().reshape(-1, V), x.reshape(-1)) + enc.quant_loss.float()
    loss.backward()
    assert torch.isfinite(loss).item()
    if quantizer == "vq":
        assert all(t.dtype == torch.long for t in enc.targets)


# ---------------------------------------------------------------- F3
def test_f3_window_packer_matches_pack_stories() -> None:
    from scripts.prep_tinystories import WindowPacker, pack_stories

    rng = np.random.RandomState(0)
    eos = 1
    for _ in range(100):
        seq_len = int(rng.choice([4, 7, 16]))
        stories = [
            rng.randint(2, 100, size=int(rng.randint(0, 30))).tolist()
            for _ in range(int(rng.randint(0, 40)))
        ]
        ref = pack_stories(stories, eos_id=eos, seq_len=seq_len)

        packer = WindowPacker(eos, seq_len)
        parts, i = [], 0
        while i < len(stories):
            j = i + int(rng.randint(1, 6))
            parts.append(packer.add(stories[i:j]))
            i = j
        got = np.concatenate(parts, axis=0) if parts else np.empty((0, seq_len), dtype=np.uint16)

        assert got.dtype == np.uint16
        assert got.shape == ref.shape
        assert (got == ref).all()


# ---------------------------------------------------------------- F4
def test_f4_no_silent_synthetic_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from jncm.train_tokenizer import setup_data

    monkeypatch.chdir(tmp_path)
    cfg = load_config(None)  # defaults: L=32, V=64
    with pytest.raises(FileNotFoundError):
        setup_data(cfg, data_dir="data/tinystories")


def test_f4_synthetic_only_when_explicitly_allowed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from jncm.train_tokenizer import setup_data

    monkeypatch.chdir(tmp_path)
    cfg = load_config(None)
    train_ds, val_ds = setup_data(
        cfg, data_dir="data/tinystories", allow_synthetic=True, temp_dir=tmp_path / "syn"
    )
    assert len(train_ds) > 0 and len(val_ds) > 0


def test_f4_meta_mismatch_raises(tmp_path: Path) -> None:
    from jncm.train_tokenizer import setup_data

    cfg = load_config(None)  # L=32, V=64
    d = tmp_path / "data" / "tinystories"
    d.mkdir(parents=True)
    np.zeros(64 * cfg.L, dtype=np.uint16).tofile(d / "train.bin")
    np.zeros(64 * cfg.L, dtype=np.uint16).tofile(d / "val.bin")
    (d / "meta.json").write_text(json.dumps({"L": 128, "V": 8192}), encoding="utf-8")
    with pytest.raises(ValueError):
        setup_data(cfg, data_dir=d)


# ---------------------------------------------------------------- F5
def test_f5_eval_data_row_help_works() -> None:
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "scripts/eval_data_row.py", "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--n_samples" in proc.stdout


# ---------------------------------------------------------------- F6
def test_f6_unknown_config_keys_raise() -> None:
    with pytest.raises(ValueError):
        load_config(None, ["tokenizer.sigma_K=0.3"])  # typo: real key is sigma_k


# ---------------------------------------------------------------- F8
def test_f8_train_writes_under_results_dir(tmp_path: Path) -> None:
    from jncm.train_tokenizer import train

    cfg = load_config(Path("configs/smoke.yaml"))
    cfg.train.eval_every = 1
    cfg.train.save_every = 2
    cfg.train.batch_size = 2
    cfg.device = "cpu"

    results_dir = tmp_path / "custom_results"
    train(
        cfg=cfg,
        max_steps=2,
        run_id="f8_run",
        results_dir=results_dir,
        data_dir=tmp_path / "no_real_data",
    )

    run_dir = results_dir / "f8_run"
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "metrics.jsonl").exists()
    assert (run_dir / "ckpt" / "latest.pt").exists()


# ---------------------------------------------------------------- F7
def test_f7_vq_chain_reproducible_from_indices() -> None:
    torch.manual_seed(0)
    tok = MultiscaleTokenizer(TokenizerConfig(d=64, K=4, quantizer="vq", V_k=64, d_code=16), L=32, V=64)
    tok.eval()
    x = torch.randint(0, 64, (4, 32))
    enc = tok.encode(x)
    assert all(t.dtype == torch.long for t in enc.targets)
    rebuilt = tok.reconstruct(enc.targets)  # this is exactly what the generator will do
    assert len(rebuilt) == len(enc.zhat)
    for a, b in zip(enc.zhat, rebuilt):
        assert torch.allclose(a, b, atol=1e-4)


# ---------------------------------------------------------------- F10
def test_f10_level_schedule_rounds_half_up() -> None:
    from jncm.levels import get_level_schedule

    assert get_level_schedule(4, 3) == [1, 3, 4]  # tie at k=2: 1 + 1.5 = 2.5 -> 3 (banker's rounding gives 2)
    assert get_level_schedule(128, 11)[5] == 65  # tie at k=6: 1 + 5*127/10 = 64.5 -> 65
    assert get_level_schedule(128, 12) == [1, 13, 24, 36, 47, 59, 70, 82, 93, 105, 116, 128]
    assert get_level_schedule(32, 4) == [1, 11, 22, 32]
    for L, K in [(32, 4), (64, 8), (128, 11), (128, 12), (128, 32)]:
        sched = get_level_schedule(L, K)
        assert sched[0] == 1 and sched[-1] == L
        assert all(a < b for a, b in zip(sched, sched[1:]))
"""
Tests for tokenizer training script (TASKS T2.3).
Verifies execution, metrics logging, checkpointing, and resume on CPU.
"""

from pathlib import Path
import pytest
import torch

from jncm.config import load_config
from jncm.train_tokenizer import train
from jncm.utils import JSONLLogger


def test_train_tokenizer_smoke_vq(tmp_path: Path) -> None:
    cfg = load_config(Path("configs/smoke.yaml"))
    cfg.tokenizer.quantizer = "vq"
    cfg.train.eval_every = 5
    cfg.train.save_every = 10
    cfg.train.batch_size = 4
    cfg.device = "cpu"

    run_dir = tmp_path / "smoke_vq_run"
    train(cfg=cfg, max_steps=10, run_id=str(run_dir.name))

    out_dir = Path("results") / run_dir.name
    assert out_dir.exists()
    assert (out_dir / "config.yaml").exists()
    assert (out_dir / "git_sha.txt").exists()
    assert (out_dir / "metrics.jsonl").exists()
    assert (out_dir / "ckpt" / "latest.pt").exists()

    logger = JSONLLogger(out_dir / "metrics.jsonl")
    records = logger.read_all()
    assert len(records) >= 2

    last_record = records[-1]
    assert "acc_cum" in last_record
    assert len(last_record["acc_cum"]) == 4
    assert "acc_full" in last_record
    assert "tok_positions" in last_record
    assert last_record["tok_positions"] == 10 * 4 * cfg.L
    assert "codebook_usage" in last_record
    assert "codebook_perplexity" in last_record


def test_train_tokenizer_smoke_noisy(tmp_path: Path) -> None:
    cfg = load_config(Path("configs/smoke.yaml"))
    cfg.tokenizer.quantizer = "noisy"
    cfg.train.eval_every = 5
    cfg.train.save_every = 10
    cfg.train.batch_size = 4
    cfg.device = "cpu"

    run_dir = tmp_path / "smoke_noisy_run"
    train(cfg=cfg, max_steps=10, run_id=str(run_dir.name))

    out_dir = Path("results") / run_dir.name
    assert out_dir.exists()
    assert (out_dir / "metrics.jsonl").exists()

    logger = JSONLLogger(out_dir / "metrics.jsonl")
    records = logger.read_all()
    last_record = records[-1]
    assert "acc_cum" in last_record
    assert "acc_full" in last_record
    assert "tok_positions" in last_record
    assert "std_min" in last_record
    assert "std_mean" in last_record
    assert "eff_rank" in last_record
    assert "nominal_bits" in last_record


def test_train_tokenizer_resume(tmp_path: Path) -> None:
    cfg = load_config(Path("configs/smoke.yaml"))
    cfg.train.eval_every = 5
    cfg.train.save_every = 5
    cfg.train.batch_size = 4
    cfg.device = "cpu"

    run_name = f"resume_test_{tmp_path.name}"
    # Train 5 steps
    train(cfg=cfg, max_steps=5, run_id=run_name)

    ckpt_path = Path("results") / run_name / "ckpt" / "latest.pt"
    assert ckpt_path.exists()

    # Resume for 5 more steps (total 10 steps)
    train(cfg=cfg, max_steps=10, run_id=run_name, resume_path=ckpt_path)

    logger = JSONLLogger(Path("results") / run_name / "metrics.jsonl")
    records = logger.read_all()
    last_record = records[-1]
    assert last_record["step"] == 10
    assert last_record["tok_positions"] == 10 * 4 * cfg.L
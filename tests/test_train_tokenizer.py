"""
Tests for tokenizer training script (TASKS T2.3).
Verifies execution, metrics logging, checkpointing, and resume on CPU.
"""

from pathlib import Path
import pytest
import torch

from jncm.config import load_config
from jncm.train_tokenizer import setup_data, train
from jncm.utils import JSONLLogger


def test_setup_data_refuses_synthetic_for_real_runs(tmp_path: Path) -> None:
    cfg = load_config(Path("configs/smoke.yaml"))
    cfg.train.allow_synthetic_data = False
    with pytest.raises(FileNotFoundError):
        setup_data(cfg, data_dir=tmp_path / "missing_data")


def test_git_sha_marks_dirty_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    import jncm.train_tokenizer as tt

    def fake_check_output(cmd: list[str], stderr: object = None) -> bytes:
        if cmd[1] == "rev-parse":
            return b"deadbeef\n"
        if cmd[1] == "status":
            return b" M jncm/dirty_file.py\n"
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(tt.subprocess, "check_output", fake_check_output)
    assert tt.get_git_sha() == "deadbeef-dirty"


def test_optimizer_excludes_embeddings_norms_and_biases_from_decay() -> None:
    from jncm.tokenizer import MultiscaleTokenizer
    from jncm.train_tokenizer import build_optimizer

    cfg = load_config(Path("configs/smoke.yaml"))
    model = MultiscaleTokenizer(cfg.tokenizer, L=cfg.L, V=cfg.V)
    optimizer = build_optimizer(model, cfg)

    decay_ids = {
        id(p)
        for group in optimizer.param_groups
        if group["weight_decay"] > 0
        for p in group["params"]
    }
    no_decay_ids = {
        id(p)
        for group in optimizer.param_groups
        if group["weight_decay"] == 0
        for p in group["params"]
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        assert id(param) in decay_ids or id(param) in no_decay_ids
        if "emb" in name or param.ndim < 2:
            assert id(param) in no_decay_ids, name
        else:
            assert id(param) in decay_ids, name


def test_train_tokenizer_smoke_vq(tmp_path: Path) -> None:
    cfg = load_config(Path("configs/smoke.yaml"))
    cfg.tokenizer.quantizer = "vq"
    cfg.train.eval_every = 5
    cfg.train.save_every = 10
    cfg.train.batch_size = 4
    cfg.device = "cpu"

    run_dir = tmp_path / "smoke_vq_run"
    results_dir = tmp_path / "results"
    train(cfg=cfg, max_steps=10, run_id=str(run_dir.name), results_dir=results_dir)

    out_dir = results_dir / run_dir.name
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
    assert last_record["tok_positions"] + last_record["gen_positions"] == last_record["total_positions"]
    assert "nominal_bits" in last_record
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
    results_dir = tmp_path / "results"
    train(cfg=cfg, max_steps=10, run_id=str(run_dir.name), results_dir=results_dir)

    out_dir = results_dir / run_dir.name
    assert out_dir.exists()
    assert (out_dir / "metrics.jsonl").exists()

    logger = JSONLLogger(out_dir / "metrics.jsonl")
    records = logger.read_all()
    last_record = records[-1]
    assert "acc_cum" in last_record
    assert "acc_full" in last_record
    assert "tok_positions" in last_record
    assert last_record["tok_positions"] + last_record["gen_positions"] == last_record["total_positions"]
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
    results_dir = tmp_path / "results"
    # Train 5 steps
    train(cfg=cfg, max_steps=5, run_id=run_name, results_dir=results_dir)

    ckpt_path = results_dir / run_name / "ckpt" / "latest.pt"
    assert ckpt_path.exists()

    # Resume for 5 more steps (total 10 steps)
    train(
        cfg=cfg,
        max_steps=10,
        run_id=run_name,
        resume_path=ckpt_path,
        results_dir=results_dir,
    )

    logger = JSONLLogger(results_dir / run_name / "metrics.jsonl")
    records = logger.read_all()
    last_record = records[-1]
    assert last_record["step"] == 10
    assert last_record["tok_positions"] == 10 * 4 * cfg.L
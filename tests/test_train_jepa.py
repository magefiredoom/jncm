"""Tests for jncm/train_jepa.py (T5.3, SPEC section 8).

Acceptance from TASKS T5.3: a smoke overfit test, and every flag combination of
the ``decoder_through_E`` x ``vicreg`` 2x2 runs 50 CPU steps without error.
The other two ablation flags (``ema``, ``head``) are covered too.
"""

from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path

import numpy as np
import pytest
import torch

import jncm.train_jepa as tj
from jncm.config import Config, load_config
from jncm.tokenizer import MultiscaleTokenizer
from jncm.train_jepa import train, validate_jepa_config
from jncm.utils import JSONLLogger, seed_everything


def _cfg(head: str = "regress") -> Config:
    """Small noisy config for CPU tests (L=16, V=16, K=4, d=16)."""
    cfg = load_config("configs/smoke.yaml")
    cfg.L = 16
    cfg.V = 16
    cfg.device = "cpu"
    cfg.tokenizer.quantizer = "noisy"
    cfg.tokenizer.d = 16
    cfg.tokenizer.d_k = 4
    cfg.tokenizer.decoder_heads = 2
    cfg.generator.head = head
    cfg.generator.H = 16
    cfg.generator.n_layers = 1
    cfg.generator.n_heads = 2
    cfg.generator.dropout = 0.0
    cfg.generator.n_rep = 1
    cfg.generator.flow_steps = 2
    cfg.generator.corrupt.mode = "none"
    cfg.generator.train.batch_size = 4
    cfg.generator.train.lr = 3e-3
    cfg.generator.train.min_lr = 3e-4
    cfg.generator.train.warmup_steps = 2
    cfg.generator.train.weight_decay = 0.0
    cfg.generator.train.eval_every = 10
    cfg.generator.train.save_every = 50
    cfg.jepa.enabled = True
    return cfg


def _save_tokenizer_ckpt(
    tmp_path: Path, cfg: Config, tok_positions: int = 1234
) -> Path:
    """Write a tokenizer checkpoint in ``train_tokenizer``'s format."""
    seed_everything(0)
    tokenizer = MultiscaleTokenizer(cfg.tokenizer, L=cfg.L, V=cfg.V)
    path = tmp_path / "tok.pt"
    torch.save(
        {
            "step": 0,
            "model_state_dict": tokenizer.state_dict(),
            "config": asdict(cfg),
            "tok_positions": tok_positions,
        },
        path,
    )
    return path


def _write_data(directory: Path, sequences: torch.Tensor) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    flat = np.asarray(sequences, dtype=np.uint16).reshape(-1)
    flat.tofile(directory / "train.bin")
    flat.tofile(directory / "val.bin")
    return directory


def _run(tmp_path: Path, cfg: Config, steps: int, **kwargs) -> tuple[Path, list[dict]]:
    ckpt = kwargs.pop("ckpt", None) or _save_tokenizer_ckpt(tmp_path, cfg)
    data_dir = kwargs.pop("data_dir", None)
    if data_dir is None:
        data_dir = _write_data(
            tmp_path / "data", torch.randint(0, cfg.V, (4, cfg.L))
        )
    run_dir = train(
        cfg,
        ckpt,
        max_steps=steps,
        run_id="jepa_test",
        results_dir=tmp_path / "results",
        data_dir=data_dir,
        allow_synthetic=False,
        **kwargs,
    )
    return run_dir, JSONLLogger(run_dir / "metrics.jsonl").read_all()


# --------------------------------------------------------------------------- #
# Acceptance 1: the 2x2 flag grid runs 50 steps
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "decoder_through_E,vicreg",
    [(True, True), (True, False), (False, True), (False, False)],
    ids=["throughE+vicreg", "throughE", "vicreg", "neither"],
)
def test_2x2_flag_combination_runs_50_steps(
    tmp_path: Path, decoder_through_E: bool, vicreg: bool
) -> None:
    cfg = _cfg()
    cfg.jepa.decoder_through_E = decoder_through_E
    cfg.jepa.vicreg = vicreg
    cfg.generator.train.eval_every = 50
    cfg.generator.train.save_every = 50

    run_dir, records = _run(tmp_path, cfg, steps=50)

    assert [record["step"] for record in records] == [1, 50]
    last = records[-1]
    for key in (
        "train_loss",
        "loss_dec",
        "loss_pred",
        "loss_vic",
        "per_level_pred_loss",
        "r2",
        "std_min",
        "std_mean",
        "target_std_min",
        "target_std_mean",
        "eff_rank",
        "target_eff_rank",
        "collapse_steps",
        "collapse_alarm",
        "joint_positions",
        "total_positions",
        "total_positions_conservative",
    ):
        assert key in last, key
    assert math.isfinite(last["train_loss"]) and last["train_loss"] > 0.0
    assert last["decoder_through_E"] is decoder_through_E
    assert last["vicreg"] is vicreg
    assert last["ema"] is True
    assert last["head"] == "regress"
    assert last["target_chain"] == "online"
    if vicreg:
        assert last["loss_vic"] >= 0.0
    else:
        assert last["loss_vic"] == 0.0
    assert len(last["r2"]) == cfg.tokenizer.K
    assert len(last["collapse_steps"]) == cfg.tokenizer.K

    per_step = 50 * cfg.generator.train.batch_size * cfg.L
    assert last["joint_positions"] == per_step
    assert last["total_positions"] == per_step
    assert last["total_positions_conservative"] == 2 * per_step
    assert "tok_positions" not in last and "gen_positions" not in last
    assert last["train_loss"] == pytest.approx(
        last["loss_dec"] + cfg.jepa.lambda_pred * last["loss_pred"] + last["loss_vic"],
        rel=1e-5,
        abs=1e-6,
    )

    state = torch.load(run_dir / "ckpt" / "latest.pt", map_location="cpu")
    for key in ("tokenizer", "backbone", "head", "optimizer_state_dict", "config"):
        assert key in state, key
    assert state["step"] == 49
    assert state["joint_positions"] == per_step
    assert state["total_positions_conservative"] == 2 * per_step
    assert (state["ema"] is not None) is cfg.jepa.ema


def test_first_metrics_record_logs_parameter_counts(tmp_path: Path) -> None:
    """T5.5b F15: tokenizer, backbone, head and total counts on the first record."""
    cfg = _cfg()
    cfg.generator.train.save_milestones = False
    run_dir, records = _run(tmp_path, cfg, steps=2)

    tokenizer = MultiscaleTokenizer(cfg.tokenizer, L=cfg.L, V=cfg.V)
    backbone, head = tj.build_generator(cfg, tokenizer)
    expected = {
        "n_params_tokenizer": sum(p.numel() for p in tokenizer.parameters()),
        "n_params_backbone": sum(p.numel() for p in backbone.parameters()),
        "n_params_head": sum(p.numel() for p in head.parameters()),
    }
    expected["n_params_total"] = sum(expected.values())
    first = records[0]
    assert {key: first[key] for key in expected} == expected
    assert "n_params_total" not in records[-1]


def test_grad_accum_uses_one_optimizer_step_per_full_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T5.5a F13: 2 micro-batches, one step, full-batch position accounting."""
    cfg = _cfg()
    cfg.generator.train.batch_size = 4
    cfg.generator.train.grad_accum = 2
    cfg.generator.train.eval_every = 1
    cfg.generator.train.save_every = 100
    cfg.generator.train.save_milestones = False

    calls = {"count": 0}
    real_joint_loss = tj.joint_loss

    def counting_joint_loss(*args, **kwargs):
        calls["count"] += 1
        return real_joint_loss(*args, **kwargs)

    monkeypatch.setattr(tj, "joint_loss", counting_joint_loss)
    real_step = torch.optim.AdamW.step
    steps = {"count": 0}

    def spy(optimizer, *args, **kwargs):
        steps["count"] += 1
        return real_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", spy)
    _, records = _run(tmp_path, cfg, steps=2)

    assert calls["count"] == 4
    assert steps["count"] == 2
    per_step = cfg.generator.train.batch_size * cfg.L
    assert [record["joint_positions"] for record in records] == [
        per_step,
        2 * per_step,
    ]
    assert all(record["sec_per_step"] > 0.0 for record in records)


def test_validate_jepa_config_rejects_bad_grad_accum() -> None:
    """T5.5a F13: micro-batches must be equal and at least one."""
    cfg = _cfg()
    cfg.generator.train.grad_accum = 0
    with pytest.raises(ValueError, match="grad_accum"):
        validate_jepa_config(cfg)

    cfg.generator.train.grad_accum = 3  # batch 4 is not divisible by 3
    with pytest.raises(ValueError, match="divisible"):
        validate_jepa_config(cfg)


def test_joint_position_accounting(tmp_path: Path) -> None:
    """SPEC 6.2: joint steps are counted once, the conservative count twice."""
    cfg = _cfg()
    cfg.generator.train.eval_every = 1
    cfg.generator.train.save_every = 100

    run_dir, records = _run(tmp_path, cfg, steps=3)

    per_step = cfg.generator.train.batch_size * cfg.L
    assert [record["step"] for record in records] == [1, 2, 3]
    assert [record["joint_positions"] for record in records] == [
        per_step,
        2 * per_step,
        3 * per_step,
    ]
    for record in records:
        assert record["total_positions"] == record["joint_positions"]
        assert record["total_positions_conservative"] == 2 * record["joint_positions"]
        assert "tok_positions" not in record
        assert "gen_positions" not in record

    state = torch.load(run_dir / "ckpt" / "latest.pt", map_location="cpu")
    assert state["joint_positions"] == 3 * per_step
    # The tokenizer budget is still recorded, but SPEC 6.2 does not charge it.
    assert state["tok_offset"] == 1234


@pytest.mark.parametrize("target_chain", ["online", "ema_clean"])
def test_target_chain_runs_50_steps_with_flow(
    tmp_path: Path, target_chain: str
) -> None:
    cfg = _cfg(head="flow")
    cfg.jepa.target_chain = target_chain
    cfg.generator.train.eval_every = 50
    cfg.generator.train.save_every = 50

    run_dir, records = _run(tmp_path, cfg, steps=50)

    last = records[-1]
    assert last["step"] == 50
    assert last["head"] == "flow"
    assert last["target_chain"] == target_chain
    assert math.isfinite(last["train_loss"]) and last["train_loss"] > 0.0
    assert math.isfinite(last["loss_pred"]) and last["loss_pred"] > 0.0
    assert last["joint_positions"] == 50 * cfg.generator.train.batch_size * cfg.L
    state = torch.load(run_dir / "ckpt" / "latest.pt", map_location="cpu")
    assert any(key.startswith("nets.") for key in state["head"])


# --------------------------------------------------------------------------- #
# Acceptance 2: smoke overfit
# --------------------------------------------------------------------------- #


def test_smoke_overfit(tmp_path: Path) -> None:
    cfg = _cfg()
    cfg.jepa.decoder_through_E = True
    cfg.jepa.vicreg = True
    cfg.generator.train.eval_every = 25
    cfg.generator.train.save_every = 300
    steps = 300
    sequences = torch.randint(
        0, cfg.V, (cfg.generator.train.batch_size, cfg.L),
        generator=torch.Generator().manual_seed(7),
    )
    data_dir = _write_data(tmp_path / "data", sequences)

    _, records = _run(tmp_path, cfg, steps=steps, data_dir=data_dir)

    first, last = records[0], records[-1]
    mean_r2_first = float(np.mean(first["r2"]))
    mean_r2_last = float(np.mean(last["r2"]))
    # Measured on CPU: CE 3.0 -> 0.26-0.98 and mean R2 -0.38 -> 0.19-0.39.
    assert last["step"] == steps
    assert last["loss_dec"] < 0.4 * first["loss_dec"], (
        f"decode CE did not drop enough: {first['loss_dec']:.4f} -> "
        f"{last['loss_dec']:.4f}"
    )
    assert mean_r2_last > mean_r2_first + 0.4, (
        f"mean R2 did not improve: {mean_r2_first:.3f} -> {mean_r2_last:.3f}"
    )
    assert mean_r2_last > 0.15, f"mean R2 after overfit {mean_r2_last:.3f} <= 0.15"
    assert all(math.isfinite(value) for value in last["r2"])


# --------------------------------------------------------------------------- #
# Other flags, EMA wiring, resume, validation
# --------------------------------------------------------------------------- #


def test_ema_update_is_applied_and_saved(tmp_path: Path) -> None:
    cfg = _cfg()
    cfg.generator.train.eval_every = 10**9
    cfg.generator.train.save_every = 1
    ckpt = _save_tokenizer_ckpt(tmp_path, cfg)
    initial = torch.load(ckpt, map_location="cpu")["model_state_dict"][
        "tok_emb.weight"
    ]

    run_dir, _ = _run(tmp_path, cfg, steps=1, ckpt=ckpt)

    state = torch.load(run_dir / "ckpt" / "latest.pt", map_location="cpu")
    online = state["tokenizer"]["tok_emb.weight"]
    target = state["ema"]["module.tok_emb.weight"]
    momentum = cfg.jepa.m_start  # cosine schedule at step 0
    assert not torch.equal(online, target)
    assert torch.allclose(target, momentum * initial + (1 - momentum) * online, atol=1e-6)


def test_ema_flag_off_leaves_no_target_encoder(tmp_path: Path) -> None:
    cfg = _cfg()
    cfg.jepa.ema = False
    cfg.generator.train.eval_every = 10**9
    cfg.generator.train.save_every = 1

    run_dir, records = _run(tmp_path, cfg, steps=1)

    state = torch.load(run_dir / "ckpt" / "latest.pt", map_location="cpu")
    assert state["ema"] is None
    assert "ema_momentum" not in records[-1]
    assert records[-1]["ema"] is False


def test_flow_head_runs_and_logs_finite_metrics(tmp_path: Path) -> None:
    cfg = _cfg(head="flow")
    cfg.jepa.vicreg = False
    cfg.generator.train.eval_every = 10
    cfg.generator.train.save_every = 10

    run_dir, records = _run(tmp_path, cfg, steps=10)

    last = records[-1]
    assert last["step"] == 10
    assert last["head"] == "flow"
    assert math.isfinite(last["loss_pred"]) and last["loss_pred"] > 0.0
    assert len(last["per_level_pred_loss"]) == cfg.tokenizer.K
    # R2 is logged even with VICReg off (recomputed for diagnostics).
    assert all(math.isfinite(value) for value in last["r2"])
    state = torch.load(run_dir / "ckpt" / "latest.pt", map_location="cpu")
    assert any(key.startswith("nets.") for key in state["head"])


def test_resume_continues_step_and_positions(tmp_path: Path) -> None:
    cfg = _cfg()
    cfg.generator.train.eval_every = 2
    cfg.generator.train.save_every = 2
    ckpt = _save_tokenizer_ckpt(tmp_path, cfg, tok_positions=99)
    data_dir = _write_data(
        tmp_path / "data", torch.randint(0, cfg.V, (4, cfg.L))
    )

    run_dir, _ = _run(tmp_path, cfg, steps=2, ckpt=ckpt, data_dir=data_dir)
    latest = run_dir / "ckpt" / "latest.pt"
    _, records = _run(
        tmp_path,
        cfg,
        steps=4,
        ckpt=ckpt,
        data_dir=data_dir,
        resume_path=latest,
    )

    last = records[-1]
    assert last["step"] == 4
    assert last["joint_positions"] == 4 * cfg.generator.train.batch_size * cfg.L
    assert last["total_positions"] == last["joint_positions"]
    state = torch.load(latest, map_location="cpu")
    assert state["step"] == 3
    assert state["tok_offset"] == 99


def test_milestones_are_saved_behind_the_flag(tmp_path: Path) -> None:
    """T5.4: milestone checkpoints at 10/25/50/100% for run_eval."""
    cfg = _cfg()
    cfg.generator.train.milestones = [0.1, 0.5, 1.0]
    cfg.generator.train.eval_every = 10**9
    cfg.generator.train.save_every = 1000
    ckpt = _save_tokenizer_ckpt(tmp_path, cfg)
    data_dir = _write_data(tmp_path / "data", torch.randint(0, cfg.V, (4, cfg.L)))

    run_dir = train(
        cfg,
        ckpt,
        max_steps=10,
        run_id="jepa_milestones",
        results_dir=tmp_path / "results",
        data_dir=data_dir,
        allow_synthetic=False,
    )

    ckpt_dir = run_dir / "ckpt"
    names = sorted(path.name for path in ckpt_dir.glob("milestone_*.pt"))
    assert names == ["milestone_010.pt", "milestone_050.pt", "milestone_100.pt"]

    state = torch.load(ckpt_dir / "milestone_050.pt", map_location="cpu")
    assert state["milestone"] == pytest.approx(0.5)
    assert state["step"] + 1 == 5
    assert state["joint_positions"] == 5 * cfg.generator.train.batch_size * cfg.L
    assert state["total_positions_conservative"] == 2 * state["joint_positions"]
    for key in ("tokenizer", "backbone", "head", "ema"):
        assert key in state, key
    # T5.5b F14: milestone files carry no Adam state; latest.pt still does.
    assert "optimizer_state_dict" not in state
    latest = torch.load(ckpt_dir / "latest.pt", map_location="cpu")
    assert "optimizer_state_dict" in latest

    cfg.generator.train.save_milestones = False
    run_off = train(
        cfg,
        ckpt,
        max_steps=4,
        run_id="jepa_no_milestones",
        results_dir=tmp_path / "results",
        data_dir=data_dir,
        allow_synthetic=False,
    )
    assert not list((run_off / "ckpt").glob("milestone_*.pt"))


def test_validate_rejects_phase1_configs() -> None:
    vq = _cfg()
    vq.tokenizer.quantizer = "vq"
    vq.generator.head = "ce"
    with pytest.raises(ValueError):
        validate_jepa_config(vq)

    ce = _cfg()
    ce.generator.head = "ce"
    with pytest.raises(ValueError):
        validate_jepa_config(ce)

    negative = _cfg()
    negative.jepa.lambda_pred = -1.0
    with pytest.raises(ValueError):
        validate_jepa_config(negative)

    chain = _cfg()
    chain.jepa.target_chain = "noisy_clean"
    with pytest.raises(ValueError):
        validate_jepa_config(chain)


def test_train_rejects_vq_before_touching_files(tmp_path: Path) -> None:
    cfg = _cfg()
    cfg.tokenizer.quantizer = "vq"
    cfg.generator.head = "ce"
    with pytest.raises(ValueError):
        train(cfg, tmp_path / "missing.pt", max_steps=1, results_dir=tmp_path)

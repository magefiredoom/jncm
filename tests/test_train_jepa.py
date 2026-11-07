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
        "tok_positions",
        "gen_positions",
        "total_positions",
    ):
        assert key in last, key
    assert math.isfinite(last["train_loss"]) and last["train_loss"] > 0.0
    assert last["decoder_through_E"] is decoder_through_E
    assert last["vicreg"] is vicreg
    assert last["ema"] is True
    assert last["head"] == "regress"
    if vicreg:
        assert last["loss_vic"] >= 0.0
    else:
        assert last["loss_vic"] == 0.0
    assert len(last["r2"]) == cfg.tokenizer.K
    assert len(last["collapse_steps"]) == cfg.tokenizer.K

    per_step = 50 * cfg.generator.train.batch_size * cfg.L
    assert last["gen_positions"] == per_step
    assert last["tok_positions"] == 1234 + per_step
    assert last["total_positions"] == 1234 + 2 * per_step
    assert last["train_loss"] == pytest.approx(
        last["loss_dec"] + cfg.jepa.lambda_pred * last["loss_pred"] + last["loss_vic"],
        rel=1e-5,
        abs=1e-6,
    )

    state = torch.load(run_dir / "ckpt" / "latest.pt", map_location="cpu")
    for key in ("tokenizer", "backbone", "head", "optimizer_state_dict", "config"):
        assert key in state, key
    assert state["step"] == 49
    assert state["gen_positions"] == per_step
    assert state["tok_positions"] == 1234 + per_step
    assert (state["ema"] is not None) is cfg.jepa.ema


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
    assert last["gen_positions"] == 4 * cfg.generator.train.batch_size * cfg.L
    assert last["tok_positions"] == 99 + 4 * cfg.generator.train.batch_size * cfg.L
    state = torch.load(latest, map_location="cpu")
    assert state["step"] == 3
    assert state["tok_offset"] == 99


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


def test_train_rejects_vq_before_touching_files(tmp_path: Path) -> None:
    cfg = _cfg()
    cfg.tokenizer.quantizer = "vq"
    cfg.generator.head = "ce"
    with pytest.raises(ValueError):
        train(cfg, tmp_path / "missing.pt", max_steps=1, results_dir=tmp_path)

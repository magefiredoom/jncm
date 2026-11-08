"""T4.5: DONE marker, sec_per_step, and the templates' DONE early exit."""

from __future__ import annotations

from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
import torch

from jncm.config import Config, load_config
from jncm.tokenizer import MultiscaleTokenizer
from jncm.train_generator import train as train_generator
from jncm.train_jepa import train as train_jepa
from jncm.train_tokenizer import train as train_tokenizer
from jncm.utils import DONE_FILE, seed_everything


def _working_bash() -> str | None:
    candidates = [shutil.which("bash"), shutil.which("bash.exe")]
    if os.name == "nt":
        candidates += [
            r"C:\Program Files\Git\bin\bash.exe",
            r"C:\Program Files\Git\usr\bin\bash.exe",
        ]
    for candidate in candidates:
        if not candidate:
            continue
        probe = subprocess.run(
            [candidate, "-c", "echo ok"], capture_output=True, text=True
        )
        if probe.returncode == 0 and "ok" in probe.stdout:
            return candidate
    return None


BASH = _working_bash()


def _tiny_cfg(quantizer: str = "vq", head: str = "ce") -> Config:
    """Small CPU config (L=16, V=16, K=4, d=16) for a two-step run."""
    cfg = load_config(Path("configs/smoke.yaml"))
    cfg.L = 16
    cfg.V = 16
    cfg.device = "cpu"
    cfg.tokenizer.quantizer = quantizer
    cfg.tokenizer.V_k = 16
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
    cfg.generator.train.eval_every = 10
    cfg.generator.train.save_every = 50
    cfg.train.batch_size = 4
    cfg.train.eval_every = 10
    cfg.train.save_every = 50
    cfg.eval.n_samples = 2
    cfg.jepa.enabled = True
    return cfg


def _write_data(directory: Path, vocab: int = 16) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    tokens = np.random.RandomState(0).randint(0, vocab, (4, 16)).astype(np.uint16)
    tokens.tofile(directory / "train.bin")
    tokens.tofile(directory / "val.bin")
    return directory


def _save_tokenizer(tmp_path: Path, cfg: Config) -> Path:
    seed_everything(0)
    tokenizer = MultiscaleTokenizer(cfg.tokenizer, L=cfg.L, V=cfg.V)
    path = tmp_path / "tok.pt"
    torch.save(
        {
            "step": 0,
            "model_state_dict": tokenizer.state_dict(),
            "config": asdict(cfg),
            "tok_positions": 0,
        },
        path,
    )
    return path


def _records(run_dir: Path) -> list[dict]:
    text = (run_dir / "metrics.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _assert_finished(run_dir: Path, steps: int) -> None:
    path = run_dir / DONE_FILE
    assert path.is_file(), f"missing {path}"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["step"] == steps
    assert math.isfinite(payload["sec_per_step"]) and payload["sec_per_step"] > 0.0
    records = _records(run_dir)
    assert records, "no metrics records"
    for record in records:
        value = record["sec_per_step"]
        assert math.isfinite(value) and value > 0.0, record["step"]


@pytest.mark.parametrize("quantizer", ["vq", "noisy"])
def test_tokenizer_writes_done_and_sec_per_step(tmp_path: Path, quantizer: str) -> None:
    cfg = _tiny_cfg(quantizer=quantizer)

    run_dir = train_tokenizer(
        cfg,
        max_steps=2,
        run_id=f"tok_{quantizer}_done",
        results_dir=tmp_path / "results",
        data_dir=_write_data(tmp_path / "data"),
        allow_synthetic=True,
    )

    _assert_finished(run_dir, 2)


def test_generator_writes_done_and_sec_per_step(tmp_path: Path) -> None:
    cfg = _tiny_cfg()
    ckpt = _save_tokenizer(tmp_path, cfg)

    run_dir = train_generator(
        cfg,
        ckpt,
        max_steps=2,
        run_id="gen_done",
        results_dir=tmp_path / "results",
        data_dir=_write_data(tmp_path / "data"),
        allow_synthetic=True,
    )

    _assert_finished(run_dir, 2)


def test_jepa_writes_done_and_sec_per_step(tmp_path: Path) -> None:
    cfg = _tiny_cfg(quantizer="noisy", head="regress")
    ckpt = _save_tokenizer(tmp_path, cfg)

    run_dir = train_jepa(
        cfg,
        ckpt,
        max_steps=2,
        run_id="jepa_done",
        results_dir=tmp_path / "results",
        data_dir=_write_data(tmp_path / "data"),
        allow_synthetic=True,
    )

    _assert_finished(run_dir, 2)


# --- scripts/slurm/*.sbatch exit 0 immediately when DONE exists ---------------


def _fixture_tree(tmp_path: Path, run_id: str = "finished") -> Path:
    """Tree where the run is DONE and every later step would fail loudly."""
    root = tmp_path / "repo"
    (root / "results" / run_id).mkdir(parents=True)
    (root / "results" / run_id / DONE_FILE).write_text('{"step": 1}\n', encoding="utf-8")
    (root / "jncm").mkdir()
    for name in ("train_tokenizer.py", "train_generator.py", "train_jepa.py"):
        (root / "jncm" / name).write_text("", encoding="utf-8")
    venv = root / ".venv" / "bin"
    venv.mkdir(parents=True)
    (venv / "activate").write_text(
        "python() { printf 'PYTHON_CALLED\\n'; }\n", encoding="utf-8"
    )
    return root


def _run_template(
    root: Path, template: str, extra_env: dict[str, str]
) -> subprocess.CompletedProcess:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"QUANT", "HEAD", "TOKENIZER", "RUN_ID", "STEPS", "SLURM_SUBMIT_DIR"}
    }
    env.update({"SLURM_SUBMIT_DIR": str(root), "RUN_ID": "finished"})
    env.update(extra_env)
    return subprocess.run(
        [BASH, str(Path("scripts/slurm", template).resolve())],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
    )


@pytest.mark.skipif(BASH is None, reason="no working bash on this host")
@pytest.mark.parametrize(
    ("template", "extra_env"),
    [
        ("tok.sbatch", {"QUANT": "vq"}),
        ("gen.sbatch", {"HEAD": "ce", "TOKENIZER": "results/missing.pt"}),
        ("jepa.sbatch", {"TOKENIZER": "results/missing.pt"}),
    ],
)
def test_templates_exit_zero_when_done_exists(
    tmp_path: Path, template: str, extra_env: dict[str, str]
) -> None:
    root = _fixture_tree(tmp_path)

    result = _run_template(root, template, extra_env)

    assert result.returncode == 0, result.stderr
    assert "DONE exists" in result.stdout
    assert "PYTHON_CALLED" not in result.stdout

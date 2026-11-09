"""Utilities: seeding, JSONL metric logging, and bits-per-position computation."""

from __future__ import annotations

import json
import math
from pathlib import Path
import random
import time
from typing import Any
import numpy as np
import torch

# Marker written into results/<run_id>/ once the final training step completes
# (T4.5). The Slurm templates exit 0 immediately when it exists.
DONE_FILE = "DONE"


def write_done(run_dir: str | Path, *, step: int, sec_per_step: float) -> Path:
    """Write ``results/<run_id>/DONE`` after the final training step (T4.5).

    The marker is what makes a chained surplus chunk a cheap no-op: every
    template checks it before loading a venv or a dataset. ``sec_per_step`` is
    the measured wall time per step over the looping part of this process, so
    ``scripts/budget.py`` and ``scripts/launch_phase.sh`` can project a full
    run from a short pilot.

    Args:
        run_dir: ``results/<run_id>`` directory of the finished run.
        step: Final step number reached (``max_steps``).
        sec_per_step: Seconds per training step measured in this process.

    Returns:
        The path of the DONE file.
    """
    path = Path(run_dir) / DONE_FILE
    payload = {
        "step": int(step),
        "sec_per_step": float(sec_per_step),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return path


def parameter_counts(**modules: torch.nn.Module) -> dict[str, int]:
    """Parameter counts of named modules plus their total (T5.5b F15).

    Args:
        **modules: Named modules, e.g. ``tokenizer=``, ``backbone=``, ``head=``.

    Returns:
        ``{"n_params_<name>": int, ..., "n_params_total": int}``.
    """
    counts = {
        f"n_params_{name}": int(sum(p.numel() for p in module.parameters()))
        for name, module in modules.items()
    }
    counts["n_params_total"] = int(sum(counts.values()))
    return counts


def seed_everything(seed: int = 42) -> None:
    """Seed Python random, NumPy, and PyTorch (CPU and CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _to_json_serializable(val: Any) -> Any:
    """Convert tensor, numpy scalar/array, or Path to JSON-serializable types."""
    if isinstance(val, torch.Tensor):
        if val.numel() == 1:
            return val.item()
        return val.detach().cpu().tolist()
    if isinstance(val, (np.integer, np.int64, np.int32)):
        return int(val)
    if isinstance(val, (np.floating, np.float32, np.float64)):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, Path):
        return str(val)
    if isinstance(val, dict):
        return {k: _to_json_serializable(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_to_json_serializable(v) for v in val]
    return val


class JSONLLogger:
    """Thread-safe, append-only logger that writes JSON lines to a file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, metrics: dict[str, Any]) -> None:
        """Serialize metrics dict to JSON and append to file with immediate flush."""
        serializable = {k: _to_json_serializable(v) for k, v in metrics.items()}
        line = json.dumps(serializable)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()

    def read_all(self) -> list[dict[str, Any]]:
        """Read all logged JSON objects from the file."""
        if not self.path.exists():
            return []
        records = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    records.append(json.loads(stripped))
        return records


def bits_per_position(
    cfg: Any = None,
    *,
    quantizer: str | None = None,
    d_k: int | None = None,
    sigma_k: float | None = None,
    V_k: int | None = None,
) -> float:
    """
    Compute theoretical capacity bound in bits per position (SPEC 4.3).

    For vq: log2(V_k).
    For noisy: (d_k / 2) * log2(1 + 1 / sigma_k^2).
    """
    if cfg is not None:
        tok_cfg = getattr(cfg, "tokenizer", cfg)
        if quantizer is None:
            quantizer = getattr(tok_cfg, "quantizer", None)
            if quantizer is None and isinstance(tok_cfg, dict):
                quantizer = tok_cfg.get("quantizer")
        if d_k is None:
            d_k = getattr(tok_cfg, "d_k", None)
            if d_k is None and isinstance(tok_cfg, dict):
                d_k = tok_cfg.get("d_k")
        if sigma_k is None:
            sigma_k = getattr(tok_cfg, "sigma_k", None)
            if sigma_k is None and isinstance(tok_cfg, dict):
                sigma_k = tok_cfg.get("sigma_k")
        if V_k is None:
            V_k = getattr(tok_cfg, "V_k", None)
            if V_k is None and isinstance(tok_cfg, dict):
                V_k = tok_cfg.get("V_k")

    if quantizer == "vq":
        if V_k is None or V_k <= 0:
            raise ValueError(f"V_k must be positive for vq quantizer, got {V_k}")
        return float(math.log2(V_k))

    if quantizer == "noisy":
        if d_k is None or d_k <= 0:
            raise ValueError(f"d_k must be positive for noisy quantizer, got {d_k}")
        if sigma_k is None or sigma_k <= 0:
            raise ValueError(f"sigma_k must be positive for noisy quantizer, got {sigma_k}")
        snr = 1.0 / (sigma_k**2)
        return float((d_k / 2.0) * math.log2(1.0 + snr))

    raise ValueError(f"Unknown quantizer: {quantizer} (expected 'vq' or 'noisy')")
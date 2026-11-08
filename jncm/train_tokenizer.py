"""
Tokenizer training script (SPEC 4.4, 4.5, TASKS T2.3).
AdamW with cosine schedule, nested dropout, checkpointing, resume,
position accounting, and periodic diagnostics.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F

from jncm.config import Config, load_config, save_config
from jncm.data import InfiniteBatchIterator, MemmapDataset
from jncm.tokenizer.trunk import MultiscaleTokenizer
from jncm.utils import JSONLLogger, bits_per_position, seed_everything, write_done


def get_git_sha() -> str:
    """Retrieve current commit SHA, suffixed with '-dirty' if the tree is dirty."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        sha = out.decode("utf-8").strip()
    except Exception:
        return "unknown"
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        )
        if status.decode("utf-8").strip():
            return f"{sha}-dirty"
    except Exception:
        pass
    return sha


def build_optimizer(
    model: torch.nn.Module,
    cfg: Config,
) -> torch.optim.AdamW:
    """AdamW with weight decay only on weights (not embeddings, norms, or biases)."""
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "emb" in name:
            no_decay.append(param)
        else:
            decay.append(param)

    groups = [
        {"params": decay, "weight_decay": cfg.train.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    groups = [group for group in groups if group["params"]]
    return torch.optim.AdamW(
        groups,
        lr=cfg.train.lr,
        betas=tuple(cfg.train.betas),
    )


def get_lr(step: int, warmup_steps: int, max_steps: int, lr: float, min_lr: float) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if step < warmup_steps:
        return lr * float(step + 1) / float(max(1, warmup_steps))
    if step >= max_steps:
        return min_lr
    progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (lr - min_lr) * cosine_decay


def setup_data(
    cfg: Config,
    data_dir: str | Path = "data/tinystories",
    temp_dir: Path | None = None,
    allow_synthetic: bool | None = None,
) -> tuple[MemmapDataset, MemmapDataset]:
    """
    Load training and validation datasets.

    Real memmaps win whenever both exist. Synthetic data is created only when
    `allow_synthetic` (or cfg.train.allow_synthetic_data when None) is true.
    """
    d_path = Path(data_dir)
    train_bin = d_path / "train.bin"
    val_bin = d_path / "val.bin"

    # Use real datasets if present. Never silently substitute random data on a
    # real run: synthetic fallback must be explicitly enabled in the config.
    if train_bin.exists() and val_bin.exists():
        meta_path = d_path / "meta.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if int(meta.get("L", cfg.L)) != cfg.L:
                raise ValueError(
                    f"meta.json L={meta.get('L')} does not match config L={cfg.L}"
                )
            if int(meta.get("V", cfg.V)) != cfg.V:
                raise ValueError(
                    f"meta.json V={meta.get('V')} does not match config V={cfg.V}"
                )
        train_ds = MemmapDataset(train_bin, seq_len=cfg.L)
        val_ds = MemmapDataset(val_bin, seq_len=cfg.L)
        return train_ds, val_ds

    allowed = cfg.train.allow_synthetic_data if allow_synthetic is None else allow_synthetic
    if not allowed:
        raise FileNotFoundError(
            f"Prepared data not found at {d_path} (need train.bin and val.bin). "
            "Run scripts/prep_tinystories.py on a login node, or pass "
            "--allow_synthetic for a synthetic smoke run."
        )

    # Create synthetic dataset for CPU smoke tests and self-contained runs
    target_dir = temp_dir if temp_dir is not None else Path("data/synthetic_smoke")
    target_dir.mkdir(parents=True, exist_ok=True)
    syn_train = target_dir / f"train_syn_{cfg.L}_{cfg.V}.bin"
    syn_val = target_dir / f"val_syn_{cfg.L}_{cfg.V}.bin"

    rng = np.random.RandomState(42)
    if not syn_train.exists():
        train_tokens = rng.randint(0, cfg.V, size=(128, cfg.L), dtype=np.uint16)
        train_tokens.tofile(syn_train)
    if not syn_val.exists():
        val_tokens = rng.randint(0, cfg.V, size=(32, cfg.L), dtype=np.uint16)
        val_tokens.tofile(syn_val)

    return MemmapDataset(syn_train, seq_len=cfg.L), MemmapDataset(syn_val, seq_len=cfg.L)


def detect_data_source(data_dir: str | Path = "data/tinystories") -> str:
    """Return 'real' when both prepared memmaps exist, else 'synthetic'."""
    d_path = Path(data_dir)
    if (d_path / "train.bin").exists() and (d_path / "val.bin").exists():
        return "real"
    return "synthetic"


def run_diagnostics(
    model: MultiscaleTokenizer,
    val_batch: torch.Tensor,
    step: int,
    tok_positions: int,
    cfg: Config,
    loss_val: float,
    current_lr: float,
    sec_per_step: float,
    data_source: str = "unknown",
) -> dict[str, Any]:
    """
    Evaluate full hierarchy on fixed held-out batch without nested dropout (SPEC 4.5).
    """
    model.eval()
    with torch.no_grad():
        enc = model.encode(val_batch, cutoff=None)

        acc_cum: list[float] = []
        for k in range(1, model.K + 1):
            logits_k = model.decode(enc.zhat[k])
            preds_k = torch.argmax(logits_k, dim=-1)
            acc_k = (preds_k == val_batch).float().mean().item()
            acc_cum.append(float(acc_k))

        acc_full = acc_cum[-1]

        record: dict[str, Any] = {
            "step": step,
            "train_loss": float(loss_val),
            "lr": float(current_lr),
            "sec_per_step": float(sec_per_step),
            "data_source": data_source,
            "acc_cum": acc_cum,
            "acc_full": float(acc_full),
            "tok_positions": tok_positions,
            "gen_positions": 0,
            "total_positions": tok_positions,
        }

        # Nominal capacity is logged for both variants (SPEC 4.3).
        record["nominal_bits"] = bits_per_position(cfg)

        if cfg.tokenizer.quantizer == "vq":
            usages = []
            perplexities = []
            for k in range(model.K):
                diag = model.quantizers[k].compute_diagnostics(enc.targets[k])
                usages.append(float(diag["usage_frac"]))
                perplexities.append(float(diag["perplexity"]))
            record["codebook_usage"] = usages
            record["codebook_perplexity"] = perplexities
        elif cfg.tokenizer.quantizer == "noisy":
            stds_min = []
            stds_mean = []
            eff_ranks = []
            for k in range(model.K):
                diag = model.quantizers[k].compute_diagnostics(enc.targets[k])
                stds_min.append(float(diag["std_min"]))
                stds_mean.append(float(diag["std_mean"]))
                eff_ranks.append(float(diag["eff_rank"]))
            record["std_min"] = stds_min
            record["std_mean"] = stds_mean
            record["eff_rank"] = eff_ranks

    model.train()
    return record


def train(
    cfg: Config,
    max_steps: int | None = None,
    run_id: str | None = None,
    resume_path: str | Path | None = None,
    results_dir: str | Path = "results",
    allow_synthetic: bool | None = None,
    data_dir: str | Path = "data/tinystories",
) -> Path:
    """Execute tokenizer training loop."""
    seed_everything(cfg.seed)

    if max_steps is not None:
        cfg.train.max_steps = max_steps

    device_str = cfg.device
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    if run_id is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        run_id = f"tok_{cfg.tokenizer.quantizer}_{ts}"

    run_dir = Path(results_dir) / run_id
    ckpt_dir = run_dir / "ckpt"
    samples_dir = run_dir / "samples"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    save_config(cfg, run_dir / "config.yaml")
    (run_dir / "git_sha.txt").write_text(get_git_sha(), encoding="utf-8")
    logger = JSONLLogger(run_dir / "metrics.jsonl")

    # Data
    data_source = detect_data_source(data_dir)
    train_ds, val_ds = setup_data(
        cfg,
        data_dir=data_dir,
        allow_synthetic=allow_synthetic,
    )
    train_loader = InfiniteBatchIterator(
        train_ds, batch_size=cfg.train.batch_size, seed=cfg.seed, shuffle=True
    )
    val_loader = InfiniteBatchIterator(
        val_ds, batch_size=cfg.train.batch_size, seed=cfg.seed + 1, shuffle=False
    )
    fixed_val_batch = next(val_loader).to(device)

    # Model
    model = MultiscaleTokenizer(cfg.tokenizer, L=cfg.L, V=cfg.V).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = build_optimizer(model, cfg)

    start_step = 0
    if resume_path:
        r_path = Path(resume_path)
        if not r_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {r_path}")
        checkpoint = torch.load(r_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = checkpoint["step"] + 1
        print(f"Resumed from step {start_step}")

    model.train()
    use_autocast = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_autocast and torch.cuda.is_bf16_supported()) else torch.float32

    print(f"Starting training run {run_id} ({cfg.tokenizer.quantizer}) for {cfg.train.max_steps} steps on {device}...")

    recent_loss = 0.0
    sec_per_step = 0.0
    loop_start = time.time()
    for step in range(start_step, cfg.train.max_steps):
        curr_lr = get_lr(
            step,
            warmup_steps=cfg.train.warmup_steps,
            max_steps=cfg.train.max_steps,
            lr=cfg.train.lr,
            min_lr=cfg.train.min_lr,
        )
        for pg in optimizer.param_groups:
            pg["lr"] = curr_lr

        x_batch = next(train_loader).to(device)
        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_autocast):
            logits, enc, cut = model(x_batch, p_drop=cfg.tokenizer.p_drop)
            ce_loss = F.cross_entropy(logits.view(-1, cfg.V), x_batch.view(-1))
            total_loss = ce_loss + enc.quant_loss

        total_loss.backward()
        if cfg.train.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, cfg.train.grad_clip)
        optimizer.step()

        # Dead code reinitialization for VQ every dead_code_reinit_steps
        if cfg.tokenizer.quantizer == "vq" and (step + 1) % cfg.tokenizer.dead_code_reinit_steps == 0:
            for q in model.quantizers:
                q.reinit_dead_codes()

        recent_loss = total_loss.item()
        tok_positions = (step + 1) * cfg.train.batch_size * cfg.L
        sec_per_step = (time.time() - loop_start) / float(step + 1 - start_step)

        # Periodic diagnostics
        is_last = (step + 1 == cfg.train.max_steps)
        if (step + 1) % cfg.train.eval_every == 0 or is_last or step == 0:
            metrics = run_diagnostics(
                model=model,
                val_batch=fixed_val_batch,
                step=step + 1,
                tok_positions=tok_positions,
                cfg=cfg,
                loss_val=recent_loss,
                current_lr=curr_lr,
                sec_per_step=sec_per_step,
                data_source=data_source,
            )
            logger.log(metrics)
            print(
                f"Step {step + 1:5d} | Loss: {recent_loss:.4f} | "
                f"acc_full: {metrics['acc_full']:.4f} | "
                f"tok_pos: {tok_positions}"
            )

        # Periodic checkpoint
        if (step + 1) % cfg.train.save_every == 0 or is_last:
            ckpt_file = ckpt_dir / f"step_{step + 1:06d}.pt"
            state = {
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": asdict(cfg),
                "tok_positions": tok_positions,
            }
            torch.save(state, ckpt_file)
            torch.save(state, ckpt_dir / "latest.pt")

    write_done(run_dir, step=cfg.train.max_steps, sec_per_step=sec_per_step)
    print(f"Training completed. Results written to {run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Multiscale Tokenizer")
    parser.add_argument("--config", type=str, default="configs/smoke.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--allow_synthetic",
        action="store_true",
        help="Explicitly allow random synthetic data when real memmaps are missing",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    train(
        cfg=cfg,
        max_steps=args.steps,
        run_id=args.run_id,
        resume_path=args.resume,
        allow_synthetic=True if args.allow_synthetic else None,
    )


if __name__ == "__main__":
    main()
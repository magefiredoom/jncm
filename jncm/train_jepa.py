"""JEPA joint training script (SPEC section 8, TASKS T5.3).

Trains the online noisy tokenizer (encoder E and decoder D) and the generator
(backbone G and ``regress``/``flow`` head) jointly against EMA targets. The
ablation flags live in the config and are all consumed here or in
``jncm.jepa.joint``:

- ``jepa.ema``: targets from the EMA copy (true) or the online encoder under
  stop-gradient (false).
- ``jepa.vicreg``: add the per-level VICReg terms on ``c_k`` and ``c-hat_k``.
- ``jepa.decoder_through_E``: let ``L_dec`` backpropagate into E (false detaches
  the decoder input).
- ``jepa.target_chain``: ``online`` (default) teacher-forces E-bar on the online
  chain; ``ema_clean`` uses E-bar's own noise-free chain (the v1.3 definition).
- ``generator.head``: ``regress`` or ``flow`` (``ce`` and ``vq`` are Phase 1).

Periodic diagnostics log, per level, the min/mean std of ``c_k`` and ``t_k``,
both effective ranks, the explained variance ``R2_k = 1 - MSE_k / Var(t_k)``,
and a collapse alarm that fires after 200 consecutive steps with mean std < 0.1
(SPEC 8). Every record carries the SPEC 6.2 joint accounting: ``joint_positions``
= steps * B * L, ``total_positions = joint_positions`` and
``total_positions_conservative = 2 * joint_positions`` (the count comparable to
a staged run when both objectives are charged).

Milestone checkpoints (``ckpt/milestone_<pct>.pt`` at 10/25/50/100% of the
budget) are written when ``generator.train.save_milestones`` is on (the
default), matching the Phase-1 trainer; ``scripts/run_eval.py`` evaluates them.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import time
from typing import Any

import torch

from jncm.config import Config, load_config, save_config
from jncm.data import InfiniteBatchIterator
from jncm.jepa.diagnostics import CollapseAlarm, code_diagnostics
from jncm.jepa.ema import EMAModule
from jncm.jepa.joint import (
    JEPA_HEADS, TARGET_CHAINS, JointStep, joint_loss, predicted_codes,
)
from jncm.train_generator import (
    build_generator,
    build_optimizer,
    load_tokenizer,
    milestone_steps,
)
from jncm.train_tokenizer import (
    detect_data_source,
    get_git_sha,
    get_lr,
    setup_data,
)
from jncm.utils import JSONLLogger, seed_everything, write_done


def validate_jepa_config(cfg: Config) -> None:
    """Fail loudly on a Phase-1 config or a negative JEPA weight (SPEC 8).

    Raises:
        ValueError: if the quantizer is not ``noisy``, the head is not
            ``regress``/``flow``, or a loss weight is negative.
    """
    if cfg.tokenizer.quantizer != "noisy":
        raise ValueError(
            "JEPA joint mode requires quantizer='noisy' (SPEC 8), got "
            f"{cfg.tokenizer.quantizer!r}"
        )
    if cfg.generator.head not in JEPA_HEADS:
        raise ValueError(
            f"JEPA joint mode needs head in {JEPA_HEADS}, got "
            f"{cfg.generator.head!r}"
        )
    if cfg.jepa.target_chain not in TARGET_CHAINS:
        raise ValueError(
            f"jepa.target_chain must be one of {TARGET_CHAINS}, got "
            f"{cfg.jepa.target_chain!r}"
        )
    for name in ("lambda_pred", "lambda_var", "lambda_cov", "gamma"):
        value = getattr(cfg.jepa, name)
        if float(value) < 0.0:
            raise ValueError(f"jepa.{name} must be >= 0, got {value}")
    if cfg.generator.train.batch_size < 1:
        raise ValueError(
            f"generator.train.batch_size must be >= 1, got {cfg.generator.train.batch_size}"
        )


def joint_named_parameters(
    tokenizer: torch.nn.Module,
    backbone: torch.nn.Module,
    head: torch.nn.Module,
) -> list[tuple[str, torch.nn.Parameter]]:
    """Named parameters of E+D+G for the weight-decay grouping."""
    return (
        list(tokenizer.named_parameters())
        + list(backbone.named_parameters())
        + list(head.named_parameters())
    )


def load_online_tokenizer(
    ckpt_path: str | Path, cfg: Config
) -> tuple[torch.nn.Module, int]:
    """Load the tokenizer checkpoint with gradients enabled (online E and D).

    Returns:
        ``(tokenizer, tok_positions)``; ``tok_positions`` is the tokenizer
        budget recorded in the checkpoint (kept as bookkeeping; SPEC 6.2 does
        not charge it to a joint run).
    """
    tokenizer, tok_positions = load_tokenizer(ckpt_path, cfg)
    for parameter in tokenizer.parameters():
        parameter.requires_grad_(True)
    return tokenizer, tok_positions


def run_diagnostics(
    joint: JointStep,
    head: torch.nn.Module,
    cfg: Config,
    *,
    step: int,
    train_loss: float,
    current_lr: float,
    joint_positions: int,
    sec_per_step: float,
    data_source: str,
    alarm: CollapseAlarm,
    momentum: float | None = None,
    generator: torch.Generator | None = None,
) -> dict[str, Any]:
    """One SPEC 8 metrics record from a finished joint step.

    The VICReg predictions are reused when they exist; otherwise ``c-hat_k`` is
    recomputed under ``no_grad`` (with a diagnostic-only RNG) purely for R2 and
    the per-level std diagnostics, so R2 is logged whether or not VICReg runs.
    """
    predictions = list(joint.predictions)
    if not predictions:
        with torch.no_grad():
            predictions = [
                p.detach()
                for p in predicted_codes(head, joint.hidden, generator=generator)
            ]
    diag = code_diagnostics(joint.codes, joint.targets, predictions)
    alarmed = alarm.update(diag["std_mean"])

    record: dict[str, Any] = {
        "step": int(step),
        "train_loss": float(train_loss),
        "loss_dec": float(joint.loss_dec.detach()),
        "loss_pred": float(joint.loss_pred.detach()),
        "loss_vic": float(joint.loss_vic.detach()),
        "per_level_pred_loss": [float(value.detach()) for value in joint.per_level_pred],
        "lr": float(current_lr),
        "sec_per_step": float(sec_per_step),
        "data_source": data_source,
        "cut": int(joint.cut),
        "head": cfg.generator.head,
        "ema": bool(cfg.jepa.ema),
        "target_chain": cfg.jepa.target_chain,
        "vicreg": bool(cfg.jepa.vicreg),
        "decoder_through_E": bool(cfg.jepa.decoder_through_E),
        "joint_positions": int(joint_positions),
        "total_positions": int(joint_positions),
        "total_positions_conservative": int(2 * joint_positions),
        "collapse_steps": list(alarm.counts),
        "collapse_alarm": bool(any(alarmed)),
    }
    if momentum is not None:
        record["ema_momentum"] = float(momentum)
    record.update(diag)
    return record


def train(
    cfg: Config,
    tokenizer_ckpt: str | Path,
    max_steps: int | None = None,
    run_id: str | None = None,
    resume_path: str | Path | None = None,
    results_dir: str | Path = "results",
    data_dir: str | Path = "data/tinystories",
    allow_synthetic: bool | None = None,
) -> Path:
    """Jointly train E, D and G against EMA targets (SPEC 8).

    Returns the run directory under ``results_dir``.
    """
    validate_jepa_config(cfg)
    seed_everything(cfg.seed)
    gen_train = cfg.generator.train
    if max_steps is not None:
        gen_train.max_steps = max_steps

    device_str = cfg.device
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    if run_id is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        run_id = f"jepa_{cfg.generator.head}_{ts}"

    run_dir = Path(results_dir) / run_id
    ckpt_dir = run_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    save_config(cfg, run_dir / "config.yaml")
    (run_dir / "git_sha.txt").write_text(get_git_sha(), encoding="utf-8")
    logger = JSONLLogger(run_dir / "metrics.jsonl")

    data_source = detect_data_source(data_dir)
    train_ds, _ = setup_data(
        cfg, data_dir=data_dir, allow_synthetic=allow_synthetic
    )
    train_loader = InfiniteBatchIterator(
        train_ds, batch_size=gen_train.batch_size, seed=cfg.seed, shuffle=True
    )

    tokenizer, tok_offset = load_online_tokenizer(tokenizer_ckpt, cfg)
    tokenizer = tokenizer.to(device)
    tokenizer.train()
    backbone, head = build_generator(cfg, tokenizer)
    backbone = backbone.to(device).train()
    head = head.to(device).train()
    optimizer = build_optimizer(
        joint_named_parameters(tokenizer, backbone, head),
        lr=gen_train.lr,
        weight_decay=gen_train.weight_decay,
        betas=gen_train.betas,
    )
    target_encoder = (
        EMAModule(
            tokenizer,
            total_steps=gen_train.max_steps,
            momentum=cfg.jepa.m_start,
            final_momentum=cfg.jepa.m_end,
        )
        if cfg.jepa.ema
        else None
    )
    alarm = CollapseAlarm()
    parameters = [
        parameter for _, parameter in joint_named_parameters(tokenizer, backbone, head)
    ]

    start_step = 0
    if resume_path:
        r_path = Path(resume_path)
        if not r_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {r_path}")
        checkpoint = torch.load(r_path, map_location=device)
        tokenizer.load_state_dict(checkpoint["tokenizer"])
        backbone.load_state_dict(checkpoint["backbone"])
        head.load_state_dict(checkpoint["head"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if target_encoder is not None and checkpoint.get("ema") is not None:
            target_encoder.load_state_dict(checkpoint["ema"])
        alarm.load_state_dict(checkpoint.get("collapse_alarm") or {})
        start_step = checkpoint["step"] + 1
        tok_offset = int(checkpoint.get("tok_offset", tok_offset))
        print(f"Resumed from step {start_step}")

    use_autocast = device.type == "cuda"
    amp_dtype = (
        torch.bfloat16
        if (use_autocast and torch.cuda.is_bf16_supported())
        else torch.float32
    )
    rng = torch.Generator(device=device).manual_seed(cfg.seed)

    print(
        f"Starting JEPA joint run {run_id} (head={cfg.generator.head}, "
        f"ema={cfg.jepa.ema}, vicreg={cfg.jepa.vicreg}, "
        f"decoder_through_E={cfg.jepa.decoder_through_E}) "
        f"for {gen_train.max_steps} steps on {device}..."
    )

    milestones = (
        milestone_steps(gen_train.max_steps, gen_train.milestones)
        if gen_train.save_milestones
        else {}
    )

    sec_per_step = 0.0
    loop_start = time.time()
    for step in range(start_step, gen_train.max_steps):
        curr_lr = get_lr(
            step,
            warmup_steps=gen_train.warmup_steps,
            max_steps=gen_train.max_steps,
            lr=gen_train.lr,
            min_lr=gen_train.min_lr,
        )
        for group in optimizer.param_groups:
            group["lr"] = curr_lr

        x_batch = next(train_loader).to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=use_autocast
        ):
            joint = joint_loss(
                cfg,
                tokenizer,
                backbone,
                head,
                x_batch,
                target_tokenizer=None if target_encoder is None else target_encoder.module,
                generator=rng,
            )
        joint.loss.backward()
        if gen_train.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(parameters, gen_train.grad_clip)
        optimizer.step()

        momentum = target_encoder.update(tokenizer, step) if target_encoder else None

        joint_positions = (step + 1) * gen_train.batch_size * cfg.L
        sec_per_step = (time.time() - loop_start) / float(step + 1 - start_step)

        is_last = step + 1 == gen_train.max_steps
        if (step + 1) % gen_train.eval_every == 0 or is_last or step == 0:
            diag_generator = torch.Generator(device=device).manual_seed(
                cfg.seed + step + 1
            )
            metrics = run_diagnostics(
                joint,
                head,
                cfg,
                step=step + 1,
                train_loss=float(joint.loss.detach()),
                current_lr=curr_lr,
                joint_positions=joint_positions,
                sec_per_step=sec_per_step,
                data_source=data_source,
                alarm=alarm,
                momentum=momentum,
                generator=diag_generator,
            )
            logger.log(metrics)
            mean_r2 = sum(metrics["r2"]) / len(metrics["r2"])
            print(
                f"Step {step + 1:5d} | Loss: {metrics['train_loss']:.4f} | "
                f"dec {metrics['loss_dec']:.4f} | pred {metrics['loss_pred']:.4f} | "
                f"vic {metrics['loss_vic']:.4f} | R2 {mean_r2:.3f} | "
                f"joint_pos: {joint_positions}"
            )

        milestone = milestones.get(step + 1)
        is_periodic = (step + 1) % gen_train.save_every == 0 or is_last
        if is_periodic or milestone is not None:
            state = {
                "step": step,
                "tokenizer": tokenizer.state_dict(),
                "backbone": backbone.state_dict(),
                "head": head.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "ema": target_encoder.state_dict() if target_encoder else None,
                "collapse_alarm": alarm.state_dict(),
                "config": asdict(cfg),
                "tok_offset": tok_offset,
                "joint_positions": joint_positions,
                "total_positions": joint_positions,
                "total_positions_conservative": 2 * joint_positions,
            }
            if milestone is not None:
                torch.save(
                    {**state, "milestone": milestone},
                    ckpt_dir / f"milestone_{round(milestone * 100):03d}.pt",
                )
            if is_periodic:
                torch.save(state, ckpt_dir / f"step_{step + 1:06d}.pt")
                torch.save(state, ckpt_dir / "latest.pt")

    write_done(run_dir, step=gen_train.max_steps, sec_per_step=sec_per_step)
    print(f"Training completed. Results written to {run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the JEPA joint mode (SPEC 8)")
    parser.add_argument("--config", type=str, default="configs/smoke.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Tokenizer checkpoint to start from (results/<tok_run>/ckpt/latest.pt)",
    )
    parser.add_argument("--data_dir", type=str, default="data/tinystories")
    parser.add_argument(
        "--allow_synthetic",
        action="store_true",
        help="Explicitly allow random synthetic data when real memmaps are missing",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    train(
        cfg=cfg,
        tokenizer_ckpt=args.tokenizer,
        max_steps=args.steps,
        run_id=args.run_id,
        resume_path=args.resume,
        data_dir=args.data_dir,
        allow_synthetic=True if args.allow_synthetic else None,
    )


if __name__ == "__main__":
    main()

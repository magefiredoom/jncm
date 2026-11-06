"""Generator training script (SPEC 5.5 to 5.7, TASKS T3.5).

The tokenizer is frozen: it is loaded from a tokenizer checkpoint, put in eval
mode, and its parameters get no gradients. One training step

1. encodes the token batch with the frozen tokenizer and collects the chain
   codes and the clean targets (SPEC 5.1),
2. corrupts the chain codes once for the whole hierarchy; targets stay clean
   (SPEC 5.4),
3. runs all K levels through the backbone in one packed pass (SPEC 5.2),
4. minimizes the position-weighted sum of the per-level losses (SPEC 5.5).

Periodic diagnostics sample the hierarchy coarse to fine with ``sample_tokens``
(SPEC 5.6), log the unigram entropy of the generated dataset tokens, optionally
compute Gen PPL through a scorer, and write up to 32 decoded samples to
``results/<run_id>/samples/step_<n>.txt`` (SPEC 6.3).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import time
from typing import Any, Callable, Iterable

import torch

from jncm.config import Config, load_config, save_config
from jncm.data import InfiniteBatchIterator
from jncm.eval.genppl import gen_ppl, unigram_entropy
from jncm.generator.backbone import GeneratorBackbone
from jncm.generator.corruption import corrupt_codes
from jncm.generator.heads import LevelHead, build_head, weighted_total
from jncm.generator.sample import sample_tokens
from jncm.tokenizer.trunk import MultiscaleTokenizer
from jncm.train_tokenizer import (
    detect_data_source,
    get_git_sha,
    get_lr,
    setup_data,
)
from jncm.utils import JSONLLogger, seed_everything

DecodeFn = Callable[[list[int]], str]


def load_tokenizer(
    ckpt_path: str | Path,
    cfg: Config,
) -> tuple[MultiscaleTokenizer, int]:
    """Load the frozen tokenizer from a tokenizer checkpoint (SPEC 5.1).

    Args:
        ckpt_path: Path to a checkpoint written by ``train_tokenizer``.
        cfg: Config whose ``tokenizer`` section and ``L``/``V`` must match the
            checkpoint.

    Returns:
        ``(tokenizer, tok_positions)`` where the tokenizer is in eval mode with
        ``requires_grad`` off, and ``tok_positions`` is the tokenizer budget
        recorded in the checkpoint (0 if the checkpoint has none).
    """
    path = Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint {path} has no 'model_state_dict'")

    saved = checkpoint.get("config") or {}
    if saved:
        if int(saved.get("L", cfg.L)) != cfg.L or int(saved.get("V", cfg.V)) != cfg.V:
            raise ValueError(
                f"tokenizer checkpoint L={saved.get('L')}, V={saved.get('V')} does "
                f"not match config L={cfg.L}, V={cfg.V}"
            )
        saved_tok = saved.get("tokenizer") or {}
        if saved_tok.get("quantizer", cfg.tokenizer.quantizer) != cfg.tokenizer.quantizer:
            raise ValueError(
                "tokenizer checkpoint variant "
                f"{saved_tok.get('quantizer')!r} does not match config "
                f"quantizer {cfg.tokenizer.quantizer!r}"
            )
        if int(saved_tok.get("K", cfg.tokenizer.K)) != cfg.tokenizer.K:
            raise ValueError(
                f"tokenizer checkpoint K={saved_tok.get('K')} does not match "
                f"config K={cfg.tokenizer.K}"
            )

    model = MultiscaleTokenizer(cfg.tokenizer, L=cfg.L, V=cfg.V)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, int(checkpoint.get("tok_positions", 0))


def build_generator(
    cfg: Config,
    tokenizer: MultiscaleTokenizer,
) -> tuple[GeneratorBackbone, LevelHead]:
    """Build the backbone and the configured head for a frozen tokenizer.

    ``ce`` is the discrete (vq) head; ``regress`` and ``flow`` are the
    continuous (noisy) heads (SPEC 5.3).
    """
    if tokenizer.quantizer_type != cfg.tokenizer.quantizer:
        raise ValueError(
            f"tokenizer is {tokenizer.quantizer_type!r} but config says "
            f"{cfg.tokenizer.quantizer!r}"
        )
    if cfg.generator.head == "ce" and cfg.tokenizer.quantizer != "vq":
        raise ValueError("head 'ce' is for the vq variant (SPEC 5.3)")
    if cfg.generator.head in ("regress", "flow") and cfg.tokenizer.quantizer != "noisy":
        raise ValueError(
            f"head {cfg.generator.head!r} is for the noisy variant (SPEC 5.3)"
        )
    backbone = GeneratorBackbone.from_tokenizer(tokenizer, cfg.generator)
    head = build_head(
        cfg.generator,
        H=cfg.generator.H,
        K=tokenizer.K,
        V_k=cfg.tokenizer.V_k,
        d_k=cfg.tokenizer.d_k,
        sigma_k=cfg.tokenizer.sigma_k,
    )
    return backbone, head


def build_optimizer(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    *,
    lr: float,
    weight_decay: float,
    betas: list[float],
) -> torch.optim.AdamW:
    """AdamW with weight decay only on weights (not embeddings, norms, biases)."""
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, param in named_parameters:
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "emb" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    groups = [group for group in groups if group["params"]]
    return torch.optim.AdamW(groups, lr=lr, betas=tuple(betas))


def compute_generator_loss(
    tokenizer: MultiscaleTokenizer,
    backbone: GeneratorBackbone,
    head: LevelHead,
    x: torch.Tensor,
    cfg: Config,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """One SPEC 5.5 loss for a token batch ``x`` of shape [B, L].

    The frozen tokenizer runs under ``no_grad``; its chain codes are corrupted
    once for the whole hierarchy while the targets stay clean; all K levels go
    through the backbone in one packed forward pass. All randomness (channel
    noise, corruption draws, flow-matching draws) consumes the single
    ``generator`` in that order, so a seed reproduces the step.

    Returns:
        (weighted total loss, list of per-level mean losses).
    """
    with torch.no_grad():
        enc = tokenizer.encode(x, generator=generator)
        codes = corrupt_codes(
            enc.codes,
            cfg.generator.corrupt,
            generator=generator,
            quantizer=cfg.tokenizer.quantizer,
        )
        level_inputs = backbone.embed_levels(tokenizer, codes)
    hidden = backbone.forward_levels(level_inputs)
    _, per_level = head.loss(hidden, enc.targets, generator=generator)
    total = weighted_total(per_level, backbone.schedule, cfg.generator.alpha)
    return total, per_level


def sample_and_score(
    tokenizer: MultiscaleTokenizer,
    backbone: GeneratorBackbone,
    head: LevelHead,
    n_samples: int,
    generator: torch.Generator | None = None,
    decode_fn: DecodeFn | None = None,
    scorer: Any = None,
) -> tuple[torch.Tensor, list[str] | None, dict[str, float]]:
    """Sample ``n_samples`` sequences and score them (SPEC 5.6/6.1).

    The modules are switched to eval mode for sampling and restored to their
    previous train/eval modes afterwards.

    Returns:
        ``(tokens [n_samples, L], texts or None, metrics)`` where metrics has
        ``sample_entropy`` (nats, dataset tokens) and, when both a decoder and
        a scorer are given, ``gen_ppl``.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    was_training = (tokenizer.training, backbone.training, head.training)
    tokenizer.eval()
    backbone.eval()
    head.eval()
    try:
        tokens = sample_tokens(
            tokenizer, backbone, head, n_samples, generator=generator
        )
    finally:
        tokenizer.train(was_training[0])
        backbone.train(was_training[1])
        head.train(was_training[2])

    id_lists = tokens.tolist()
    metrics: dict[str, float] = {"sample_entropy": unigram_entropy(id_lists)}
    texts = None
    if decode_fn is not None:
        texts = [decode_fn(ids) for ids in id_lists]
        if scorer is not None:
            metrics["gen_ppl"] = gen_ppl(texts, scorer)
    return tokens, texts, metrics


def run_diagnostics(
    tokenizer: MultiscaleTokenizer,
    backbone: GeneratorBackbone,
    head: LevelHead,
    cfg: Config,
    step: int,
    train_loss: float,
    per_level_loss: list[float],
    current_lr: float,
    tok_positions: int,
    gen_positions: int,
    data_source: str = "unknown",
    decode_fn: DecodeFn | None = None,
    scorer: Any = None,
    samples_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Metrics record for one evaluation step (SPEC 6.2 account fields included)."""
    device = tokenizer.tok_emb.weight.device
    sample_generator = torch.Generator(device=device).manual_seed(cfg.seed + step)
    _, texts, sample_metrics = sample_and_score(
        tokenizer,
        backbone,
        head,
        cfg.eval.n_samples,
        generator=sample_generator,
        decode_fn=decode_fn,
        scorer=scorer,
    )
    if texts is not None and samples_dir is not None:
        sample_path = Path(samples_dir) / f"step_{step:06d}.txt"
        sample_path.write_text(
            "\n".join(texts[:32]) + "\n", encoding="utf-8"
        )

    record: dict[str, Any] = {
        "step": step,
        "train_loss": float(train_loss),
        "per_level_loss": [float(value) for value in per_level_loss],
        "lr": float(current_lr),
        "data_source": data_source,
        "tok_positions": int(tok_positions),
        "gen_positions": int(gen_positions),
        "total_positions": int(tok_positions + gen_positions),
    }
    record.update(sample_metrics)
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
    decode_fn: DecodeFn | None = None,
    scorer: Any = None,
) -> Path:
    """Train the generator against a frozen tokenizer checkpoint (SPEC 5.5-5.7).

    Returns the run directory under ``results_dir``.
    """
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
        run_id = f"gen_{cfg.generator.head}_{ts}"

    run_dir = Path(results_dir) / run_id
    ckpt_dir = run_dir / "ckpt"
    samples_dir = run_dir / "samples"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    save_config(cfg, run_dir / "config.yaml")
    (run_dir / "git_sha.txt").write_text(get_git_sha(), encoding="utf-8")
    logger = JSONLLogger(run_dir / "metrics.jsonl")

    # Data (all tokenizer training inputs are reused unchanged).
    data_source = detect_data_source(data_dir)
    train_ds, _ = setup_data(
        cfg, data_dir=data_dir, allow_synthetic=allow_synthetic
    )
    train_loader = InfiniteBatchIterator(
        train_ds, batch_size=gen_train.batch_size, seed=cfg.seed, shuffle=True
    )

    # Frozen tokenizer and generator.
    tokenizer, tok_positions = load_tokenizer(tokenizer_ckpt, cfg)
    tokenizer = tokenizer.to(device)
    backbone, head = build_generator(cfg, tokenizer)
    backbone = backbone.to(device)
    head = head.to(device)
    params = list(backbone.parameters()) + list(head.parameters())
    optimizer = build_optimizer(
        list(backbone.named_parameters()) + list(head.named_parameters()),
        lr=gen_train.lr,
        weight_decay=gen_train.weight_decay,
        betas=gen_train.betas,
    )

    start_step = 0
    if resume_path:
        r_path = Path(resume_path)
        if not r_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {r_path}")
        checkpoint = torch.load(r_path, map_location=device)
        backbone.load_state_dict(checkpoint["backbone"])
        head.load_state_dict(checkpoint["head"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = checkpoint["step"] + 1
        tok_positions = int(checkpoint.get("tok_positions", tok_positions))
        print(f"Resumed from step {start_step}")

    backbone.train()
    head.train()
    rng = torch.Generator(device=device).manual_seed(cfg.seed)
    use_autocast = device.type == "cuda"
    amp_dtype = (
        torch.bfloat16
        if (use_autocast and torch.cuda.is_bf16_supported())
        else torch.float32
    )

    print(
        f"Starting generator run {run_id} (head={cfg.generator.head}, "
        f"quantizer={cfg.tokenizer.quantizer}, corrupt={cfg.generator.corrupt.mode}) "
        f"for {gen_train.max_steps} steps on {device}..."
    )

    recent_loss = 0.0
    recent_per_level = [0.0] * tokenizer.K
    for step in range(start_step, gen_train.max_steps):
        curr_lr = get_lr(
            step,
            warmup_steps=gen_train.warmup_steps,
            max_steps=gen_train.max_steps,
            lr=gen_train.lr,
            min_lr=gen_train.min_lr,
        )
        for pg in optimizer.param_groups:
            pg["lr"] = curr_lr

        x_batch = next(train_loader).to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=use_autocast
        ):
            loss, per_level = compute_generator_loss(
                tokenizer, backbone, head, x_batch, cfg, generator=rng
            )
        loss.backward()
        if gen_train.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, gen_train.grad_clip)
        optimizer.step()

        recent_loss = float(loss.detach())
        recent_per_level = [float(value.detach()) for value in per_level]
        gen_positions = (step + 1) * gen_train.batch_size * cfg.L

        is_last = step + 1 == gen_train.max_steps
        if (step + 1) % gen_train.eval_every == 0 or is_last or step == 0:
            metrics = run_diagnostics(
                tokenizer=tokenizer,
                backbone=backbone,
                head=head,
                cfg=cfg,
                step=step + 1,
                train_loss=recent_loss,
                per_level_loss=recent_per_level,
                current_lr=curr_lr,
                tok_positions=tok_positions,
                gen_positions=gen_positions,
                data_source=data_source,
                decode_fn=decode_fn,
                scorer=scorer,
                samples_dir=samples_dir,
            )
            logger.log(metrics)
            line = (
                f"Step {step + 1:5d} | Loss: {recent_loss:.4f} | "
                f"gen_pos: {gen_positions} | total_pos: {metrics['total_positions']}"
            )
            if "gen_ppl" in metrics:
                line += f" | Gen PPL: {metrics['gen_ppl']:.2f}"
            print(line)

        if (step + 1) % gen_train.save_every == 0 or is_last:
            state = {
                "step": step,
                "backbone": backbone.state_dict(),
                "head": head.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": asdict(cfg),
                "gen_positions": gen_positions,
                "tok_positions": tok_positions,
            }
            torch.save(state, ckpt_dir / f"step_{step + 1:06d}.pt")
            torch.save(state, ckpt_dir / "latest.pt")

    print(f"Training completed. Results written to {run_dir}")
    return run_dir


def build_decode_fn(tokenizer_json: str | Path) -> DecodeFn:
    """Decode dataset token ids with the dataset tokenizer (offline)."""
    from tokenizers import Tokenizer

    path = Path(tokenizer_json)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset tokenizer not found at {path} (needed to decode samples)"
        )
    tokenizer = Tokenizer.from_file(str(path))
    return lambda ids: tokenizer.decode(ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the multiscale generator")
    parser.add_argument("--config", type=str, default="configs/smoke.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Path to a tokenizer checkpoint (e.g. results/<tok_run>/ckpt/latest.pt)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/tinystories",
        help="Directory with train.bin/val.bin and optional tokenizer.json",
    )
    parser.add_argument(
        "--allow_synthetic",
        action="store_true",
        help="Explicitly allow random synthetic data when real memmaps are missing",
    )
    parser.add_argument(
        "--gen_ppl",
        action="store_true",
        help="Score samples with GPT-2-large (needs models/gpt2-large, offline)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)

    decode_fn: DecodeFn | None = None
    scorer: Any = None
    if args.gen_ppl:
        decode_fn = build_decode_fn(Path(args.data_dir) / "tokenizer.json")
        from jncm.eval.genppl import GPT2Scorer

        device = "cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"
        scorer = GPT2Scorer(model_path=cfg.eval.gpt2_model_path, device=device)

    train(
        cfg=cfg,
        tokenizer_ckpt=args.tokenizer,
        max_steps=args.steps,
        run_id=args.run_id,
        resume_path=args.resume,
        data_dir=args.data_dir,
        allow_synthetic=True if args.allow_synthetic else None,
        decode_fn=decode_fn,
        scorer=scorer,
    )


if __name__ == "__main__":
    main()

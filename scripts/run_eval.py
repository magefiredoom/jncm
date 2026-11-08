#!/usr/bin/env python
"""Evaluate generator checkpoints (SPEC 6.1/6.2, TASKS T4.2).

Loads a generator checkpoint and the tokenizer it was trained against, samples
``n_samples`` sequences for each seed, and reports Gen PPL (mean/std over
seeds), unigram entropy H on GPT-2 tokens and on dataset tokens, the SPEC 6.1
diversity metrics, and the position accounting recorded in the checkpoint.

A staged (Phase-1) checkpoint needs the frozen tokenizer passed with
``--tokenizer``. A ``train_jepa`` joint checkpoint carries its own tokenizer
under the ``tokenizer`` key, so no ``--tokenizer`` is needed for it, and its
records carry the SPEC 6.2 joint accounting (``joint_positions``,
``total_positions`` and ``total_positions_conservative``) instead of the staged
``tok_positions``/``gen_positions`` pair.

The trainer writes milestone checkpoints at 10/25/50/100% of the budget to
``results/<run_id>/ckpt/milestone_<pct>.pt`` when ``generator.train.save_milestones``
is on (default). By default this script evaluates exactly those; ``--ckpt``
selects checkpoints explicitly instead.

Output is ``results/<run_id>/eval/step_<n>.json`` per checkpoint, where ``<n>``
is the completed training step, zero-padded like the checkpoint file names. A
per-seed shard (``step_<n>.seed_<i>.json``) is written for every seed so a Slurm
seed array can run in parallel; ``--merge`` folds the shards of each step into
its aggregate file without rescoring.

Usage:
    python scripts/run_eval.py --run_id gen_vq_ce_dev \\
        --tokenizer results/tok_vq_dev/ckpt/latest.pt
    python scripts/run_eval.py --run_id gen_vq_ce_dev --merge
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
from typing import Any

import numpy as np
import torch

from jncm.config import Config, config_from_dict, load_config
from jncm.eval.genppl import GPT2Scorer, gen_ppl, unigram_entropy
from jncm.generator.backbone import GeneratorBackbone
from jncm.generator.heads import LevelHead
from jncm.generator.sample import sample_tokens
from jncm.tokenizer.trunk import MultiscaleTokenizer
from jncm.train_generator import (
    build_decode_fn,
    build_generator,
    diversity_metrics,
    load_tokenizer,
)
from jncm.utils import seed_everything

SHARD_RE = re.compile(r"^step_\d{6}\.seed_\d+\.json$")
METRIC_KEYS = ("ppl", "h_gpt2", "h_dataset", "unique_frac", "pairwise_agree")
META_KEYS = (
    "run_id",
    "checkpoint",
    "step",
    "fraction",
    "tok_positions",
    "gen_positions",
    "joint_positions",
    "total_positions",
    "total_positions_conservative",
    "n_samples",
    "n_seeds",
)


def _gpt2_ids(gpt2_tokenizer: Any, text: str) -> list[int]:
    """Token ids under the GPT-2 tokenizer (HF returns ids directly)."""
    ids = gpt2_tokenizer.encode(text)
    if hasattr(ids, "ids"):
        ids = ids.ids
    return [int(token) for token in ids]


def is_joint_checkpoint(state: dict[str, Any]) -> bool:
    """True for a ``train_jepa`` checkpoint (tokenizer and generator together)."""
    return "tokenizer" in state


def tokenizer_from_joint_checkpoint(
    state: dict[str, Any], cfg: Config
) -> MultiscaleTokenizer:
    """Rebuild the online tokenizer stored inside a joint checkpoint."""
    tokenizer = MultiscaleTokenizer(cfg.tokenizer, L=cfg.L, V=cfg.V)
    tokenizer.load_state_dict(state["tokenizer"])
    tokenizer.eval()
    for parameter in tokenizer.parameters():
        parameter.requires_grad_(False)
    return tokenizer


def resolve_checkpoint_config(
    cfg: Config, state: dict[str, Any], checkpoint: Path
) -> Config:
    """Return the checkpoint's saved config, validated against the CLI config.

    The checkpoint carries the architecture it was trained with, so eval must
    rebuild from it (a smoke run's H or layer count need not appear in the CLI
    YAML). The CLI config must still describe the same task: L, V, head,
    quantizer and K are checked and a mismatch fails loudly.
    """
    saved = state.get("config") or {}
    if not saved:
        return cfg
    if int(saved.get("L", cfg.L)) != cfg.L or int(saved.get("V", cfg.V)) != cfg.V:
        raise ValueError(
            f"checkpoint {checkpoint} L={saved.get('L')}, V={saved.get('V')} "
            f"does not match config L={cfg.L}, V={cfg.V}"
        )
    if is_joint_checkpoint(state):
        # A joint checkpoint carries its encoder and generator together, so its
        # own config is authoritative: there is no separate tokenizer
        # checkpoint whose head, quantizer or K could disagree with it.
        return config_from_dict(saved)
    saved_generator = saved.get("generator") or {}
    if saved_generator.get("head", cfg.generator.head) != cfg.generator.head:
        raise ValueError(
            f"checkpoint {checkpoint} head={saved_generator.get('head')!r} does "
            f"not match config head {cfg.generator.head!r}"
        )
    saved_tokenizer = saved.get("tokenizer") or {}
    if saved_tokenizer.get("quantizer", cfg.tokenizer.quantizer) != (
        cfg.tokenizer.quantizer
    ):
        raise ValueError(
            f"checkpoint {checkpoint} quantizer="
            f"{saved_tokenizer.get('quantizer')!r} does not match config "
            f"quantizer {cfg.tokenizer.quantizer!r}"
        )
    if int(saved_tokenizer.get("K", cfg.tokenizer.K)) != cfg.tokenizer.K:
        raise ValueError(
            f"checkpoint {checkpoint} K={saved_tokenizer.get('K')} does not "
            f"match config K={cfg.tokenizer.K}"
        )
    return config_from_dict(saved)


def _checkpoint_meta(
    cfg: Config, state: dict[str, Any], checkpoint: Path, run_id: str
) -> dict[str, Any]:
    """Validate the checkpoint against ``cfg`` and build the shared metadata."""
    resolve_checkpoint_config(cfg, state, checkpoint)
    step = int(state["step"]) + 1
    meta: dict[str, Any] = {
        "run_id": run_id,
        "checkpoint": str(checkpoint),
        "step": step,
        "fraction": state.get("milestone"),
    }
    if state.get("total_positions_conservative") is not None:
        joint_positions = int(
            state.get("joint_positions", state.get("total_positions", 0))
        )
        meta["joint_positions"] = joint_positions
        meta["total_positions"] = int(state.get("total_positions", joint_positions))
        meta["total_positions_conservative"] = int(
            state["total_positions_conservative"]
        )
    else:
        tok_positions = int(state.get("tok_positions", 0))
        gen_positions = int(
            state.get(
                "gen_positions", step * cfg.generator.train.batch_size * cfg.L
            )
        )
        meta["tok_positions"] = tok_positions
        meta["gen_positions"] = gen_positions
        meta["total_positions"] = tok_positions + gen_positions
    meta["n_samples"] = 0
    meta["n_seeds"] = 0
    return meta


def aggregate_records(
    records: list[dict[str, Any]], *, meta: dict[str, Any]
) -> dict[str, Any]:
    """Mean/std over one record per seed, keeping the per-seed values as runs."""
    seeds = sorted(int(record["seed"]) for record in records)
    expected = list(range(int(meta["n_seeds"])))
    if seeds != expected:
        raise ValueError(f"expected one record per seed {expected}, got seeds {seeds}")
    result = dict(meta)
    for key in METRIC_KEYS:
        values = [float(record[key]) for record in records]
        result[f"{key}_mean"] = float(np.mean(values))
        result[f"{key}_std"] = float(np.std(values))
        result[f"{key}_runs"] = values
    result["seeds"] = seeds
    return result


def evaluate_seed(
    cfg: Config,
    tokenizer: MultiscaleTokenizer,
    backbone: GeneratorBackbone,
    head: LevelHead,
    *,
    n_samples: int,
    seed: int,
    decode_fn: Any,
    scorer: Any,
    gpt2_tokenizer: Any,
    device: torch.device,
) -> dict[str, float]:
    """Sample ``n_samples`` sequences for one seed and compute SPEC 6.1 metrics."""
    generator = torch.Generator(device=device).manual_seed(int(cfg.seed) + int(seed))
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
    texts = [decode_fn(ids) for ids in id_lists]
    metrics: dict[str, float] = {
        "ppl": gen_ppl(texts, scorer),
        "h_dataset": unigram_entropy(id_lists),
        "h_gpt2": unigram_entropy(
            [_gpt2_ids(gpt2_tokenizer, text) for text in texts]
        ),
    }
    metrics.update(diversity_metrics(id_lists))
    return metrics


def evaluate_checkpoint(
    *,
    cfg: Config,
    tokenizer: MultiscaleTokenizer,
    backbone: GeneratorBackbone,
    head: LevelHead,
    state: dict[str, Any],
    checkpoint: str | Path,
    run_id: str,
    n_samples: int,
    n_seeds: int,
    seeds: list[int],
    decode_fn: Any,
    scorer: Any,
    gpt2_tokenizer: Any,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Score one checkpoint for every requested seed.

    Returns:
        ``(seed_records, aggregate)`` where the aggregate is built only when
        ``seeds`` covers ``0..n_seeds-1`` (a full local run); a partial seed
        set (a Slurm array task) writes shards only.
    """
    if not seeds:
        raise ValueError("seeds must not be empty")
    for seed in seeds:
        if not 0 <= int(seed) < int(n_seeds):
            raise ValueError(f"seed {seed} is outside 0..{int(n_seeds) - 1}")

    meta = _checkpoint_meta(cfg, state, Path(checkpoint), run_id)
    meta["n_samples"] = int(n_samples)
    meta["n_seeds"] = int(n_seeds)

    seed_records: list[dict[str, Any]] = []
    for seed in seeds:
        record = dict(meta)
        record["seed"] = int(seed)
        record.update(
            evaluate_seed(
                cfg,
                tokenizer,
                backbone,
                head,
                n_samples=n_samples,
                seed=int(seed),
                decode_fn=decode_fn,
                scorer=scorer,
                gpt2_tokenizer=gpt2_tokenizer,
                device=device,
            )
        )
        seed_records.append(record)

    aggregate = None
    if sorted(int(seed) for seed in seeds) == list(range(int(n_seeds))):
        aggregate = aggregate_records(seed_records, meta=meta)
    return seed_records, aggregate


def discover_milestones(run_dir: str | Path) -> list[Path]:
    """Milestone checkpoints of a run, in training order (file names sort)."""
    ckpt_dir = Path(run_dir) / "ckpt"
    found = sorted(ckpt_dir.glob("milestone_*.pt"))
    if not found:
        raise FileNotFoundError(
            f"no milestone checkpoints under {ckpt_dir}; train with "
            "generator.train.save_milestones=true (the default) or pass --ckpt"
        )
    return found


def _write_json(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")


def merge_shards(eval_dir: str | Path) -> list[Path]:
    """Fold ``step_<n>.seed_<i>.json`` shards into ``step_<n>.json`` files."""
    directory = Path(eval_dir)
    groups: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(directory.glob("step_*.seed_*.json")):
        if not SHARD_RE.match(path.name):
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        key = f"step_{int(record['step']):06d}"
        groups.setdefault(key, []).append(record)
    if not groups:
        raise ValueError(f"no per-seed eval shards found under {directory}")

    written: list[Path] = []
    for key in sorted(groups):
        records = groups[key]
        meta = {
            name: value
            for name in META_KEYS
            if (value := records[0].get(name)) is not None
        }
        path = directory / f"{key}.json"
        _write_json(path, aggregate_records(records, meta=meta))
        written.append(path)
    return written


def run_eval(
    cfg: Config,
    *,
    run_id: str,
    tokenizer_ckpt: str | Path | None = None,
    results_dir: str | Path = "results",
    data_dir: str | Path = "data/tinystories",
    n_samples: int = 256,
    n_seeds: int = 5,
    seeds: list[int] | None = None,
    ckpt_paths: list[str | Path] | None = None,
    decode_fn: Any = None,
    scorer: Any = None,
    gpt2_tokenizer: Any = None,
    device: str | torch.device = "cpu",
) -> list[Path]:
    """Evaluate milestone checkpoints and write the eval JSON files."""
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    seed_everything(cfg.seed)
    run_dir = Path(results_dir) / run_id
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    device_obj = torch.device(device)

    seed_list = (
        list(range(n_seeds))
        if seeds is None
        else sorted(int(seed) for seed in seeds)
    )
    ckpts = (
        [Path(path) for path in ckpt_paths]
        if ckpt_paths is not None
        else discover_milestones(run_dir)
    )
    if decode_fn is None:
        decode_fn = build_decode_fn(Path(data_dir) / "tokenizer.json")

    shared_tokenizer: MultiscaleTokenizer | None = None

    written: list[Path] = []
    for checkpoint in ckpts:
        state = torch.load(checkpoint, map_location="cpu")
        model_cfg = resolve_checkpoint_config(cfg, state, Path(checkpoint))
        if is_joint_checkpoint(state):
            tokenizer = tokenizer_from_joint_checkpoint(state, model_cfg)
        else:
            if tokenizer_ckpt is None:
                raise ValueError(
                    f"checkpoint {checkpoint} is a staged generator checkpoint; "
                    "pass --tokenizer (a joint JEPA checkpoint carries its own)"
                )
            if shared_tokenizer is None:
                shared_tokenizer, _ = load_tokenizer(tokenizer_ckpt, cfg)
                shared_tokenizer = shared_tokenizer.to(device_obj)
                shared_tokenizer.eval()
            tokenizer = shared_tokenizer
        tokenizer = tokenizer.to(device_obj).eval()
        backbone, head = build_generator(model_cfg, tokenizer)
        backbone.load_state_dict(state["backbone"])
        head.load_state_dict(state["head"])
        backbone.to(device_obj).eval()
        head.to(device_obj).eval()

        seed_records, aggregate = evaluate_checkpoint(
            cfg=model_cfg,
            tokenizer=tokenizer,
            backbone=backbone,
            head=head,
            state=state,
            checkpoint=checkpoint,
            run_id=run_id,
            n_samples=n_samples,
            n_seeds=n_seeds,
            seeds=seed_list,
            decode_fn=decode_fn,
            scorer=scorer,
            gpt2_tokenizer=gpt2_tokenizer,
            device=device_obj,
        )
        for record in seed_records:
            path = eval_dir / f"step_{record['step']:06d}.seed_{record['seed']}.json"
            _write_json(path, record)
            written.append(path)
            print(
                f"step {record['step']} seed {record['seed']}: "
                f"Gen PPL {record['ppl']:.2f}, H {record['h_dataset']:.2f}"
            )
        if aggregate is not None:
            path = eval_dir / f"step_{aggregate['step']:06d}.json"
            _write_json(path, aggregate)
            written.append(path)
            line = (
                f"step {aggregate['step']} ({aggregate['fraction']}): "
                f"Gen PPL {aggregate['ppl_mean']:.2f} +/- "
                f"{aggregate['ppl_std']:.2f}, total_positions "
                f"{aggregate['total_positions']}"
            )
            if aggregate.get("total_positions_conservative") is not None:
                line += (
                    f" (conservative {aggregate['total_positions_conservative']})"
                )
            print(line)
    return written


def main(argv: list[str] | None = None) -> None:
    default_gpt2 = os.environ.get("GPT2_LARGE_PATH", "models/gpt2-large")
    parser = argparse.ArgumentParser(
        description="Evaluate generator checkpoints (SPEC 6.1)"
    )
    parser.add_argument("--run_id", required=True, help="results/<run_id> of the generator run")
    parser.add_argument("--config", type=str, default="configs/tinystories_dev.yaml")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help=(
            "Frozen tokenizer checkpoint for a staged generator run; a joint "
            "train_jepa checkpoint carries its own and needs none"
        ),
    )
    parser.add_argument("--data_dir", type=str, default="data/tinystories")
    parser.add_argument(
        "--ckpt",
        dest="ckpts",
        action="append",
        default=None,
        help="Checkpoint path (repeatable); default: ckpt/milestone_*.pt",
    )
    parser.add_argument("--n_samples", type=int, default=256)
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seed indices, e.g. 2 for one array task (default: all)",
    )
    parser.add_argument("--gpt2", type=str, default=default_gpt2)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--results", type=str, default="results")
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Fold per-seed shards into step_<n>.json without rescoring",
    )
    args = parser.parse_args(argv)

    run_dir = Path(args.results) / args.run_id
    if args.merge:
        for path in merge_shards(run_dir / "eval"):
            print(f"wrote {path}")
        return

    cfg = load_config(args.config)
    seeds = None
    if args.seeds:
        seeds = [int(part) for part in args.seeds.split(",") if part.strip()]
    scorer = GPT2Scorer(model_path=args.gpt2, device=args.device)
    paths = run_eval(
        cfg,
        run_id=args.run_id,
        tokenizer_ckpt=args.tokenizer,
        results_dir=args.results,
        data_dir=args.data_dir,
        n_samples=args.n_samples,
        n_seeds=args.n_seeds,
        seeds=seeds,
        ckpt_paths=args.ckpts,
        scorer=scorer,
        gpt2_tokenizer=scorer.tokenizer,
        device=args.device,
    )
    print(f"wrote {len(paths)} eval records under {run_dir / 'eval'}")


if __name__ == "__main__":
    main()

"""Tests for scripts/run_eval.py and generator milestone checkpoints (T4.2).

CPU only: a smoke-scale tokenizer and generator checkpoint, a fake scorer and a
fake GPT-2 tokenizer stand in for the real models.
"""

from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path

import pytest
import torch

from jncm.config import Config, load_config
from jncm.tokenizer.trunk import MultiscaleTokenizer
from jncm.train_generator import (
    build_generator,
    load_tokenizer,
    milestone_steps,
    train,
)
from jncm.train_jepa import train as train_jepa
from jncm.utils import seed_everything
from scripts.run_eval import (
    discover_milestones,
    evaluate_checkpoint,
    merge_shards,
    run_eval,
)

TOK_POSITIONS = 1024


def _smoke_cfg() -> Config:
    cfg = load_config(Path("configs/smoke.yaml"))
    cfg.generator.H = 32
    cfg.generator.n_layers = 1
    cfg.generator.n_heads = 2
    cfg.generator.dropout = 0.0
    cfg.generator.train.batch_size = 4
    cfg.eval.n_samples = 2
    cfg.device = "cpu"
    return cfg


def _save_tokenizer_ckpt(tmp_path: Path, cfg: Config) -> Path:
    seed_everything(0)
    tok = MultiscaleTokenizer(cfg.tokenizer, L=cfg.L, V=cfg.V)
    path = tmp_path / "tok.pt"
    torch.save(
        {
            "step": 0,
            "model_state_dict": tok.state_dict(),
            "config": asdict(cfg),
            "tok_positions": TOK_POSITIONS,
        },
        path,
    )
    return path


def _save_generator_ckpt(
    path: Path,
    cfg: Config,
    tok_ckpt: Path,
    step: int,
    fraction: float | None = None,
) -> Path:
    tokenizer, _ = load_tokenizer(tok_ckpt, cfg)
    backbone, head = build_generator(cfg, tokenizer)
    state = {
        "step": step,
        "backbone": backbone.state_dict(),
        "head": head.state_dict(),
        "config": asdict(cfg),
        "gen_positions": (step + 1) * cfg.generator.train.batch_size * cfg.L,
        "tok_positions": TOK_POSITIONS,
    }
    if fraction is not None:
        state["milestone"] = fraction
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    return path


def _decode_fn(ids: list[int]) -> str:
    return " ".join(str(i % 5) for i in ids)


class _FakeScorer:
    """2.0 nats per token, 4 scored tokens per text."""

    def score_texts(self, texts: list[str]) -> tuple[float, int]:
        tokens = 4 * len(texts)
        return 2.0 * tokens, tokens


class _FakeGPT2Tokenizer:
    def encode(self, text: str) -> list[int]:
        return [len(word) for word in text.split()] or [0]


def _evaluate(cfg: Config, tok_ckpt: Path, ckpt: Path, seeds: tuple[int, ...] = (0, 1)):
    tokenizer, _ = load_tokenizer(tok_ckpt, cfg)
    backbone, head = build_generator(cfg, tokenizer)
    state = torch.load(ckpt, map_location="cpu")
    backbone.load_state_dict(state["backbone"])
    head.load_state_dict(state["head"])
    return evaluate_checkpoint(
        cfg=cfg,
        tokenizer=tokenizer,
        backbone=backbone,
        head=head,
        state=state,
        checkpoint=ckpt,
        run_id="run_a",
        n_samples=2,
        n_seeds=len(seeds),
        seeds=list(seeds),
        decode_fn=_decode_fn,
        scorer=_FakeScorer(),
        gpt2_tokenizer=_FakeGPT2Tokenizer(),
        device=torch.device("cpu"),
    )


def test_evaluate_checkpoint_reports_spec_61_metrics(tmp_path: Path) -> None:
    cfg = _smoke_cfg()
    tok_ckpt = _save_tokenizer_ckpt(tmp_path, cfg)
    ckpt = _save_generator_ckpt(
        tmp_path / "milestone_050.pt", cfg, tok_ckpt, step=9, fraction=0.5
    )

    seed_records, record = _evaluate(cfg, tok_ckpt, ckpt)

    assert [seed["seed"] for seed in seed_records] == [0, 1]
    assert record["step"] == 10
    assert record["fraction"] == 0.5
    assert record["tok_positions"] == TOK_POSITIONS
    assert record["gen_positions"] == 10 * 4 * cfg.L
    assert record["total_positions"] == TOK_POSITIONS + 10 * 4 * cfg.L
    assert record["n_samples"] == 2
    assert record["n_seeds"] == 2
    assert record["ppl_mean"] == pytest.approx(math.exp(2.0))
    assert record["ppl_std"] == pytest.approx(0.0)
    assert len(record["ppl_runs"]) == 2
    assert record["h_dataset_mean"] >= 0.0
    assert record["h_gpt2_mean"] >= 0.0
    assert 0.0 <= record["unique_frac_mean"] <= 1.0
    assert 0.0 <= record["pairwise_agree_mean"] <= 1.0


def test_evaluate_checkpoint_rejects_a_different_generator(tmp_path: Path) -> None:
    cfg = _smoke_cfg()
    tok_ckpt = _save_tokenizer_ckpt(tmp_path, cfg)
    ckpt = _save_generator_ckpt(tmp_path / "milestone_050.pt", cfg, tok_ckpt, step=9)
    state = torch.load(ckpt, map_location="cpu")
    state["config"]["generator"]["head"] = "flow"

    tokenizer, _ = load_tokenizer(tok_ckpt, cfg)
    backbone, head = build_generator(cfg, tokenizer)
    with pytest.raises(ValueError, match="head"):
        evaluate_checkpoint(
            cfg=cfg,
            tokenizer=tokenizer,
            backbone=backbone,
            head=head,
            state=state,
            checkpoint=ckpt,
            run_id="run_a",
            n_samples=2,
            n_seeds=1,
            seeds=[0],
            decode_fn=_decode_fn,
            scorer=_FakeScorer(),
            gpt2_tokenizer=_FakeGPT2Tokenizer(),
            device=torch.device("cpu"),
        )


def test_run_eval_uses_the_checkpoint_architecture(tmp_path: Path) -> None:
    """A checkpoint trained with overridden H rebuilds from its own config."""
    cfg = _smoke_cfg()
    tok_ckpt = _save_tokenizer_ckpt(tmp_path, cfg)
    results = tmp_path / "results"
    ckpt_dir = results / "run_a" / "ckpt"
    _save_generator_ckpt(ckpt_dir / "milestone_100.pt", cfg, tok_ckpt, step=9, fraction=1.0)

    other = _smoke_cfg()
    other.generator.H = 64
    other.generator.n_layers = 2
    written = run_eval(
        other,
        run_id="run_a",
        tokenizer_ckpt=tok_ckpt,
        results_dir=results,
        n_samples=2,
        n_seeds=2,
        decode_fn=_decode_fn,
        scorer=_FakeScorer(),
        gpt2_tokenizer=_FakeGPT2Tokenizer(),
        device=torch.device("cpu"),
    )

    aggregates = sorted(path.name for path in written if ".seed_" not in path.name)
    assert aggregates == ["step_000010.json"]


def test_run_eval_writes_step_json_and_seed_shards(tmp_path: Path) -> None:
    cfg = _smoke_cfg()
    tok_ckpt = _save_tokenizer_ckpt(tmp_path, cfg)
    results = tmp_path / "results"
    ckpt_dir = results / "run_a" / "ckpt"
    _save_generator_ckpt(ckpt_dir / "milestone_010.pt", cfg, tok_ckpt, step=1, fraction=0.1)
    _save_generator_ckpt(ckpt_dir / "milestone_100.pt", cfg, tok_ckpt, step=9, fraction=1.0)

    written = run_eval(
        cfg,
        run_id="run_a",
        tokenizer_ckpt=tok_ckpt,
        results_dir=results,
        n_samples=2,
        n_seeds=2,
        decode_fn=_decode_fn,
        scorer=_FakeScorer(),
        gpt2_tokenizer=_FakeGPT2Tokenizer(),
        device=torch.device("cpu"),
    )

    aggregates = sorted(path.name for path in written if ".seed_" not in path.name)
    assert aggregates == ["step_000002.json", "step_000010.json"]
    eval_dir = results / "run_a" / "eval"
    record = json.loads((eval_dir / "step_000002.json").read_text(encoding="utf-8"))
    assert record["run_id"] == "run_a"
    assert record["step"] == 2
    assert record["fraction"] == 0.1
    assert record["ppl_mean"] == pytest.approx(math.exp(2.0))
    for seed in (0, 1):
        assert (eval_dir / f"step_000002.seed_{seed}.json").exists()


def test_discover_milestones_is_sorted_and_fails_loudly(tmp_path: Path) -> None:
    cfg = _smoke_cfg()
    tok_ckpt = _save_tokenizer_ckpt(tmp_path, cfg)
    ckpt_dir = tmp_path / "run_a" / "ckpt"
    _save_generator_ckpt(ckpt_dir / "milestone_100.pt", cfg, tok_ckpt, step=9, fraction=1.0)
    _save_generator_ckpt(ckpt_dir / "milestone_010.pt", cfg, tok_ckpt, step=1, fraction=0.1)

    found = discover_milestones(tmp_path / "run_a")

    assert [path.name for path in found] == ["milestone_010.pt", "milestone_100.pt"]
    with pytest.raises(FileNotFoundError):
        discover_milestones(tmp_path / "no_such_run")


def test_merge_shards_combines_seeds_and_requires_all_of_them(tmp_path: Path) -> None:
    eval_dir = tmp_path / "run_a" / "eval"
    eval_dir.mkdir(parents=True)

    def shard(step: int, seed: int, ppl: float) -> None:
        record = {
            "run_id": "run_a",
            "checkpoint": "milestone_010.pt",
            "step": step,
            "fraction": 0.1,
            "tok_positions": 100,
            "gen_positions": 900,
            "total_positions": 1000,
            "n_samples": 2,
            "n_seeds": 2,
            "seed": seed,
            "ppl": ppl,
            "h_gpt2": 4.0,
            "h_dataset": 4.1,
            "unique_frac": 0.5,
            "pairwise_agree": 0.1,
        }
        (eval_dir / f"step_{step:06d}.seed_{seed}.json").write_text(
            json.dumps(record), encoding="utf-8"
        )

    shard(2, 0, 2.0)
    shard(2, 1, 4.0)
    written = merge_shards(eval_dir)

    assert written == [eval_dir / "step_000002.json"]
    record = json.loads(written[0].read_text(encoding="utf-8"))
    assert record["ppl_mean"] == pytest.approx(3.0)
    assert record["ppl_std"] == pytest.approx(1.0)
    assert record["ppl_runs"] == [2.0, 4.0]
    assert record["total_positions"] == 1000
    assert record["n_seeds"] == 2

    (eval_dir / "step_000002.seed_1.json").unlink()
    with pytest.raises(ValueError, match="seed"):
        merge_shards(eval_dir)


def test_milestone_steps_rounds_clamps_and_validates() -> None:
    assert milestone_steps(1000, [0.1, 0.25, 0.5, 1.0]) == {
        100: 0.1,
        250: 0.25,
        500: 0.5,
        1000: 1.0,
    }
    assert milestone_steps(3, [0.1, 0.5, 1.0]) == {1: 0.1, 2: 0.5, 3: 1.0}
    for bad in ([], [0.0], [1.5]):
        with pytest.raises(ValueError):
            milestone_steps(10, bad)


def test_trainer_saves_milestone_checkpoints_behind_the_flag(tmp_path: Path) -> None:
    cfg = _smoke_cfg()
    cfg.generator.train.milestones = [0.1, 0.5, 1.0]
    cfg.generator.train.eval_every = 10
    tok_ckpt = _save_tokenizer_ckpt(tmp_path, cfg)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    run_dir = train(
        cfg,
        tok_ckpt,
        max_steps=20,
        run_id="milestones",
        results_dir=tmp_path / "results",
        data_dir=data_dir,
        allow_synthetic=True,
    )

    ckpt_dir = run_dir / "ckpt"
    names = sorted(path.name for path in ckpt_dir.glob("milestone_*.pt"))
    assert names == ["milestone_010.pt", "milestone_050.pt", "milestone_100.pt"]
    state = torch.load(ckpt_dir / "milestone_050.pt", map_location="cpu")
    assert state["milestone"] == pytest.approx(0.5)
    assert state["step"] + 1 == 10
    assert state["gen_positions"] == 10 * cfg.generator.train.batch_size * cfg.L

    cfg.generator.train.save_milestones = False
    run_dir_off = train(
        cfg,
        tok_ckpt,
        max_steps=4,
        run_id="no_milestones",
        results_dir=tmp_path / "results",
        data_dir=data_dir,
        allow_synthetic=True,
    )
    assert not list((run_dir_off / "ckpt").glob("milestone_*.pt"))


def _jepa_cfg() -> Config:
    """configs/jepa_smoke.yaml shrunk to a fast CPU JEPA run."""
    cfg = load_config(Path("configs/jepa_smoke.yaml"))
    cfg.tokenizer.d = 32
    cfg.tokenizer.decoder_heads = 4
    cfg.generator.H = 32
    cfg.generator.n_layers = 1
    cfg.generator.n_heads = 2
    cfg.generator.dropout = 0.0
    cfg.generator.n_rep = 1
    cfg.generator.flow_steps = 2
    cfg.generator.train.batch_size = 4
    cfg.eval.n_samples = 2
    cfg.device = "cpu"
    return cfg


def test_run_eval_reads_a_joint_jepa_checkpoint(tmp_path: Path) -> None:
    """T5.4: tokenizer and generator both come from the joint checkpoint."""
    cfg = _jepa_cfg()
    tok_ckpt = _save_tokenizer_ckpt(tmp_path, cfg)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    results = tmp_path / "results"

    run_dir = train_jepa(
        cfg,
        tok_ckpt,
        max_steps=20,
        run_id="jepa_eval",
        results_dir=results,
        data_dir=data_dir,
        allow_synthetic=True,
    )
    milestones = sorted(
        path.name for path in (run_dir / "ckpt").glob("milestone_*.pt")
    )
    assert milestones == [
        "milestone_010.pt",
        "milestone_025.pt",
        "milestone_050.pt",
        "milestone_100.pt",
    ]

    written = run_eval(
        cfg,
        run_id="jepa_eval",
        tokenizer_ckpt=None,
        results_dir=results,
        n_samples=2,
        n_seeds=2,
        decode_fn=_decode_fn,
        scorer=_FakeScorer(),
        gpt2_tokenizer=_FakeGPT2Tokenizer(),
        device=torch.device("cpu"),
    )

    aggregates = sorted(path.name for path in written if ".seed_" not in path.name)
    assert aggregates == [
        "step_000002.json",
        "step_000005.json",
        "step_000010.json",
        "step_000020.json",
    ]
    record = json.loads(
        (run_dir / "eval" / "step_000020.json").read_text(encoding="utf-8")
    )
    for key in (
        "ppl_mean",
        "ppl_std",
        "h_gpt2_mean",
        "h_dataset_mean",
        "unique_frac_mean",
        "pairwise_agree_mean",
        "joint_positions",
        "total_positions",
        "total_positions_conservative",
    ):
        assert key in record, key
    joint_positions = 20 * cfg.generator.train.batch_size * cfg.L
    assert record["joint_positions"] == joint_positions
    assert record["total_positions"] == joint_positions
    assert record["total_positions_conservative"] == 2 * joint_positions
    assert "tok_positions" not in record and "gen_positions" not in record
    assert record["ppl_mean"] == pytest.approx(math.exp(2.0))

    merged = merge_shards(run_dir / "eval")
    merged_record = json.loads(merged[-1].read_text(encoding="utf-8"))
    assert merged_record["total_positions_conservative"] == 2 * joint_positions

"""T4.16: the SPEC 10.4 final eval (1024 x 5) gets its own folder, eval/final/.

Before T4.16 the final eval wrote the same eval/step_<n>.seed_<i>.json shards as the
256 x 5 final milestone and its merge rewrote eval/step_<n>.json, so the 256-sample
record was lost (docs/REVIEW_2026-11-26.md M1). CPU only: a smoke-scale tokenizer and
generator, a fake scorer and a fake GPT-2 tokenizer; the small sample counts 2 and 4
stand in for 256 and 1024.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch

from jncm.config import Config, load_config
from jncm.tokenizer.trunk import MultiscaleTokenizer
from jncm.train_generator import build_generator, load_tokenizer
from jncm.utils import seed_everything
from scripts import gate_report, paper_results, plot_curves, plot_fig_a
from scripts.run_eval import main as run_eval_main
from scripts.run_eval import merge_shards, run_eval

CURVE_N, FINAL_N = 2, 4  # stand-ins for 256 and 1024


def _smoke_cfg() -> Config:
    cfg = load_config(Path("configs/smoke.yaml"))
    cfg.generator.H = 32
    cfg.generator.n_layers = 1
    cfg.generator.n_heads = 2
    cfg.generator.dropout = 0.0
    cfg.generator.train.batch_size = 4
    cfg.device = "cpu"
    return cfg


def _decode_fn(ids: list[int]) -> str:
    return " ".join(str(i % 5) for i in ids)


class _FakeScorer:
    def score_texts(self, texts: list[str]) -> tuple[float, int]:
        tokens = 4 * len(texts)
        return 2.0 * tokens, tokens


class _FakeGPT2Tokenizer:
    def encode(self, text: str) -> list[int]:
        return [len(word) for word in text.split()] or [0]


def _run(tmp_path: Path) -> tuple[Config, Path, Path, Path]:
    """A run with one final milestone checkpoint; returns cfg, tok, ckpt, results."""
    cfg = _smoke_cfg()
    seed_everything(0)
    tok = MultiscaleTokenizer(cfg.tokenizer, L=cfg.L, V=cfg.V)
    tok_ckpt = tmp_path / "tok.pt"
    torch.save({"step": 0, "model_state_dict": tok.state_dict(), "config": asdict(cfg),
                "tok_positions": 1024}, tok_ckpt)
    tokenizer, _ = load_tokenizer(tok_ckpt, cfg)
    backbone, head = build_generator(cfg, tokenizer)
    results = tmp_path / "results"
    ckpt = results / "run_a" / "ckpt" / "milestone_100.pt"
    ckpt.parent.mkdir(parents=True)
    torch.save({"step": 9, "backbone": backbone.state_dict(), "head": head.state_dict(),
                "config": asdict(cfg), "gen_positions": 10 * 4 * cfg.L,
                "tok_positions": 1024, "milestone": 1.0}, ckpt)
    return cfg, tok_ckpt, ckpt, results


def _eval(cfg: Config, tok_ckpt: Path, ckpt: Path, results: Path, *, seed: int,
          n_samples: int, final: bool = False) -> list[Path]:
    """One array task: one seed, so only its shard is written."""
    return run_eval(cfg, run_id="run_a", tokenizer_ckpt=tok_ckpt, results_dir=results,
                    n_samples=n_samples, n_seeds=2, seeds=[seed], ckpt_paths=[ckpt],
                    decode_fn=_decode_fn, scorer=_FakeScorer(),
                    gpt2_tokenizer=_FakeGPT2Tokenizer(), device=torch.device("cpu"),
                    final=final)


def _record(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_both_final_records_survive_both_merges_in_either_order(tmp_path: Path) -> None:
    """TASKS T4.16 acceptance: the final eval starts before the 256-sample merge."""
    cfg, tok_ckpt, ckpt, results = _run(tmp_path)
    eval_dir = results / "run_a" / "eval"
    for seed in (0, 1):
        _eval(cfg, tok_ckpt, ckpt, results, seed=seed, n_samples=CURVE_N)
    for seed in (0, 1):
        _eval(cfg, tok_ckpt, ckpt, results, seed=seed, n_samples=FINAL_N, final=True)
    assert sorted(p.name for p in (eval_dir / "final").iterdir()) == [
        "step_000010.seed_0.json", "step_000010.seed_1.json"]

    assert merge_shards(eval_dir / "final") == [eval_dir / "final" / "step_000010.json"]
    assert merge_shards(eval_dir) == [eval_dir / "step_000010.json"]

    curve = _record(eval_dir / "step_000010.json")
    final = _record(eval_dir / "final" / "step_000010.json")
    assert (curve["n_samples"], curve["n_seeds"]) == (CURVE_N, 2)
    assert (final["n_samples"], final["n_seeds"]) == (FINAL_N, 2)
    for seed in (0, 1):
        assert _record(eval_dir / f"step_000010.seed_{seed}.json")["n_samples"] == CURVE_N


def test_each_merge_ignores_the_other_folder_and_its_incomplete_shards(tmp_path: Path) -> None:
    cfg, tok_ckpt, ckpt, results = _run(tmp_path)
    eval_dir = results / "run_a" / "eval"
    for seed in (0, 1):
        _eval(cfg, tok_ckpt, ckpt, results, seed=seed, n_samples=CURVE_N)
    _eval(cfg, tok_ckpt, ckpt, results, seed=0, n_samples=FINAL_N, final=True)
    # The final array is still running (seed 1 missing): the curve merge is unaffected.
    assert merge_shards(eval_dir) == [eval_dir / "step_000010.json"]
    with pytest.raises(ValueError, match="seed"):
        merge_shards(eval_dir / "final")
    _eval(cfg, tok_ckpt, ckpt, results, seed=1, n_samples=FINAL_N, final=True)
    (eval_dir / "step_000010.seed_1.json").unlink()  # now the curve array is incomplete
    assert merge_shards(eval_dir / "final") == [eval_dir / "final" / "step_000010.json"]


def test_cli_merge_final_touches_only_eval_final(tmp_path: Path, capsys) -> None:
    cfg, tok_ckpt, ckpt, results = _run(tmp_path)
    eval_dir = results / "run_a" / "eval"
    _eval(cfg, tok_ckpt, ckpt, results, seed=0, n_samples=CURVE_N)  # curve array incomplete
    for seed in (0, 1):
        _eval(cfg, tok_ckpt, ckpt, results, seed=seed, n_samples=FINAL_N, final=True)
    run_eval_main(["--run_id", "run_a", "--results", str(results), "--merge", "--final"])
    assert (eval_dir / "final" / "step_000010.json").is_file()
    assert not (eval_dir / "step_000010.json").exists()
    assert "eval/final/step_000010.json" in capsys.readouterr().out.replace("\\", "/")


def test_final_needs_an_explicit_checkpoint(tmp_path: Path) -> None:
    cfg, tok_ckpt, _, results = _run(tmp_path)
    with pytest.raises(ValueError, match="--ckpt"):
        run_eval(cfg, run_id="run_a", tokenizer_ckpt=tok_ckpt, results_dir=results,
                 n_samples=FINAL_N, n_seeds=2, seeds=[0], decode_fn=_decode_fn,
                 scorer=_FakeScorer(), gpt2_tokenizer=_FakeGPT2Tokenizer(),
                 device=torch.device("cpu"), final=True)


def test_an_eval_without_final_refuses_to_replace_another_sample_count(tmp_path: Path) -> None:
    """A forgotten FINAL=1 must not overwrite the 256-sample shard (the M1 data loss)."""
    cfg, tok_ckpt, ckpt, results = _run(tmp_path)
    shard = results / "run_a" / "eval" / "step_000010.seed_0.json"
    _eval(cfg, tok_ckpt, ckpt, results, seed=0, n_samples=CURVE_N)
    before = shard.read_bytes()
    with pytest.raises(ValueError, match="--final"):
        _eval(cfg, tok_ckpt, ckpt, results, seed=0, n_samples=FINAL_N)
    assert shard.read_bytes() == before
    _eval(cfg, tok_ckpt, ckpt, results, seed=0, n_samples=CURVE_N)  # same count: allowed


def test_merge_refuses_a_step_whose_shards_mix_sample_counts(tmp_path: Path) -> None:
    cfg, tok_ckpt, ckpt, results = _run(tmp_path)
    eval_dir = results / "run_a" / "eval"
    for seed in (0, 1):
        _eval(cfg, tok_ckpt, ckpt, results, seed=seed, n_samples=CURVE_N)
    shard = eval_dir / "step_000010.seed_1.json"
    record = _record(shard)
    record["n_samples"] = FINAL_N  # as a pre-T4.16 final eval would have left it
    shard.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="n_samples"):
        merge_shards(eval_dir)
    assert not (eval_dir / "step_000010.json").exists()


def _write_aggregate(path: Path, n_samples: int, ppl: float, h: float, unique: float,
                     agree: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "run_id": "gen_vq_ce_dev", "step": 10000, "fraction": 1.0,
        "total_positions": 655_360_000 + 10000 * 1024 * 128,
        "n_samples": n_samples, "n_seeds": 5, "ppl_mean": ppl, "ppl_std": 0.5,
        "h_gpt2_mean": h, "unique_frac_mean": unique, "pairwise_agree_mean": agree,
        "n_params_backbone": 85_459_200, "n_params_head": 37_797_888,
    }), encoding="utf-8")


def _paper_fixture(root: Path, *, with_final: bool = True) -> None:
    run = root / "gen_vq_ce_dev"
    for i in range(1, 10):
        rec_path = run / "eval" / f"step_{1000 * i:06d}.json"
        _write_aggregate(rec_path, 256, 40.0 / i, 4.0, 1.0, 0.02)
        record = _record(rec_path)
        record.update(step=1000 * i, fraction=i / 10,
                      total_positions=655_360_000 + 1000 * i * 1024 * 128)
        rec_path.write_text(json.dumps(record), encoding="utf-8")
    _write_aggregate(run / "eval" / "step_010000.json", 256, 4.10, 3.90, 0.990, 0.031)
    if with_final:
        _write_aggregate(run / "eval" / "final" / "step_010000.json", 1024, 4.00, 3.95,
                         0.997, 0.024)
    (run / "DONE").write_text("{}", encoding="utf-8")


def test_table_1_reads_the_final_record_and_figure_a_the_256_one(tmp_path: Path) -> None:
    _paper_fixture(tmp_path)
    arms = [("vq + categorical", "gen_vq_ce_dev")]
    rows, placeholder = paper_results.table_rows(tmp_path, arms)
    assert rows[0]["reasons"] == [] and placeholder is False
    line = next(ln for ln in paper_results.render_table(rows, placeholder).splitlines()
                if ln.startswith("| vq + categorical"))
    # every metric column from the 1024 x 5 record: PPL 4.00, H 3.95, unique 0.997, agree 0.024
    assert "| 4.00 ± 0.50 |" in line and "| 3.95 |" in line
    assert "| 0.997 | 0.024 |" in line
    series = plot_fig_a.arm_series(tmp_path, arms)
    assert series[0]["reasons"] == []
    last = series[0]["curves"]["total_positions"][-1]
    assert last["ppl"] == pytest.approx(4.10)
    assert series[0]["large_steps"] == set()


def test_table_1_without_a_final_eval_says_where_it_looks(tmp_path: Path) -> None:
    _paper_fixture(tmp_path, with_final=False)
    rows, placeholder = paper_results.table_rows(tmp_path, [("vq + categorical", "gen_vq_ce_dev")])
    assert placeholder is True
    assert any("eval/final/" in reason for reason in rows[0]["reasons"])
    line = next(ln for ln in paper_results.render_table(rows, placeholder).splitlines()
                if ln.startswith("| vq + categorical"))
    assert "| 4.10 ± 0.50 |" in line  # the last 256 x 5 record, daggered


def test_curve_loaders_never_read_eval_final(tmp_path: Path) -> None:
    _paper_fixture(tmp_path)
    records = gate_report.load_run(tmp_path, "gen_vq_ce_dev")
    assert {r["n_samples"] for r in records} == {256} and len(records) == 10
    points = plot_curves.collect_points(tmp_path)["gen_vq_ce_dev"]
    assert {p["n_samples"] for p in points} == {256} and len(points) == 10

"""TASKS E7: Table 1, Figure A and Figure B scripts on fixture records only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import paper_results, plot_fig_a, plot_fig_b

TOK = 655_360_000


def _eval(root: Path, run_id: str, steps: list[int], *, n_samples: int = 256,
          ppl0: float = 40.0, joint: bool = False, done: bool = False,
          last_fraction: float | None = None) -> None:
    eval_dir = root / run_id / "eval"
    eval_dir.mkdir(parents=True)
    for index, step in enumerate(steps, start=1):
        positions = step * 1024 * 128
        record = {
            "run_id": run_id, "step": step,
            "fraction": index / 10 if last_fraction is None or index < len(steps)
            else last_fraction,
            "total_positions": TOK + positions,
            "n_samples": n_samples, "n_seeds": 5,
            "ppl_mean": ppl0 / index, "ppl_std": 0.5,
            "h_gpt2_mean": 4.0, "h_dataset_mean": 3.9,
            "unique_frac_mean": 1.0, "pairwise_agree_mean": 0.02,
            "n_params_total": 123_300_000,
        }
        if joint:
            record["total_positions_conservative"] = TOK + 2 * positions
        (eval_dir / f"step_{step:06d}.json").write_text(json.dumps(record),
                                                       encoding="utf-8")
    if done:
        (root / run_id / "DONE").write_text("{}", encoding="utf-8")


def _metrics(root: Path, run_id: str, records: list[dict], done: bool = True) -> None:
    run = root / run_id
    run.mkdir(parents=True, exist_ok=True)
    (run / "metrics.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    if done:
        (run / "DONE").write_text("{}", encoding="utf-8")


def test_parse_arms_overrides_by_label_and_rejects_bad_specs() -> None:
    arms = paper_results.parse_arms(["vq + categorical=pilot_gen_vq_ce", "extra=x"])
    assert arms[0] == ("vq + categorical", "pilot_gen_vq_ce")
    assert arms[-1] == ("extra", "x")
    assert len(arms) == len(paper_results.DEFAULT_ARMS) + 1
    with pytest.raises(ValueError):
        paper_results.parse_arms(["no_equals"])


def test_table_final_row_has_no_dagger(tmp_path: Path) -> None:
    _eval(tmp_path, "gen_vq_ce_dev", [1000, 2000], n_samples=1024, done=True,
          last_fraction=1.0)
    rows, placeholder = paper_results.table_rows(
        tmp_path, [("vq + categorical", "gen_vq_ce_dev")])
    assert rows[0]["reasons"] == [] and placeholder is False
    text = paper_results.render_table(rows, placeholder)
    assert "PLACEHOLDER" not in text and paper_results.DAGGER not in text
    # last record: ppl 40 / 2 = 20.00; ratio to 10.80
    assert "| 20.00 ± 0.50 | 1.852 |" in text
    assert "123.3M" in text


def test_table_marks_pilot_partial_small_eval_and_missing(tmp_path: Path) -> None:
    _eval(tmp_path, "pilot_gen_vq_ce", [30, 60, 90])
    out = tmp_path / "t1.md"
    assert paper_results.main([
        "--results", str(tmp_path), "--out", str(out),
        "--arm", "vq + categorical=pilot_gen_vq_ce"]) == 0
    text = out.read_text(encoding="utf-8")
    assert text.startswith("**PLACEHOLDER")
    line = next(l for l in text.splitlines() if l.startswith("- † vq"))
    for reason in ("pilot or sweep run", "no DONE", "last milestone 30%",
                   "eval 256 x 5 (needs 1024 x 5)"):
        assert reason in line
    assert "`gen_noisy_flow_dev`): no merged eval records" in text
    assert "| 13.33 ± 0.50 |" in text  # 40 / 3, the last point by position


def test_fig_a_draws_both_accountings_and_flags_placeholder(tmp_path: Path, capsys) -> None:
    _eval(tmp_path, "gen_vq_ce_dev", [1000, 2000], done=True, last_fraction=1.0)
    _eval(tmp_path, "jepa_base_dev", [1000], joint=True)
    series = plot_fig_a.arm_series(tmp_path, paper_results.DEFAULT_ARMS)
    jepa = next(s for s in series if s["run_id"] == "jepa_base_dev")
    once, cons = (jepa["curves"][a][0]["x"] for a in plot_fig_a.ACCOUNTINGS)
    assert cons - once == 1000 * 1024 * 128
    vq = next(s for s in series if s["run_id"] == "gen_vq_ce_dev")
    assert vq["reasons"] == []
    out = tmp_path / "a.png"
    assert plot_fig_a.main(["--results", str(tmp_path), "--out", str(out)]) == 0
    assert out.stat().st_size > 10_000
    printed = capsys.readouterr().out
    assert "gen_vq_ce_dev): final" in printed
    assert "gen_noisy_flow_dev): MISSING" in printed
    assert "(PLACEHOLDER)" in printed


def test_fig_a_without_any_run_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        plot_fig_a.main(["--results", str(tmp_path), "--out", str(tmp_path / "a.png")])


def test_fig_b_reads_last_acc_cum_and_r2_and_lists_unlogged(tmp_path: Path, capsys) -> None:
    _metrics(tmp_path, "tok_vq_10k", [{"step": 1, "acc_cum": [0.0, 0.0]},
                                      {"step": 2, "acc_cum": [0.1, 0.99]}])
    _metrics(tmp_path, "tok_noisy_10k", [{"step": 2, "acc_cum": [0.2, 0.98]}])
    _metrics(tmp_path, "pilot_jepa_base",
             [{"step": 5, "r2": [0.1, 0.5]}, {"step": 9, "loss": 1.0}], done=False)
    _metrics(tmp_path, "gen_noisy_flow_dev", [{"step": 5, "per_level_loss": [1, 2]}])
    out = tmp_path / "b.png"
    assert plot_fig_b.main(["--results", str(tmp_path), "--out", str(out),
                            "--r2", "JEPA (base)=pilot_jepa_base"]) == 0
    assert out.stat().st_size > 10_000
    printed = capsys.readouterr().out
    assert "| vq | `tok_vq_10k` | 2 | 0.100 | 0.990 |" in printed
    assert "| JEPA (base) † | `pilot_jepa_base` | 5 | 0.100 | 0.500 |" in printed
    assert "`gen_noisy_flow_dev` | - | not logged" in printed
    assert "(PLACEHOLDER)" in printed


def test_fig_b_malformed_metrics_raise(tmp_path: Path) -> None:
    (tmp_path / "tok_vq_10k").mkdir()
    (tmp_path / "tok_vq_10k" / "metrics.jsonl").write_text("{bad\n", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        plot_fig_b.main(["--results", str(tmp_path), "--out", str(tmp_path / "b.png")])

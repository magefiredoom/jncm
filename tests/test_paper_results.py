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
            "n_params_tokenizer": 11_930_000, "n_params_backbone": 85_500_000,
            "n_params_head": 37_800_000, "n_params_total": 135_230_000,
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


def _final(root: Path, run_id: str, n_samples: int = 1024) -> None:
    """T4.16 layout: the SPEC 10.4 final eval in eval/final/, a copy of the last
    milestone record re-evaluated at ``n_samples`` x 5."""
    last = sorted((root / run_id / "eval").glob("step_*.json"))[-1]
    record = json.loads(last.read_text(encoding="utf-8"))
    record["n_samples"] = n_samples
    final = root / run_id / "eval" / "final" / last.name
    final.parent.mkdir(parents=True, exist_ok=True)
    final.write_text(json.dumps(record), encoding="utf-8")


def test_table_final_row_has_no_dagger(tmp_path: Path) -> None:
    _eval(tmp_path, "gen_vq_ce_dev", [1000, 2000], done=True, last_fraction=1.0)
    _final(tmp_path, "gen_vq_ce_dev")
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


def _end_state(root: Path, run_id: str, *, joint: bool = False) -> None:
    """SPEC 10.4 end state (T4.16 layout): 256 x 5 at every milestone in eval/, and
    the final milestone re-evaluated at 1024 x 5 in eval/final/."""
    _eval(root, run_id, [1000 * i for i in range(1, 11)], joint=joint, done=True,
          last_fraction=1.0)
    _final(root, run_id)


def _legacy_end_state(root: Path, run_id: str) -> None:
    """Pre-T4.16 layout: the 1024 x 5 final eval replaced the final milestone's
    ``eval/step_<n>.json`` (review 2026-11-26, M1)."""
    _eval(root, run_id, [1000 * i for i in range(1, 11)], done=True, last_fraction=1.0)
    final = root / run_id / "eval" / "step_010000.json"
    record = json.loads(final.read_text(encoding="utf-8"))
    record["n_samples"] = 1024
    final.write_text(json.dumps(record), encoding="utf-8")


def _set_params(root: Path, run_id: str, **counts: int) -> None:
    for path in (root / run_id / "eval").rglob("step_*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        for key in ("n_params_tokenizer", "n_params_backbone", "n_params_head",
                    "n_params_total"):
            record.pop(key, None)
        record.update({f"n_params_{name}": value for name, value in counts.items()})
        path.write_text(json.dumps(record), encoding="utf-8")


def test_table_and_fig_a_are_both_final_in_the_spec_end_state(tmp_path: Path, capsys) -> None:
    """Review 2026-11-26 M1 and T4.16: with the 1024 x 5 final eval in eval/final/,
    Table 1 and Figure A are final in the same state."""
    _end_state(tmp_path, "gen_vq_ce_dev")
    arms = [("vq + categorical", "gen_vq_ce_dev")]
    rows, placeholder = paper_results.table_rows(tmp_path, arms)
    assert rows[0]["reasons"] == [] and placeholder is False
    series = plot_fig_a.arm_series(tmp_path, arms)
    assert series[0]["reasons"] == []
    assert series[0]["large_steps"] == set()
    out = tmp_path / "a.png"
    assert plot_fig_a.main(["--results", str(tmp_path), "--out", str(out)]) == 0
    assert out.stat().st_size > 10_000
    printed = capsys.readouterr().out
    assert "gen_vq_ce_dev): final\n" in printed


def test_fig_a_accepts_a_legacy_1024_final_point(tmp_path: Path, capsys) -> None:
    """Review 2026-11-26 M1, kept for records written before T4.16."""
    _legacy_end_state(tmp_path, "gen_vq_ce_dev")
    series = plot_fig_a.arm_series(tmp_path, [("vq + categorical", "gen_vq_ce_dev")])
    assert series[0]["reasons"] == []
    assert series[0]["large_steps"] == {10000}
    assert plot_fig_a.main(["--results", str(tmp_path), "--out", str(tmp_path / "a.png")]) == 0
    assert "gen_vq_ce_dev): final (last point 1024 x 5)" in capsys.readouterr().out


@pytest.mark.parametrize("samples, seeds", [(32, 1), (128, 5), (256, 4)])
def test_fig_a_still_flags_under_sampled_points(tmp_path: Path, samples: int,
                                                seeds: int) -> None:
    _end_state(tmp_path, "gen_vq_ce_dev")
    path = tmp_path / "gen_vq_ce_dev" / "eval" / "step_003000.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record.update(n_samples=samples, n_seeds=seeds)
    path.write_text(json.dumps(record), encoding="utf-8")
    series = plot_fig_a.arm_series(tmp_path, [("vq + categorical", "gen_vq_ce_dev")])
    assert series[0]["reasons"] == [f"eval {samples} x {seeds} (needs at least 256 x 5)"]


def test_fig_a_formats_both_panels(tmp_path: Path, monkeypatch) -> None:
    """Both panels get the log scale, title and axis label, with and without a
    1024-sample final point (the legend proxy once broke this loop)."""
    kept: list = []
    monkeypatch.setattr(plot_fig_a.plt, "close", kept.append)
    _legacy_end_state(tmp_path, "gen_vq_ce_dev")
    _eval(tmp_path, "gen_noisy_flow_dev", [1000, 2000], done=True, last_fraction=1.0)
    titles = [plot_fig_a.TITLES[a] for a in plot_fig_a.ACCOUNTINGS]
    for arm in (("vq + categorical", "gen_vq_ce_dev"), ("continuous + flow", "gen_noisy_flow_dev")):
        plot_fig_a.plot(plot_fig_a.arm_series(tmp_path, [arm]), tmp_path / "a.png")
        axes = kept[-1].axes
        assert [ax.get_title() for ax in axes] == titles
        assert all(ax.get_yscale() == "log" and ax.get_xlabel() for ax in axes)

def test_table_footnotes_the_tokenizer_a_joint_run_trains(tmp_path: Path) -> None:
    """Review 2026-11-26 M2: a joint run also trains its tokenizer (train_jepa.py gives
    AdamW the tokenizer, backbone and head), so its backbone + head cell is footnoted."""
    for run_id, joint in (("gen_noisy_flow_dev", False), ("jepa_base_dev", True)):
        _end_state(tmp_path, run_id, joint=joint)
        _set_params(tmp_path, run_id, tokenizer=8_270_048, backbone=85_459_200,
                    head=15_065_184, total=108_794_432)
    arms = [("continuous + flow", "gen_noisy_flow_dev"), ("JEPA (base)", "jepa_base_dev")]
    rows, placeholder = paper_results.table_rows(tmp_path, arms)
    assert placeholder is False
    text = paper_results.render_table(rows, placeholder)
    lines = text.splitlines()
    flow = next(line for line in lines if line.startswith("| continuous + flow"))
    jepa = next(line for line in lines if line.startswith("| JEPA (base)"))
    mark = paper_results.JOINT_MARK
    assert "| 100.5M |" in flow and mark not in flow
    assert f"| 100.5M{mark} |" in jepa
    note = next(line for line in lines if line.startswith(f"- {mark}"))
    assert "`jepa_base_dev`" in note and "8.27M" in note and "108.8M" in note
    assert "frozen tokenizer is excluded" not in text


def test_joint_row_without_a_tokenizer_count_says_so(tmp_path: Path) -> None:
    _end_state(tmp_path, "jepa_base_dev", joint=True)
    _set_params(tmp_path, "jepa_base_dev", backbone=85_459_200, head=15_065_184)
    rows, placeholder = paper_results.table_rows(tmp_path, [("JEPA (base)", "jepa_base_dev")])
    text = paper_results.render_table(rows, placeholder)
    note = next(line for line in text.splitlines()
                if line.startswith(f"- {paper_results.JOINT_MARK}"))
    assert "n_params_tokenizer" in note and "not logged" in note


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

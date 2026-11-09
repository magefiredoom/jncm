"""Tests for scripts/gate_report.py (T4.4, gates G0/G1/G2).

Fixture eval JSON only: no GPU, no torch, no real runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.gate_report import main


def _write_run(root: Path, run_id: str, points: list[dict]) -> Path:
    """Write one merged aggregate per point under results/<run_id>/eval/."""
    eval_dir = root / run_id / "eval"
    eval_dir.mkdir(parents=True)
    for point in points:
        step = int(point["step"])
        record = {
            "run_id": run_id,
            "checkpoint": f"milestone_{step:03d}.pt",
            "step": step,
            "fraction": point.get("fraction"),
            "total_positions": point["total_positions"],
            "n_samples": 256,
            "n_seeds": 5,
            "ppl_mean": point["ppl"],
            "ppl_std": point.get("std", 0.10),
            "h_gpt2_mean": point.get("h", 4.00),
            "h_dataset_mean": point.get("h_dataset", 4.00),
            "unique_frac_mean": point.get("unique", 0.90),
            "pairwise_agree_mean": point.get("agree", 0.10),
        }
        if point.get("conservative") is not None:
            record["total_positions_conservative"] = point["conservative"]
        (eval_dir / f"step_{step:06d}.json").write_text(
            json.dumps(record), encoding="utf-8"
        )
    return eval_dir


def _args(root: Path, **overrides: str) -> list[str]:
    args = ["--results", str(root), "--jepa", overrides.pop("jepa", "jepa_run")]
    for key, value in overrides.items():
        args += [f"--{key}", value]
    return args


def test_table_has_one_row_per_milestone_and_judges_g0(
    tmp_path: Path, capsys
) -> None:
    g0 = "gen_vq_ce_dev"
    eval_dir = _write_run(
        tmp_path,
        g0,
        [
            {"step": 100, "fraction": 0.5, "total_positions": 1000, "ppl": 18.40, "h": 3.95},
            {"step": 200, "fraction": 1.0, "total_positions": 2000, "ppl": 22.00, "h": 3.70},
        ],
    )
    # A per-seed shard must be ignored (it has no aggregate fields).
    (eval_dir / "step_000100.seed_0.json").write_text(
        json.dumps({"step": 100, "seed": 0, "ppl": 999.9}), encoding="utf-8"
    )

    exit_code = main(_args(tmp_path))

    out = capsys.readouterr().out
    assert exit_code == 1
    assert "18.40 +/- 0.10" in out
    assert "1.704" in out  # 18.40 / 10.80
    assert "-0.080" in out  # 3.95 - 4.03
    assert "999.90" not in out
    assert "G0 step 100 (50%): PASS" in out
    assert "G0 step 200 (100%): FAIL" in out
    assert "G0: FAIL" in out


def test_g1_interpolates_and_flags_regress(tmp_path: Path, capsys) -> None:
    _write_run(
        tmp_path,
        "gen_vq_ce_dev",
        [
            {"step": 100, "total_positions": 100, "ppl": 10.00},
            {"step": 200, "total_positions": 300, "ppl": 20.00},
        ],
    )
    _write_run(
        tmp_path,
        "gen_noisy_flow_dev",
        [
            # x = 200 sits between G0's two points: interpolated G0 PPL is 15.00.
            {"step": 100, "total_positions": 200, "ppl": 17.00, "h": 4.05},
            # x = 1000 is past G0's range: not comparable.
            {"step": 200, "total_positions": 1000, "ppl": 12.00, "h": 4.10},
        ],
    )
    _write_run(
        tmp_path,
        "gen_noisy_regress_dev",
        [{"step": 100, "total_positions": 200, "ppl": 25.00, "agree": 0.90}],
    )
    _write_run(
        tmp_path,
        "jepa_run",
        [
            {
                "step": 100,
                "total_positions": 5000,
                "conservative": 10000,
                "ppl": 12.00,
            }
        ],
    )

    exit_code = main(_args(tmp_path))

    out = capsys.readouterr().out
    assert "ratio 1.133" in out
    assert "G1 gen_noisy_flow_dev step 100 (n/a): PASS" in out
    assert "not comparable at total_positions 1000" in out
    assert "outside gen_vq_ce_dev [100, 300]" in out
    assert "FLAG pairwise_agree > 0.5" in out
    assert "reported only" in out
    assert "G1 (gen_noisy_flow_dev): PASS (1 compared milestones)" in out
    assert exit_code == 0


def test_g1_fails_when_flow_is_over_the_ratio(tmp_path: Path, capsys) -> None:
    _write_run(
        tmp_path,
        "gen_vq_ce_dev",
        [{"step": 100, "total_positions": 1000, "ppl": 15.00}],
    )
    _write_run(
        tmp_path,
        "gen_noisy_flow_dev",
        [{"step": 100, "total_positions": 1000, "ppl": 18.00}],  # ratio 1.20
    )
    _write_run(
        tmp_path,
        "gen_noisy_regress_dev",
        [{"step": 100, "total_positions": 1000, "ppl": 30.00}],
    )

    exit_code = main(_args(tmp_path))

    out = capsys.readouterr().out
    assert "ratio 1.200 > 1.150" in out
    assert "G1 (gen_noisy_flow_dev): FAIL" in out
    assert exit_code == 1


def test_g2_judges_both_accountings_and_names_them(tmp_path: Path, capsys) -> None:
    _write_run(
        tmp_path,
        "gen_vq_ce_dev",
        [{"step": 100, "total_positions": 500, "ppl": 11.00, "h": 3.95}],
    )
    _write_run(
        tmp_path,
        "gen_noisy_flow_dev",
        [
            {"step": 100, "total_positions": 500, "ppl": 11.00, "std": 0.50},
            {"step": 200, "total_positions": 1500, "ppl": 13.00, "std": 0.50},
        ],
    )
    _write_run(
        tmp_path,
        "gen_noisy_regress_dev",
        [
            {"step": 100, "total_positions": 500, "ppl": 14.00, "std": 0.50},
            {"step": 200, "total_positions": 1500, "ppl": 16.00, "std": 0.50},
        ],
    )
    _write_run(
        tmp_path,
        "jepa_run",
        [
            {
                "step": 100,
                "total_positions": 1000,
                "conservative": 2000,
                "ppl": 12.10,
                "std": 0.20,
                "h": 4.00,
            }
        ],
    )

    exit_code = main(_args(tmp_path))

    out = capsys.readouterr().out
    # total_positions: interpolated best Phase-1 is flow at 12.00 +/- 0.50.
    assert "vs best Phase-1 gen_noisy_flow_dev 12.00 +/- 0.50 (limit 12.50)" in out
    assert "at total_positions 1000" in out
    assert "G2 (accounting: total_positions): PASS (1 compared)" in out
    assert "G2 (accounting: total_positions_conservative): not comparable" in out
    assert "G2: total_positions PASS, total_positions_conservative not comparable" in out
    assert exit_code == 0


def test_g2_fails_when_jepa_is_worse(tmp_path: Path, capsys) -> None:
    for run_id, ppl in (("gen_noisy_flow_dev", 11.00), ("gen_noisy_regress_dev", 14.00)):
        _write_run(
            tmp_path,
            run_id,
            [{"step": 100, "total_positions": 1000, "ppl": ppl, "std": 0.50}],
        )
    _write_run(tmp_path, "gen_vq_ce_dev", [{"step": 100, "total_positions": 1000, "ppl": 12.00}])
    _write_run(
        tmp_path,
        "jepa_run",
        [
            {
                "step": 100,
                "total_positions": 1000,
                "conservative": 2000,
                "ppl": 20.00,
                "h": 4.00,
            }
        ],
    )

    exit_code = main(_args(tmp_path))

    out = capsys.readouterr().out
    assert "G2 (accounting: total_positions): FAIL" in out
    assert exit_code == 1


def test_dense_ten_point_curves_get_one_row_per_milestone(
    tmp_path: Path, capsys
) -> None:
    """T5.5b F14: the 10% milestone grid keeps every point in the report."""
    points = [
        {
            "step": 100 * index,
            "fraction": round(0.1 * index, 1),
            "total_positions": 100 * index,
            "ppl": 10.0 + index,
        }
        for index in range(1, 11)
    ]
    _write_run(tmp_path, "gen_vq_ce_dev", points)
    _write_run(tmp_path, "gen_noisy_flow_dev", points)
    _write_run(tmp_path, "gen_noisy_regress_dev", points)
    _write_run(tmp_path, "jepa_run", points)

    exit_code = main(_args(tmp_path))

    out = capsys.readouterr().out
    for index in range(1, 11):
        assert f"{index * 10}%" in out
    assert "G0: PASS" in out
    assert "G1 (gen_noisy_flow_dev): PASS (10 compared milestones)" in out
    assert "G2 (accounting: total_positions): PASS (10 compared)" in out
    assert exit_code == 0


def test_missing_runs_are_reported_not_silently_passed(
    tmp_path: Path, capsys
) -> None:
    exit_code = main(["--results", str(tmp_path), "--jepa", "jepa_run"])

    out = capsys.readouterr().out
    assert "G0 gen_vq_ce_dev: MISSING" in out
    assert "G1 gen_noisy_flow_dev: MISSING" in out
    assert "G2 jepa_run: MISSING" in out
    assert exit_code == 1

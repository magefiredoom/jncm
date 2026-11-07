"""Tests for scripts/plot_curves.py (T4.2) on synthetic eval JSON fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.plot_curves import collect_points, plot_curves


def _write(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")


def _record(run_id: str, step: int, **overrides: object) -> dict:
    record: dict = {
        "run_id": run_id,
        "step": step,
        "ppl_mean": 10.0 + step,
        "ppl_std": 0.2,
        "h_dataset_mean": 4.0,
        "h_gpt2_mean": 4.1,
        "unique_frac_mean": 0.95,
        "pairwise_agree_mean": 0.02,
    }
    record.update(overrides)
    return record


def test_collect_points_reads_aggregates_and_sorts_by_positions(tmp_path: Path) -> None:
    results = tmp_path / "results"
    _write(results / "run_a" / "eval" / "step_000200.json", _record("run_a", 200, total_positions=2000))
    _write(results / "run_a" / "eval" / "step_000100.json", _record("run_a", 100, total_positions=1000))
    _write(results / "run_b" / "eval" / "step_000100.json", _record("run_b", 100, total_positions=500))

    points = collect_points(results)

    assert sorted(points) == ["run_a", "run_b"]
    assert [point["x"] for point in points["run_a"]] == [1000, 2000]
    assert [point["x"] for point in points["run_b"]] == [500]


def test_collect_points_adds_tokenizer_budget_offset(tmp_path: Path) -> None:
    results = tmp_path / "results"
    _write(
        results / "run_b" / "eval" / "step_000100.json",
        _record("run_b", 100, gen_positions=500, tok_positions=100),
    )

    points = collect_points(results)

    assert [point["x"] for point in points["run_b"]] == [600]


def test_collect_points_ignores_per_seed_shards(tmp_path: Path) -> None:
    results = tmp_path / "results"
    _write(results / "run_a" / "eval" / "step_000100.json", _record("run_a", 100, total_positions=1000))
    _write(
        results / "run_a" / "eval" / "step_000100.seed_0.json",
        _record("run_a", 100, total_positions=1000, seed=0),
    )

    points = collect_points(results)

    assert len(points["run_a"]) == 1


def test_collect_points_requires_eval_records(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no eval results"):
        collect_points(tmp_path / "results")


def test_plot_curves_writes_png(tmp_path: Path) -> None:
    results = tmp_path / "results"
    for step in (100, 200):
        _write(
            results / "run_a" / "eval" / f"step_{step:06d}.json",
            _record("run_a", step, total_positions=step * 10),
        )
    _write(
        results / "run_b" / "eval" / "step_000100.json",
        _record("run_b", 100, total_positions=600, h_gpt2_mean=None),
    )
    out = tmp_path / "curves.png"

    written = plot_curves(results, out)

    assert written == out
    assert out.exists()
    assert out.stat().st_size > 1000

"""Tests for the ``--runs`` option of scripts/g0a_report.py (G0a gate)."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.g0a_report import DEFAULT_RUNS, main


def _write_run(root: Path, run_id: str, quantizer: str, acc: float) -> Path:
    """Create results/<run_id>/{config.yaml,metrics.jsonl} for one fixture run."""
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(
        f"tokenizer:\n  quantizer: {quantizer}\n", encoding="utf-8"
    )
    record = {
        "step": 10000,
        "acc_full": acc,
        "acc_cum": [0.5, 0.8, 0.9, acc],
        "data_source": "real",
        "nominal_bits": 12.0,
    }
    (run_dir / "metrics.jsonl").write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )
    return run_dir


def test_default_runs_are_the_10k_gate_pair(tmp_path: Path, capsys) -> None:
    assert DEFAULT_RUNS == ["tok_vq_10k", "tok_noisy_10k"]

    exit_code = main(["--results", str(tmp_path)])

    out = capsys.readouterr().out
    assert exit_code == 1
    assert "tok_vq_10k: MISSING" in out
    assert "tok_noisy_10k: MISSING" in out


def test_explicit_runs_use_their_variant_threshold(
    tmp_path: Path, capsys
) -> None:
    _write_run(tmp_path, "tok_vq_fixture", "vq", 0.9995)
    _write_run(tmp_path, "tok_noisy_fixture", "noisy", 0.996)

    exit_code = main(
        [
            "--results",
            str(tmp_path),
            "--runs",
            "tok_vq_fixture",
            "tok_noisy_fixture",
        ]
    )

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "PASS tok_vq_fixture (vq)" in out
    assert "PASS tok_noisy_fixture (noisy)" in out
    assert "vs 0.999" in out
    assert "vs 0.995" in out


def test_vq_threshold_stays_the_stricter_one(tmp_path: Path, capsys) -> None:
    _write_run(tmp_path, "tok_vq_fixture", "vq", 0.998)

    exit_code = main(["--results", str(tmp_path), "--runs", "tok_vq_fixture"])

    out = capsys.readouterr().out
    assert exit_code == 1
    assert "FAIL tok_vq_fixture (vq)" in out
    assert "vs 0.999" in out


def test_variant_falls_back_to_the_run_id_without_config(
    tmp_path: Path, capsys
) -> None:
    run_dir = _write_run(tmp_path, "tok_noisy_10k", "noisy", 0.9955)
    (run_dir / "config.yaml").unlink()

    exit_code = main(["--results", str(tmp_path), "--runs", "tok_noisy_10k"])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "PASS tok_noisy_10k (noisy)" in out
    assert "vs 0.995" in out


def test_unknown_variant_is_a_failure_not_a_pass(
    tmp_path: Path, capsys
) -> None:
    _write_run(tmp_path, "mystery_run", "vq", 1.0)
    (tmp_path / "mystery_run" / "config.yaml").unlink()

    exit_code = main(["--results", str(tmp_path), "--runs", "mystery_run"])

    out = capsys.readouterr().out
    assert exit_code == 1
    assert "cannot tell the quantizer variant" in out

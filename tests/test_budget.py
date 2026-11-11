"""T4.5: scripts/budget.py projected H100-hours and the 12 h flag."""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import budget  # noqa: E402  (scripts/ is not a package)


def _write_pilot(root: Path, pilot_id: str, sec_per_step: float) -> None:
    directory = root / pilot_id
    directory.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({"step": 1, "sec_per_step": sec_per_step / 2}),
        json.dumps({"step": 2, "sec_per_step": sec_per_step}),
    ]
    (directory / "metrics.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _all_pilots(root: Path, sec_per_step: float, suffix: str = "") -> None:
    for _, _, pilot_id, _ in budget.RUNS:
        _write_pilot(root, budget.suffixed_run_id(pilot_id, suffix), sec_per_step)


def _run(root: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(Path("scripts/budget.py")), "--results", str(root)],
        capture_output=True,
        text=True,
    )


def test_projects_every_run_from_its_pilot(tmp_path: Path) -> None:
    _all_pilots(tmp_path, 1.0)

    result = _run(tmp_path)

    assert result.returncode == 0, result.stderr
    # 10000 steps x 1.0 s/step = 2.78 H100-hours per run, 12 runs.
    assert "2.78" in result.stdout
    assert "TOTAL: 33.33 H100-hours over 12 run(s)" in result.stdout
    assert "FLAG" not in result.stdout
    assert "MISSING" not in result.stdout
    for run_id in ("tok_vq_10k", "gen_noisy_flow_dev", "jepa_base_dev"):
        assert run_id in result.stdout


def test_uses_the_latest_pilot_record(tmp_path: Path) -> None:
    _all_pilots(tmp_path, 1.0)

    result = _run(tmp_path)

    # The first record of every pilot carries 0.5 s/step; only 1.0 must be used.
    assert "0.5000" not in result.stdout
    assert "1.0000" in result.stdout


def test_flags_a_run_above_twelve_hours(tmp_path: Path) -> None:
    _all_pilots(tmp_path, 1.0)
    _write_pilot(tmp_path, "pilot_jepa_base", 5.0)

    result = _run(tmp_path)

    assert result.returncode == 1
    assert "FLAG jepa_base_dev" in result.stdout
    assert "> 12.0 h" in result.stdout


def test_reports_missing_pilots_and_excludes_them(tmp_path: Path) -> None:
    _write_pilot(tmp_path, "pilot_tok_vq", 1.0)

    result = _run(tmp_path)

    assert result.returncode == 1
    assert "MISSING tok_vq_10k" not in result.stdout  # measured
    assert "MISSING gen_vq_ce_dev (pilot pilot_gen_vq_ce)" in result.stdout
    assert "TOTAL: 2.78 H100-hours over 1 run(s)" in result.stdout


def test_accepts_and_prints_the_config(tmp_path: Path) -> None:
    """T5.5b: --config is accepted and names the configuration projected."""
    _all_pilots(tmp_path, 1.0, "lite")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/budget.py",
            "--results",
            str(tmp_path),
            "--config",
            "configs/tinystories_lite.yaml",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "configs/tinystories_lite.yaml" in result.stdout
    assert "gen_vq_ce_lite" in result.stdout


def test_missing_config_fails_loudly(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/budget.py",
            "--results",
            str(tmp_path),
            "--config",
            "configs/nope.yaml",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "nope.yaml" in result.stderr


def test_config_suffix_matches_the_launch_kit() -> None:
    """T5.5c/T5.6: dev gets no suffix, lite gets 'lite', tokenizers never."""
    assert budget.config_suffix(None) == ""
    assert budget.config_suffix(Path("configs/tinystories_dev.yaml")) == ""
    assert budget.config_suffix(Path("configs/jepa_dev.yaml")) == ""
    assert budget.config_suffix(Path("configs/tinystories_lite.yaml")) == "lite"
    assert budget.config_suffix(Path("configs/jepa_lite.yaml")) == "lite"
    assert budget.with_suffix("gen_vq_ce_dev", "lite") == "gen_vq_ce_lite"
    assert budget.with_suffix("gen_vq_ce_dev", "") == "gen_vq_ce_dev"
    assert budget.suffixed_run_id("tok_vq_10k", "lite") == "tok_vq_10k"
    assert budget.suffixed_run_id("pilot_tok_noisy", "lite") == "pilot_tok_noisy"
    assert budget.suffixed_run_id("gen_vq_ce_dev", "lite") == "gen_vq_ce_lite"


def test_lite_config_uses_only_suffixed_pilots(tmp_path: Path) -> None:
    """T5.5c: a dev pilot record never sizes a lite projection."""
    _all_pilots(tmp_path, 1.0)
    _write_pilot(tmp_path, "pilot_gen_vq_ce_lite", 3.0)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/budget.py",
            "--results",
            str(tmp_path),
            "--config",
            "configs/tinystories_lite.yaml",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1  # every other lite pilot is missing
    assert "run-id suffix: _lite" in result.stdout
    assert "gen_vq_ce_lite" in result.stdout
    assert "3.0000" in result.stdout
    assert "gen_vq_ce_dev" not in result.stdout
    assert "MISSING gen_noisy_flow_lite (pilot pilot_gen_noisy_flow_lite)" in result.stdout


def test_lite_projection_is_missing_without_lite_pilots(tmp_path: Path) -> None:
    _all_pilots(tmp_path, 1.0)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/budget.py",
            "--results",
            str(tmp_path),
            "--config",
            "configs/jepa_lite.yaml",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "MISSING jepa_base_lite" in result.stdout
    # T5.6: the two tokenizer rows are shared with the dev config, so they are
    # measured even without lite pilots: 2 x 2.78 H100-hours.
    assert "TOTAL: 5.56 H100-hours over 2 run(s)" in result.stdout


def test_lite_projection_keeps_the_tokenizer_rows_unsuffixed(tmp_path: Path) -> None:
    """T5.6: only generator/JEPA rows carry the 'lite' suffix."""
    _all_pilots(tmp_path, 1.0, "lite")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/budget.py",
            "--results",
            str(tmp_path),
            "--config",
            "configs/jepa_lite.yaml",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "tok_vq_10k_lite" not in result.stdout
    assert "tok_noisy_10k_lite" not in result.stdout
    assert re.search(r"^G0 +tok_vq_10k +10000 +1\.0000", result.stdout, re.M)
    assert re.search(r"^G1 +tok_noisy_10k +10000 +1\.0000", result.stdout, re.M)
    assert re.search(r"^G2 +jepa_base_lite ", result.stdout, re.M)


def test_pilot_reader_ignores_records_without_the_field(tmp_path: Path) -> None:
    directory = tmp_path / "pilot_x"
    directory.mkdir()
    (directory / "metrics.jsonl").write_text(
        json.dumps({"step": 1}) + "\n", encoding="utf-8"
    )

    assert budget.pilot_sec_per_step(tmp_path, "pilot_x") is None
    assert budget.pilot_sec_per_step(tmp_path, "pilot_missing") is None


@pytest.mark.parametrize("step", [0, -1])
def test_run_table_covers_the_seven_jepa_arms(tmp_path: Path, step: int) -> None:
    jepa_runs = [run_id for phase, run_id, _, _ in budget.RUNS if phase == "G2"]

    assert len(jepa_runs) == 7
    assert jepa_runs[step] in {
        "jepa_base_dev",
        "jepa_no_vicreg_dev",
        "jepa_no_dec_through_E_dev",
        "jepa_neither_dev",
        "jepa_ema_clean_dev",
        "jepa_regress_dev",
        "jepa_no_ema_dev",
    }

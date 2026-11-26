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


def _done(root: Path, run_id: str) -> None:
    directory = root / run_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "DONE").write_text("{}\n", encoding="utf-8")


def test_prefers_train_sec_per_step_over_wall_time(tmp_path: Path) -> None:
    """T5.7 m1: wall time includes diagnostics, Gen PPL and checkpoint saves."""
    _all_pilots(tmp_path, 9.0)  # wall time only: would be 25 h per run
    pilot = tmp_path / "pilot_gen_vq_ce"
    last = {"step": 2, "sec_per_step": 9.0, "train_sec_per_step": 1.0}
    (pilot / "metrics.jsonl").write_text(
        json.dumps({"step": 1, "sec_per_step": 4.5}) + "\n" + json.dumps(last) + "\n",
        encoding="utf-8",
    )

    assert budget.pilot_sec_per_step(tmp_path, "pilot_gen_vq_ce") == 1.0
    assert budget.pilot_sec_per_step(tmp_path, "pilot_tok_vq") == 9.0  # older pilot
    result = _run(tmp_path)
    assert "train_sec_per_step" in result.stdout


def test_a_done_tokenizer_is_finished_zero_hours_not_missing(tmp_path: Path) -> None:
    """T5.7 m5(a): results/<run>/DONE prints 0.00 hours, never MISSING."""
    _done(tmp_path, "tok_vq_10k")
    _write_pilot(tmp_path, "pilot_gen_vq_ce", 1.0)

    result = _run(tmp_path)

    assert result.returncode == 1  # everything else is still missing
    assert "MISSING tok_vq_10k" not in result.stdout
    assert re.search(r"^G0 +tok_vq_10k +10000 +DONE +0\.00$", result.stdout, re.M)
    # The DONE row counts as measured (0.00 h), so it joins the total.
    assert "TOTAL: 2.78 H100-hours over 2 run(s)" in result.stdout
    assert "MISSING gen_vq_ce_dev (pilot pilot_gen_vq_ce)" not in result.stdout
    assert "MISSING gen_noisy_flow_dev (pilot pilot_gen_noisy_flow)" in result.stdout


def test_total_covers_all_twelve_runs_with_done_tokenizers(tmp_path: Path) -> None:
    """T5.7 m5: DONE tokenizers (0 h), generator pilots and the base pilot."""
    _done(tmp_path, "tok_vq_10k")
    _done(tmp_path, "tok_noisy_10k")
    _write_pilot(tmp_path, "pilot_gen_vq_ce", 1.0)
    _write_pilot(tmp_path, "pilot_gen_noisy_flow", 1.0)
    _write_pilot(tmp_path, "pilot_gen_noisy_regress", 1.0)
    _write_pilot(tmp_path, "pilot_jepa_base", 1.0)
    # No ablation pilot exists on purpose: the ablations read the base pilot.

    result = _run(tmp_path)

    assert result.returncode == 0, result.stderr
    assert "MISSING" not in result.stdout
    assert "FLAG" not in result.stdout
    # 10 measured runs x 2.78 h + the two DONE tokenizer rows at 0.00 hours.
    assert "TOTAL: 27.78 H100-hours over 12 run(s)" in result.stdout
    for phase in ("G0", "G1", "G2"):
        assert f"{phase} subtotal:" in result.stdout


def test_ablations_are_projected_from_the_base_pilot(tmp_path: Path) -> None:
    """T5.7 m5(b): the six G2 ablation rows reuse pilot_jepa_base's measurement."""
    _all_pilots(tmp_path, 1.0)
    _write_pilot(tmp_path, "pilot_jepa_base", 5.0)

    result = _run(tmp_path)

    assert result.returncode == 1  # the base and every ablation are flagged
    assert "FLAG jepa_base_dev" in result.stdout
    for run_id in (
        "jepa_no_vicreg_dev",
        "jepa_no_dec_through_E_dev",
        "jepa_neither_dev",
        "jepa_ema_clean_dev",
        "jepa_regress_dev",
        "jepa_no_ema_dev",
    ):
        assert re.search(rf"^G2 +{run_id} +10000 +5\.0000 +13\.89", result.stdout, re.M)


def test_lite_ablations_are_sized_from_the_lite_base_pilot(tmp_path: Path) -> None:
    """T5.7 m5: the ablation fallback carries the config suffix (_lite)."""
    _done(tmp_path, "tok_vq_10k")
    _done(tmp_path, "tok_noisy_10k")
    _write_pilot(tmp_path, "pilot_gen_vq_ce_lite", 1.0)
    _write_pilot(tmp_path, "pilot_gen_noisy_flow_lite", 1.0)
    _write_pilot(tmp_path, "pilot_gen_noisy_regress_lite", 1.0)
    _write_pilot(tmp_path, "pilot_jepa_base_lite", 1.0)
    # No ablation _lite pilot exists on purpose: they read the base _lite pilot.

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
    assert "MISSING" not in result.stdout
    assert re.search(r"^G2 +jepa_no_vicreg_lite +10000 +1\.0000", result.stdout, re.M)
    assert "TOTAL: 27.78 H100-hours over 12 run(s)" in result.stdout


# T4.10: the disk a run writes (SPEC 7 v1.7). Counts from the real pilots' first
# metrics records (results/pilot_gen_vq_ce, results/pilot_jepa_base, 2026-11-20/21).
CE_COUNTS = {
    "n_params_tokenizer": 11932544,
    "n_params_backbone": 85459200,
    "n_params_head": 37797888,
    "n_params_total": 135189632,
}
JEPA_COUNTS = {
    "n_params_tokenizer": 8270048,
    "n_params_backbone": 85459200,
    "n_params_head": 15065184,
    "n_params_total": 108794432,
}


def _write_counted_pilot(
    root: Path, pilot_id: str, counts: dict[str, int], **extra: object
) -> None:
    directory = root / pilot_id
    directory.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({"step": 1, "sec_per_step": 2.0, **counts, **extra}),
        json.dumps({"step": 300, "sec_per_step": 3.0, "train_sec_per_step": 2.94}),
    ]
    (directory / "metrics.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _disk_line(stdout: str, run_id: str) -> str:
    lines = [line for line in stdout.splitlines() if line.startswith(f"DISK {run_id}:")]
    assert len(lines) == 1, (run_id, stdout)
    return lines[0]


def test_dev_ce_run_writes_about_9_4_gb(tmp_path: Path) -> None:
    """Acceptance: 10 milestones of 4 B x (backbone + head) = 0.493 GB each, plus
    latest.pt and step_005000/step_010000.pt with Adam state (3 x 0.493 GB each);
    the frozen tokenizer is not in a generator checkpoint. FIR measured 493,107,205 B
    per milestone and 1,479,308,875 B for latest.pt (pilot_gen_vq_ce)."""
    _all_pilots(tmp_path, 1.0)
    _write_counted_pilot(tmp_path, "pilot_gen_vq_ce", CE_COUNTS)

    result = _run(tmp_path)

    assert result.returncode == 0, result.stderr
    line = _disk_line(result.stdout, "gen_vq_ce_dev")
    assert "9.37 GB" in line and "8.72 GiB" in line
    assert "10 milestone(s) x 0.493 GB" in line
    assert "3 resume file(s) x 1.479 GB" in line
    assert "whole run" in line
    assert "pilot_gen_vq_ce" in line
    assert "configs/tinystories_dev.yaml" in line


def test_joint_run_counts_the_tokenizer_and_its_ema_copy(tmp_path: Path) -> None:
    """JEPA checkpoints hold E, D, G and the EMA copy of the tokenizer (Q10);
    Adam covers E, D and G only. Predicted milestone 4 x (108,794,432 + 8,270,048)
    = 468,257,920 B against 468,437,855 B measured; latest.pt 12 x 108,794,432 +
    4 x 8,270,048 = 1,338,613,376 B against 1,339,093,729 B (pilot_jepa_base)."""
    _all_pilots(tmp_path, 1.0)
    _write_counted_pilot(tmp_path, "pilot_jepa_base", JEPA_COUNTS, ema=True)

    result = _run(tmp_path)

    assert result.returncode == 0, result.stderr
    line = _disk_line(result.stdout, "jepa_base_dev")
    assert "10 milestone(s) x 0.468 GB" in line
    assert "3 resume file(s) x 1.339 GB" in line
    assert "8.70 GB" in line
    assert "configs/jepa_dev.yaml" in line


def test_joint_run_without_ema_has_no_ema_copy(tmp_path: Path) -> None:
    _all_pilots(tmp_path, 1.0)
    _write_counted_pilot(tmp_path, "pilot_jepa_base", JEPA_COUNTS, ema=False)

    result = _run(tmp_path)

    line = _disk_line(result.stdout, "jepa_base_dev")
    assert "10 milestone(s) x 0.435 GB" in line  # 4 x 108,794,432
    assert "3 resume file(s) x 1.306 GB" in line  # 12 x 108,794,432


def test_ablations_report_an_upper_bound_from_the_base_pilot(tmp_path: Path) -> None:
    _all_pilots(tmp_path, 1.0)
    _write_counted_pilot(tmp_path, "pilot_jepa_base", JEPA_COUNTS, ema=True)

    result = _run(tmp_path)

    base = _disk_line(result.stdout, "jepa_base_dev")
    assert "upper bound" not in base
    for run_id in ("jepa_regress_dev", "jepa_no_ema_dev", "jepa_no_vicreg_dev"):
        line = _disk_line(result.stdout, run_id)
        assert "8.70 GB" in line
        assert "upper bound (base pilot pilot_jepa_base)" in line


def test_disk_is_na_without_parameter_counts_and_the_exit_code_is_unchanged(
    tmp_path: Path,
) -> None:
    _all_pilots(tmp_path, 1.0)  # no n_params_* anywhere

    result = _run(tmp_path)

    assert result.returncode == 0, result.stderr
    line = _disk_line(result.stdout, "gen_vq_ce_dev")
    assert "n/a" in line and "pilot_gen_vq_ce has no n_params_*" in line


def test_a_done_run_writes_nothing_more(tmp_path: Path) -> None:
    _all_pilots(tmp_path, 1.0)
    _done(tmp_path, "tok_vq_10k")

    result = _run(tmp_path)

    assert "DONE" in _disk_line(result.stdout, "tok_vq_10k")


def test_disk_total_sums_the_runs_with_a_size(tmp_path: Path) -> None:
    _all_pilots(tmp_path, 1.0)
    _write_counted_pilot(tmp_path, "pilot_gen_vq_ce", CE_COUNTS)
    _write_counted_pilot(tmp_path, "pilot_jepa_base", JEPA_COUNTS, ema=True)

    result = _run(tmp_path)

    # gen_vq_ce_dev 9.3675 GB + 7 JEPA rows x 8.6984 GB = 70.26 GB; 4 rows n/a.
    totals = [line for line in result.stdout.splitlines() if line.startswith("DISK TOTAL:")]
    assert len(totals) == 1, result.stdout
    assert "70.26 GB" in totals[0]
    assert "8 run(s)" in totals[0] and "4 n/a" in totals[0]
    assert "GB = 1e9 B" in result.stdout


def test_lite_config_reads_the_cadence_of_both_families(tmp_path: Path) -> None:
    for pilot in ("pilot_gen_vq_ce_lite", "pilot_jepa_base_lite"):
        _write_counted_pilot(tmp_path, pilot, CE_COUNTS if "gen" in pilot else JEPA_COUNTS, ema=True)

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

    assert "configs/tinystories_lite.yaml" in _disk_line(result.stdout, "gen_vq_ce_lite")
    assert "configs/jepa_lite.yaml" in _disk_line(result.stdout, "jepa_base_lite")

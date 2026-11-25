"""scripts/pull_results.sh builds a copy-only rsync from FIR (no username in the repo)."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "pull_results.sh"
BASH = shutil.which("bash")

pytestmark = pytest.mark.skipif(BASH is None, reason="needs bash")


def _run(args: list[str], tmp_path: Path, **env: str) -> subprocess.CompletedProcess[str]:
    (tmp_path / "results").mkdir(exist_ok=True)
    full_env = {k: v for k, v in os.environ.items() if k not in {"FIR_USER", "FIR_SSH", "FIR_RESULTS"}}
    full_env.update(env)
    return subprocess.run(
        [BASH, str(SCRIPT), *args],
        cwd=tmp_path,
        env=full_env,
        capture_output=True,
        text=True,
    )


def test_refuses_without_fir_user(tmp_path: Path) -> None:
    result = _run(["--print"], tmp_path)
    assert result.returncode == 2
    assert "source site.env" in result.stderr


def test_all_runs_copy_only_and_skip_tmp(tmp_path: Path) -> None:
    result = _run(["--print"], tmp_path, FIR_USER="someone")
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "rsync -rt --partial" in out
    assert "--exclude=\\*.tmp" in out or "--exclude=*.tmp" in out
    assert "--delete" not in out
    assert "someone@fir.alliancecan.ca:/scratch/someone/jncm/results/" in out
    assert out.strip().endswith("results/")


def test_skip_running_excludes_the_main_chains(tmp_path: Path) -> None:
    result = _run(["--print", "--skip-running"], tmp_path, FIR_USER="someone")
    assert result.returncode == 0, result.stderr
    for run in ["gen_vq_ce_dev", "gen_noisy_flow_dev", "gen_noisy_regress_dev", "jepa_base_dev"]:
        assert f"--exclude=/{run}/" in result.stdout


def test_named_runs_and_a_bad_id(tmp_path: Path) -> None:
    ok = _run(["--print", "corr_p080", "tok_vq_10k"], tmp_path, FIR_USER="someone", FIR_SSH="fir")
    assert ok.returncode == 0, ok.stderr
    assert "fir:/scratch/someone/jncm/results/corr_p080" in ok.stdout
    assert "fir:/scratch/someone/jncm/results/tok_vq_10k" in ok.stdout
    bad = _run(["--print", "../etc"], tmp_path, FIR_USER="someone")
    assert bad.returncode == 2

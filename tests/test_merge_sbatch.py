"""T4.9: scripts/slurm/merge.sbatch, the CPU job that merges per-seed eval shards.

SPEC 6.1 (v1.7): per-seed eval shards are merged by a CPU job chained `afterok`
on the eval array; no Python runs on a login node. The template asks for one
CPU, 4 GB and 15 minutes, requests no GPU, requires RUN_ID (exit 2) and runs
`python scripts/run_eval.py --run_id "$RUN_ID" --merge`.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest

SCRIPT = Path("scripts/slurm/merge.sbatch")


def _working_bash() -> str | None:
    """First bash that can run a trivial command (Windows may find broken WSL)."""
    candidates = [shutil.which("bash"), shutil.which("bash.exe")]
    if os.name == "nt":
        candidates += [
            r"C:\Program Files\Git\bin\bash.exe",
            r"C:\Program Files\Git\usr\bin\bash.exe",
        ]
    for candidate in candidates:
        if not candidate:
            continue
        probe = subprocess.run(
            [candidate, "-c", "echo ok"], capture_output=True, text=True
        )
        if probe.returncode == 0 and "ok" in probe.stdout:
            return candidate
    return None


BASH = _working_bash()


def _text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def _directive(name: str) -> str | None:
    match = re.search(rf"^#SBATCH\s+--{name}=(\S+)\s*$", _text(), flags=re.MULTILINE)
    return match.group(1) if match else None


def test_merge_template_requests_one_cpu_4g_and_15_minutes() -> None:
    assert _directive("cpus-per-task") == "1"
    assert _directive("mem") == "4G"
    assert _directive("time") == "00:15:00"


def test_merge_template_requests_no_gpu_and_no_array() -> None:
    sbatch_lines = [line for line in _text().splitlines() if line.startswith("#SBATCH")]

    assert sbatch_lines, "no #SBATCH lines"
    assert not any("--gres" in line or "--gpus" in line for line in sbatch_lines)
    assert not any("--array" in line for line in sbatch_lines)
    # A CPU job: the GPU guard of the GPU templates must not be exported here.
    assert "JNCM_REQUIRE_CUDA=1" not in _text()


def test_merge_template_guesses_no_account_or_partition() -> None:
    sbatch_lines = [line for line in _text().splitlines() if line.startswith("#SBATCH")]

    assert not any("account" in line or "partition" in line for line in sbatch_lines)
    assert "set -euo pipefail" in _text()


needs_bash = pytest.mark.skipif(BASH is None, reason="no working bash on this host")


@needs_bash
def test_merge_template_parses_with_bash() -> None:
    result = subprocess.run([BASH, "-n", str(SCRIPT)], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


def _fixture_repo(tmp_path: Path) -> Path:
    """Minimal tree for the script body, a fake python and a no-op `module`."""
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "run_eval.py").write_text("", encoding="utf-8")
    (root / "results" / "fake_run" / "eval").mkdir(parents=True)
    venv = root / ".venv" / "bin"
    venv.mkdir(parents=True)
    (venv / "activate").write_text(
        "python() { printf 'PYTHON_ARGS'; printf ' %s' \"$@\"; printf '\\n'; }\n",
        encoding="utf-8",
    )
    bindir = tmp_path / "bin"
    bindir.mkdir()
    stub = bindir / "module"
    stub.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    stub.chmod(0o755)
    return root


def _run_merge(root: Path, extra_env: dict[str, str]) -> subprocess.CompletedProcess:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"RUN_ID", "SLURM_SUBMIT_DIR"}
    }
    env["SLURM_SUBMIT_DIR"] = str(root)
    env["PATH"] = str(root.parent / "bin") + os.pathsep + env.get("PATH", "")
    env.update(extra_env)
    return subprocess.run(
        [BASH, str(SCRIPT.resolve())],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
    )


@needs_bash
def test_merge_without_run_id_exits_2_with_a_message(tmp_path: Path) -> None:
    root = _fixture_repo(tmp_path)

    result = _run_merge(root, {})

    assert result.returncode == 2, (result.stdout, result.stderr)
    assert "RUN_ID" in result.stderr
    assert "PYTHON_ARGS" not in result.stdout


@needs_bash
def test_merge_with_an_empty_run_id_exits_2(tmp_path: Path) -> None:
    root = _fixture_repo(tmp_path)

    result = _run_merge(root, {"RUN_ID": ""})

    assert result.returncode == 2
    assert "RUN_ID" in result.stderr


@needs_bash
def test_merge_runs_run_eval_merge_for_the_run(tmp_path: Path) -> None:
    root = _fixture_repo(tmp_path)

    result = _run_merge(root, {"RUN_ID": "fake_run"})

    assert result.returncode == 0, result.stderr
    calls = [line for line in result.stdout.splitlines() if line.startswith("PYTHON_ARGS")]
    assert calls == ["PYTHON_ARGS scripts/run_eval.py --run_id fake_run --merge"]


@needs_bash
def test_merge_of_a_run_without_eval_shards_directory_fails_loudly(
    tmp_path: Path,
) -> None:
    root = _fixture_repo(tmp_path)

    result = _run_merge(root, {"RUN_ID": "no_such_run"})

    assert result.returncode == 1
    assert "results/no_such_run/eval" in result.stderr
    assert "PYTHON_ARGS" not in result.stdout

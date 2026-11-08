"""Checks for scripts/slurm/eval.sbatch (T4.2).

The file is syntax-checked with a real bash and audited for the T4.1
conventions: no account or partition in any #SBATCH line, the offline
variables, a seed --array, and marked placeholders instead of guesses.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest

SCRIPT = Path("scripts/slurm/eval.sbatch")


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

pytestmark = pytest.mark.skipif(BASH is None, reason="no working bash on this host")


def test_eval_sbatch_parses_with_bash() -> None:
    result = subprocess.run(
        [BASH, "-n", str(SCRIPT)], capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr


def test_eval_sbatch_follows_the_t4_1_conventions() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    sbatch_lines = [
        line for line in text.splitlines() if line.startswith("#SBATCH")
    ]

    assert sbatch_lines, "no #SBATCH lines"
    assert not any("account" in line or "partition" in line for line in sbatch_lines)
    assert any("--array" in line for line in sbatch_lines)
    assert "SLURM_ARRAY_TASK_ID" in text
    assert "set -euo pipefail" in text
    for variable in ("HF_HUB_OFFLINE=1", "HF_DATASETS_OFFLINE=1", "TRANSFORMERS_OFFLINE=1"):
        assert variable in text, variable
    assert "TODO(placeholder)" in text
    assert "results/" in text


def _fixture_repo(tmp_path: Path) -> Path:
    """Minimal tree for the script body, plus a fake python that echoes argv."""
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "run_eval.py").write_text("", encoding="utf-8")
    (root / "results" / "fake_run" / "ckpt").mkdir(parents=True)
    venv = root / ".venv" / "bin"
    venv.mkdir(parents=True)
    (venv / "activate").write_text(
        "python() { printf 'PYTHON_ARGS'; printf ' %s' \"$@\"; printf '\\n'; }\n",
        encoding="utf-8",
    )
    return root


def _run_eval_sbatch(
    root: Path, extra_env: dict[str, str]
) -> subprocess.CompletedProcess:
    """Run the sbatch body in the fixture tree with a controlled environment."""
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"TOKENIZER", "RUN_ID", "SLURM_SUBMIT_DIR", "SLURM_ARRAY_TASK_ID"}
    }
    env.update({"SLURM_SUBMIT_DIR": str(root), "RUN_ID": "fake_run"})
    env.update(extra_env)
    return subprocess.run(
        [BASH, str(SCRIPT.resolve())],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
    )


def test_joint_run_needs_no_tokenizer(tmp_path: Path) -> None:
    """T4.4: a joint JEPA run must not require TOKENIZER."""
    root = _fixture_repo(tmp_path)

    result = _run_eval_sbatch(root, {})

    assert result.returncode == 0, result.stderr
    assert "PYTHON_ARGS scripts/run_eval.py" in result.stdout
    assert "--tokenizer" not in result.stdout
    assert "--run_id fake_run" in result.stdout


def test_tokenizer_is_passed_when_set(tmp_path: Path) -> None:
    root = _fixture_repo(tmp_path)
    tokenizer = root / "results" / "tok.pt"
    tokenizer.write_bytes(b"")

    result = _run_eval_sbatch(root, {"TOKENIZER": str(tokenizer)})

    assert result.returncode == 0, result.stderr
    assert "--tokenizer" in result.stdout
    assert str(tokenizer) in result.stdout


def test_missing_tokenizer_still_fails_loudly(tmp_path: Path) -> None:
    root = _fixture_repo(tmp_path)

    result = _run_eval_sbatch(root, {"TOKENIZER": "results/nope.pt"})

    assert result.returncode == 1
    assert "not found" in result.stderr
    assert "PYTHON_ARGS" not in result.stdout

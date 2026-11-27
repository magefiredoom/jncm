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
        if key not in {"TOKENIZER", "RUN_ID", "HEAD", "CKPT", "GPT2_MODEL_PATH", "GPT2_LARGE_PATH", "SLURM_SUBMIT_DIR", "SLURM_ARRAY_TASK_ID", "FINAL", "N_SAMPLES"}
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


def test_head_is_passed_to_run_eval(tmp_path: Path) -> None:
    """T5.7 M1: a staged flow or regress run is evaluated with --head."""
    root = _fixture_repo(tmp_path)

    result = _run_eval_sbatch(root, {"HEAD": "flow"})

    assert result.returncode == 0, result.stderr
    assert "--head flow" in result.stdout


def test_no_head_means_no_head_flag(tmp_path: Path) -> None:
    root = _fixture_repo(tmp_path)

    result = _run_eval_sbatch(root, {})

    assert result.returncode == 0, result.stderr
    assert "--head" not in result.stdout


def test_an_unknown_head_is_rejected_before_anything_runs(tmp_path: Path) -> None:
    root = _fixture_repo(tmp_path)

    result = _run_eval_sbatch(root, {"HEAD": "bogus"})

    assert result.returncode == 2
    assert "HEAD must be" in result.stderr
    assert "PYTHON_ARGS" not in result.stdout


# --------------------------------------------------------------------------- #
# GPT-2 path: GPT2_LARGE_PATH is the name the rest of the kit uses
# --------------------------------------------------------------------------- #


def _gpt2_arg(stdout: str) -> str:
    words = stdout.split()
    return words[words.index("--gpt2") + 1]


def test_gpt2_large_path_is_honoured(tmp_path: Path) -> None:
    """The 16 pilots were submitted with only GPT2_LARGE_PATH exported (RUNBOOK,
    run_eval.py), but eval.sbatch passed --gpt2 models/gpt2-large explicitly, a
    path that does not exist in the FIR checkout, which overrode it."""
    root = _fixture_repo(tmp_path)

    result = _run_eval_sbatch(root, {"GPT2_LARGE_PATH": "/home/u/jncm/models/gpt2-large"})

    assert result.returncode == 0, result.stderr
    assert _gpt2_arg(result.stdout) == "/home/u/jncm/models/gpt2-large"


def test_gpt2_model_path_wins_over_gpt2_large_path(tmp_path: Path) -> None:
    root = _fixture_repo(tmp_path)

    result = _run_eval_sbatch(
        root, {"GPT2_MODEL_PATH": "/a/model", "GPT2_LARGE_PATH": "/b/large"}
    )

    assert _gpt2_arg(result.stdout) == "/a/model"


def test_gpt2_path_defaults_to_the_repo_models_directory(tmp_path: Path) -> None:
    root = _fixture_repo(tmp_path)

    result = _run_eval_sbatch(root, {})

    assert _gpt2_arg(result.stdout) == "models/gpt2-large"


def test_ckpt_evaluates_one_checkpoint(tmp_path: Path) -> None:
    """T4.8: SPEC 10.4 evaluates the final milestone again with 1024 samples x 5 seeds;
    without CKPT every milestone would be scored at that size."""
    root = _fixture_repo(tmp_path)
    (root / "results" / "fake_run" / "ckpt" / "milestone_100.pt").write_bytes(b"")

    result = _run_eval_sbatch(root, {"CKPT": "results/fake_run/ckpt/milestone_100.pt"})

    assert result.returncode == 0, result.stderr
    assert "--ckpt results/fake_run/ckpt/milestone_100.pt" in result.stdout


def test_no_ckpt_means_no_ckpt_flag(tmp_path: Path) -> None:
    root = _fixture_repo(tmp_path)

    result = _run_eval_sbatch(root, {})

    assert result.returncode == 0, result.stderr
    assert "--ckpt" not in result.stdout


def test_a_missing_ckpt_is_rejected_before_anything_runs(tmp_path: Path) -> None:
    root = _fixture_repo(tmp_path)

    result = _run_eval_sbatch(root, {"CKPT": "results/fake_run/ckpt/milestone_999.pt"})

    assert result.returncode == 2
    assert "CKPT" in result.stderr
    assert "PYTHON_ARGS" not in result.stdout


def _final_env(root: Path, **extra: str) -> dict[str, str]:
    (root / "results" / "fake_run" / "ckpt" / "milestone_100.pt").write_bytes(b"")
    return {"FINAL": "1", "CKPT": "results/fake_run/ckpt/milestone_100.pt", **extra}


def test_final_passes_final_and_defaults_to_1024_samples(tmp_path: Path) -> None:
    """T4.16: FINAL=1 writes the SPEC 10.4 final eval into eval/final/ (run_eval --final)."""
    root = _fixture_repo(tmp_path)

    result = _run_eval_sbatch(root, _final_env(root))

    assert result.returncode == 0, result.stderr
    assert " --final" in result.stdout
    assert "--n_samples 1024" in result.stdout
    assert "--ckpt results/fake_run/ckpt/milestone_100.pt" in result.stdout


def test_final_keeps_an_explicit_n_samples(tmp_path: Path) -> None:
    root = _fixture_repo(tmp_path)

    result = _run_eval_sbatch(root, _final_env(root, N_SAMPLES="2048"))

    assert result.returncode == 0, result.stderr
    assert "--n_samples 2048" in result.stdout


def test_final_without_ckpt_is_rejected_before_anything_runs(tmp_path: Path) -> None:
    root = _fixture_repo(tmp_path)

    result = _run_eval_sbatch(root, {"FINAL": "1"})

    assert result.returncode == 2
    assert "FINAL" in result.stderr and "CKPT" in result.stderr
    assert "PYTHON_ARGS" not in result.stdout


@pytest.mark.parametrize("value", ["yes", "2", "true"])
def test_final_accepts_only_0_or_1(tmp_path: Path, value: str) -> None:
    root = _fixture_repo(tmp_path)

    result = _run_eval_sbatch(root, _final_env(root, FINAL=value))

    assert result.returncode == 2
    assert "FINAL" in result.stderr
    assert "PYTHON_ARGS" not in result.stdout


def test_without_final_no_final_flag_and_256_samples(tmp_path: Path) -> None:
    root = _fixture_repo(tmp_path)

    result = _run_eval_sbatch(root, {"FINAL": "0"})

    assert result.returncode == 0, result.stderr
    assert "--final" not in result.stdout
    assert "--n_samples 256" in result.stdout

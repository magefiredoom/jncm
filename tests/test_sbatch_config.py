"""T5.5b: optional CONFIG in the Slurm templates and --config in the kit."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest

TEMPLATES = {
    "scripts/slurm/tok.sbatch": "configs/tinystories_dev.yaml",
    "scripts/slurm/gen.sbatch": "configs/tinystories_dev.yaml",
    "scripts/slurm/jepa.sbatch": "configs/jepa_dev.yaml",
    "scripts/slurm/eval.sbatch": "configs/tinystories_dev.yaml",
}


def _working_bash() -> str | None:
    candidates = [shutil.which("bash"), shutil.which("bash.exe")]
    if os.name == "nt":
        candidates = [
            r"C:\Program Files\Git\bin\bash.exe",
            r"C:\Program Files\Git\usr\bin\bash.exe",
        ] + candidates
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


@pytest.mark.parametrize("script,default", TEMPLATES.items())
def test_templates_default_to_their_dev_config(script: str, default: str) -> None:
    text = Path(script).read_text(encoding="utf-8")

    assert f'CONFIG:-{default}' in text
    assert '--config "${config}"' in text


def test_templates_parse_with_bash() -> None:
    for script in TEMPLATES:
        result = subprocess.run(
            [BASH, "-n", script], capture_output=True, text=True
        )
        assert result.returncode == 0, (script, result.stderr)


def _eval_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "run_eval.py").write_text("", encoding="utf-8")
    (root / "configs").mkdir()
    (root / "configs" / "tinystories_lite.yaml").write_text("L: 128\n", encoding="utf-8")
    (root / "results" / "fake_run" / "ckpt").mkdir(parents=True)
    venv = root / ".venv" / "bin"
    venv.mkdir(parents=True)
    (venv / "activate").write_text(
        "python() { printf 'PYTHON_ARGS'; printf ' %s' \"$@\"; printf '\\n'; }\n",
        encoding="utf-8",
    )
    return root


def test_eval_sbatch_uses_the_config_environment_variable(tmp_path: Path) -> None:
    root = _eval_fixture(tmp_path)
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"TOKENIZER", "RUN_ID", "CONFIG", "SLURM_SUBMIT_DIR"}
    }
    env.update(
        {
            "SLURM_SUBMIT_DIR": str(root),
            "RUN_ID": "fake_run",
            "CONFIG": "configs/tinystories_lite.yaml",
        }
    )

    result = subprocess.run(
        [BASH, str(Path("scripts/slurm/eval.sbatch").resolve())],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--config configs/tinystories_lite.yaml" in result.stdout

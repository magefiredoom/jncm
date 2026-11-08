"""Checks for scripts/slurm/jepa.sbatch (T5.4).

The file is syntax-checked with a real bash and audited for the gen.sbatch
conventions: no account or partition in any #SBATCH line, the offline
variables, a venv, RUN_ID/STEPS/SETS from the environment, and a resume from
results/<RUN_ID>/ckpt/latest.pt.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest

SCRIPT = Path("scripts/slurm/jepa.sbatch")


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


def test_jepa_sbatch_parses_with_bash() -> None:
    result = subprocess.run(
        [BASH, "-n", str(SCRIPT)], capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr


def test_jepa_sbatch_follows_the_gen_sbatch_conventions() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    sbatch_lines = [
        line for line in text.splitlines() if line.startswith("#SBATCH")
    ]

    assert sbatch_lines, "no #SBATCH lines"
    assert not any("account" in line or "partition" in line for line in sbatch_lines)
    assert "set -euo pipefail" in text
    for variable in (
        "HF_HUB_OFFLINE=1",
        "HF_DATASETS_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
    ):
        assert variable in text, variable
    assert "TODO(placeholder)" in text
    assert ".venv/bin/activate" in text
    assert "TODO(placeholder)" in text


def test_jepa_sbatch_exposes_the_phase2_environment() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "results/${RUN_ID}/ckpt/latest.pt" in text
    assert "--resume" in text
    assert "configs/jepa_dev.yaml" in text
    assert "jncm.train_jepa" in text
    for variable in ("RUN_ID", "STEPS", "SETS", "TOKENIZER"):
        assert variable in text, variable
    assert "--set" in text

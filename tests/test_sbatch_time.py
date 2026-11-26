"""Every Slurm template carries a valid #SBATCH --time header.

On FIR a command-line --time does not rescue a template whose header is
invalid: sbatch rejects the header first ("Invalid --time specification").
chain.sh always passes --time=<wall>, so the header is only a default, but it
must parse.
"""

from __future__ import annotations

from pathlib import Path
import re

import pytest

SLURM_DIR = Path("scripts/slurm")
TEMPLATES = sorted(SLURM_DIR.glob("*.sbatch"))

# Slurm accepts more forms; we allow only [D-]HH:MM:SS so limits read the same everywhere.
VALID_TIME = re.compile(r"^(\d+-)?\d{1,2}:[0-5]\d:[0-5]\d$")


def test_all_templates_are_found() -> None:
    """The five training and eval templates, the CPU-only import probe (2026-11-20)
    and the CPU-only merge job (T4.9, 2026-11-25)."""
    names = {path.name for path in TEMPLATES}

    assert names == {
        "prep.sbatch",
        "tok.sbatch",
        "gen.sbatch",
        "eval.sbatch",
        "jepa.sbatch",
        "probe_import.sbatch",
        "merge.sbatch",
    }


@pytest.mark.parametrize("template", TEMPLATES, ids=lambda path: path.name)
def test_template_has_exactly_one_valid_time_header(template: Path) -> None:
    lines = template.read_text(encoding="utf-8").splitlines()
    values = [line.split("=", 1)[1].strip() for line in lines if line.startswith("#SBATCH --time=")]

    assert len(values) == 1, (template.name, values)
    assert VALID_TIME.match(values[0]), (template.name, values[0])


@pytest.mark.parametrize("template", TEMPLATES, ids=lambda path: path.name)
def test_template_has_no_wall_clock_placeholder(template: Path) -> None:
    text = template.read_text(encoding="utf-8")

    assert "HH:MM:SS" not in text, template.name

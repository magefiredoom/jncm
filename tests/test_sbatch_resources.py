"""Every Slurm template that starts a torch process declares its own CPU and memory.

FIR's defaults are 1 CPU and 256 MB per job (sacct: ReqMem 256M, AllocCPUS 1). On
2026-11-11 all sixteen pilot jobs, and the 2026-11-10 timing job 3234940, were
OOM-killed by Slurm after 0.5 to 5 minutes at about 230 MB RSS, which is the size of
`import torch`, because gen.sbatch, jepa.sbatch and eval.sbatch declared neither
`--mem` nor `--cpus-per-task`. Only tok.sbatch did (6 CPUs, 32000M), and only the
tokenizer jobs ever ran. The sbatch header is parsed before any command line, so the
values have to live in the templates.
"""

from __future__ import annotations

from pathlib import Path
import re

import pytest

# The ones that start python; prep.sbatch needs the internet and runs on a login node.
TORCH_TEMPLATES = [
    "scripts/slurm/tok.sbatch",
    "scripts/slurm/gen.sbatch",
    "scripts/slurm/jepa.sbatch",
    "scripts/slurm/eval.sbatch",
]

# Far above `import torch` plus a CUDA context (about 230 MB seen on FIR, then a
# few GB for the data windows and GPT-2); tok.sbatch's 32000M is the value that
# has actually run on FIR.
MIN_MEM_MB = 8000


def _directive(text: str, name: str) -> str | None:
    match = re.search(rf"^#SBATCH\s+--{name}=(\S+)\s*$", text, flags=re.MULTILINE)
    return match.group(1) if match else None


def _mem_mb(value: str) -> int:
    match = re.fullmatch(r"(\d+)([MG]?)", value)
    assert match, f"unparsable --mem value {value!r}"
    amount, unit = int(match.group(1)), match.group(2)
    return amount * 1024 if unit == "G" else amount


@pytest.mark.parametrize("script", TORCH_TEMPLATES)
def test_template_declares_memory(script: str) -> None:
    value = _directive(Path(script).read_text(encoding="utf-8"), "mem")

    assert value is not None, (
        f"{script} has no '#SBATCH --mem=': Slurm gives 256 MB and kills the job"
        " while torch imports"
    )
    assert _mem_mb(value) >= MIN_MEM_MB, (script, value)


@pytest.mark.parametrize("script", TORCH_TEMPLATES)
def test_template_declares_cpus(script: str) -> None:
    value = _directive(Path(script).read_text(encoding="utf-8"), "cpus-per-task")

    assert value is not None, f"{script} has no '#SBATCH --cpus-per-task='"
    assert int(value) >= 2, (script, value)


def test_every_template_that_requests_a_gpu_is_covered() -> None:
    """A new GPU template must be added to TORCH_TEMPLATES, not slip through."""
    gpu_templates = {
        str(path).replace("\\", "/")
        for path in Path("scripts/slurm").glob("*.sbatch")
        if "--gres=gpu" in path.read_text(encoding="utf-8")
    }

    assert gpu_templates <= set(TORCH_TEMPLATES), gpu_templates - set(TORCH_TEMPLATES)

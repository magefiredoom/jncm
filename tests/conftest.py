"""Shared pytest fixtures."""

from __future__ import annotations

import os
from pathlib import Path
import stat
from typing import Iterator

import pytest


@pytest.fixture(scope="session", autouse=True)
def lmod_stub(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Put a no-op ``module`` command on PATH for the whole session.

    The Slurm templates run ``module load python/3.12`` before activating the venv,
    like tok.sbatch always did. Lmod exists on the cluster only; the tests that run a
    template under bash with a fixture venv would otherwise stop at that line with
    "module: command not found".
    """
    stub_dir = tmp_path_factory.mktemp("lmod")
    stub = stub_dir / "module"
    stub.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8", newline="\n")
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("PATH", f"{stub_dir}{os.pathsep}{os.environ.get('PATH', '')}")
        yield stub_dir

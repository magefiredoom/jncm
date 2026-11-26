"""T4.10: the storage guard of chain.sh and launch_phase.sh (SPEC 7, v1.7).

A submission refuses a results root that resolves under a forbidden prefix
(`JNCM_FORBID_RESULTS_UNDER`, default `/home`): on FIR the 48 GiB home quota
filled on 2026-11-21 and killed two runs on a short `torch.save`. `results`
is a symlink to scratch there, so the check follows links (`realpath`). A dry
run prints the resolved results root and only warns.

Links: a real symlink on POSIX; on Windows a directory junction, which needs no
privilege and which Git Bash's `realpath` follows like a symlink. Both ends of
every link stay inside `tmp_path`.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest

CHAIN = "scripts/slurm/chain.sh"
LAUNCH = "scripts/launch_phase.sh"


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

LEAKED_VARS = ("SETS", "GEN_PPL", "N_SAMPLES", "N_SEEDS")


def _link_dir(target: Path, link: Path) -> None:
    """`link` -> `target` (a directory); a junction on Windows."""
    target.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        import _winapi

        _winapi.CreateJunction(str(target), str(link))
    else:
        os.symlink(target, link, target_is_directory=True)


def _repo(parent: Path) -> Path:
    """A repo-shaped tree (scripts/, configs/) without results/."""
    root = parent / "repo"
    ignore = shutil.ignore_patterns("__pycache__")
    shutil.copytree("scripts", root / "scripts", ignore=ignore)
    shutil.copytree("configs", root / "configs", ignore=ignore)
    return root


def _env(tmp_path: Path, forbid: Path | str | None) -> tuple[dict[str, str], Path]:
    """Environment with an sbatch stub on PATH; `forbid=None` leaves the default."""
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    log = tmp_path / "sbatch.log"
    stub = bindir / "sbatch"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        "touch \"$FAKE_SBATCH_LOG\"\n"
        "echo \"$*\" >> \"$FAKE_SBATCH_LOG\"\n"
        "n=$(wc -l < \"$FAKE_SBATCH_LOG\")\n"
        "echo \"Submitted batch job $((1000 + n))\"\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"SBATCH_ACCOUNT", "JNCM_FORBID_RESULTS_UNDER", *LEAKED_VARS}
    }
    env["SBATCH_ACCOUNT"] = "def-test"
    env["FAKE_SBATCH_LOG"] = str(log)
    env["PATH"] = str(bindir) + os.pathsep + env.get("PATH", "")
    if forbid is not None:
        env["JNCM_FORBID_RESULTS_UNDER"] = str(forbid)
    return env, log


def _calls(log: Path) -> list[str]:
    return log.read_text(encoding="utf-8").splitlines() if log.is_file() else []


def _chain(root: Path, args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [BASH, str(root / CHAIN)] + args,
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
    )


def _launch(root: Path, args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [BASH, str(root / LAUNCH)] + args,
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
    )


EVAL_ARGS = ["scripts/slurm/eval.sbatch", "1", "--export=ALL,RUN_ID=gen_x", "--array=0-4"]


def test_chain_refuses_a_results_link_into_the_forbidden_prefix(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _link_dir(tmp_path / "home" / "u" / "real", root / "results")
    env, log = _env(tmp_path, tmp_path / "home")

    result = _chain(root, EVAL_ARGS + ["--merge-after", "--submit"], env)

    assert result.returncode == 2, (result.stdout, result.stderr)
    assert "SPEC 7" in result.stderr
    assert "JNCM_FORBID_RESULTS_UNDER" in result.stderr
    assert "/home/u/real" in result.stderr  # the resolved root, not the link
    assert _calls(log) == []  # neither the eval nor its merge was submitted


def test_chain_refuses_a_plain_results_directory_under_the_prefix(tmp_path: Path) -> None:
    root = _repo(tmp_path / "home" / "u")
    (root / "results").mkdir()
    env, log = _env(tmp_path, tmp_path / "home")

    result = _chain(root, EVAL_ARGS + ["--submit"], env)

    assert result.returncode == 2
    assert "SPEC 7" in result.stderr
    assert _calls(log) == []


def test_chain_submits_through_a_results_link_to_scratch(tmp_path: Path) -> None:
    """The FIR layout: results is a link from the repo to scratch."""
    root = _repo(tmp_path)
    _link_dir(tmp_path / "scratch" / "res", root / "results")
    env, log = _env(tmp_path, tmp_path / "home")

    result = _chain(root, EVAL_ARGS + ["--merge-after", "--submit"], env)

    assert result.returncode == 0, result.stderr
    assert len(_calls(log)) == 2


def test_chain_prefix_check_respects_path_boundaries(tmp_path: Path) -> None:
    """Forbidding <tmp>/home must not catch <tmp>/homework."""
    root = _repo(tmp_path)
    _link_dir(tmp_path / "homework" / "res", root / "results")
    env, log = _env(tmp_path, tmp_path / "home")

    result = _chain(root, EVAL_ARGS + ["--submit"], env)

    assert result.returncode == 0, result.stderr
    assert len(_calls(log)) == 1


def test_chain_dry_run_prints_the_resolved_root_and_only_warns(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _link_dir(tmp_path / "home" / "u" / "real", root / "results")
    env, log = _env(tmp_path, tmp_path / "home")

    result = _chain(root, EVAL_ARGS, env)

    assert result.returncode == 0, result.stderr
    dry = next(line for line in result.stdout.splitlines() if line.startswith("DRY-RUN"))
    assert "results root" in dry and "/home/u/real" in dry
    assert "WARNING" in result.stderr and "--submit would be refused" in result.stderr
    assert _calls(log) == []


def test_chain_dry_run_names_the_default_prefix(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _link_dir(tmp_path / "scratch" / "res", root / "results")
    env, _ = _env(tmp_path, None)

    result = _chain(root, EVAL_ARGS, env)

    assert result.returncode == 0, result.stderr
    dry = next(line for line in result.stdout.splitlines() if line.startswith("DRY-RUN"))
    assert "/scratch/res" in dry
    assert "refused under /home" in dry


def test_launch_refuses_before_submitting_anything(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _link_dir(tmp_path / "home" / "u" / "real", root / "results")
    env, log = _env(tmp_path, tmp_path / "home")

    result = _launch(root, ["g0", "pilot", "--submit"], env)

    assert result.returncode == 2, (result.stdout, result.stderr)
    assert "SPEC 7" in result.stderr
    assert "/home/u/real" in result.stderr
    assert _calls(log) == []


def test_launch_dry_run_prints_the_resolved_results_root(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _link_dir(tmp_path / "scratch" / "res", root / "results")
    env, log = _env(tmp_path, tmp_path / "home")

    result = _launch(root, ["g0", "pilot"], env)

    assert result.returncode == 0, result.stderr
    line = next(
        line for line in result.stdout.splitlines() if line.startswith("results root:")
    )
    assert "/scratch/res" in line
    assert "JNCM_FORBID_RESULTS_UNDER" in line
    assert _calls(log) == []


def test_launch_submits_through_a_results_link_to_scratch(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _link_dir(tmp_path / "scratch" / "res", root / "results")
    env, log = _env(tmp_path, tmp_path / "home")

    result = _launch(root, ["g0", "pilot", "--submit"], env)

    assert result.returncode == 0, result.stderr
    assert len(_calls(log)) == 4  # tok, gen, eval, merge

"""Tracked files never carry the cluster username or allocation account (CLAUDE.md).

The strings are assembled from pieces so that this file does not contain them.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HOOK = ROOT / "scripts" / "hooks" / "pre-commit"
USER = "aef" + "ad"
ACCOUNT = "def-" + "cheng"


def _in_git_checkout() -> bool:
    if shutil.which("git") is None:
        return False
    out = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return out.returncode == 0 and out.stdout.strip() == "true"


def _sh() -> str | None:
    return shutil.which("sh") or shutil.which("bash")


def test_no_private_ids_in_tracked_files() -> None:
    if not _in_git_checkout():
        pytest.skip("not a git checkout")
    out = subprocess.run(
        ["git", "grep", "-nIi", "-e", USER, "-e", ACCOUNT],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert out.returncode == 1, out.stdout[:2000]
    assert out.stdout == ""


def test_site_env_is_ignored_and_example_is_tracked() -> None:
    if not _in_git_checkout():
        pytest.skip("not a git checkout")
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", "site.env"], cwd=ROOT
    )
    assert ignored.returncode == 0
    example = (ROOT / "site.env.example").read_text(encoding="utf-8")
    assert "export FIR_USER=<your-username>" in example
    assert "export SBATCH_ACCOUNT=<your-allocation>" in example


def _hook_result(repo: Path, content: str, name: str = "a.txt") -> int:
    (repo / name).write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", name], cwd=repo, check=True)
    sh = _sh()
    assert sh is not None
    return subprocess.run([sh, str(HOOK)], cwd=repo, capture_output=True).returncode


@pytest.mark.skipif(_sh() is None or shutil.which("git") is None, reason="needs sh and git")
def test_pre_commit_hook_rejects_ids_and_accepts_clean_changes(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    assert _hook_result(repo, "ssh fir 'squeue -u $USER'\n") == 0
    assert _hook_result(repo, f"ssh {USER}@fir.example 'ls'\n") == 1
    assert _hook_result(repo, f"--account={ACCOUNT}23_gpu\n") == 1
    assert _hook_result(repo, f"{USER.upper()} wrote this\n") == 1
    assert _hook_result(repo, "clean again\n") == 0
    # A file name counts too.
    assert _hook_result(repo, "x\n", name=f"notes_{USER}.txt") == 1


@pytest.mark.skipif(_sh() is None or shutil.which("git") is None, reason="needs sh and git")
def test_pre_commit_hook_reads_site_env_values(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / "site.env").write_text(
        "export FIR_USER=zq-private-name\nexport SBATCH_ACCOUNT=<your-allocation>\n",
        encoding="utf-8",
    )
    assert _hook_result(repo, "path /home/zq-private-name/x\n") == 1
    # A placeholder value is not treated as a secret.
    assert _hook_result(repo, "<your-allocation> stays a placeholder\n") == 0

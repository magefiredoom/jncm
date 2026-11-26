"""T4.5: scripts/slurm/chain.sh (dry-run, submission, dependencies, account)."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest

CHAIN = Path("scripts/slurm/chain.sh")
LAUNCH = Path("scripts/launch_phase.sh")


def _working_bash() -> str | None:
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


def _clean_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"SBATCH_ACCOUNT", "SETS"}
    }
    env["SBATCH_ACCOUNT"] = "def-test"
    env.update(extra or {})
    return env


def _run(args: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [BASH, str(CHAIN)] + args,
        env=env or _clean_env(),
        capture_output=True,
        text=True,
    )


def _fake_sbatch(tmp_path: Path) -> tuple[dict[str, str], Path]:
    """A real sbatch stub on PATH; it records its argv and returns a job id."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
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
    env = _clean_env(
        {
            "FAKE_SBATCH_LOG": str(log),
            "PATH": str(bindir) + os.pathsep + os.environ.get("PATH", ""),
        }
    )
    return env, log


def _submit_commands(args: list[str], tmp_path: Path) -> tuple[subprocess.CompletedProcess, list[str]]:
    """Run chain.sh --submit against the sbatch stub and return the calls."""
    env, log = _fake_sbatch(tmp_path)
    result = subprocess.run(
        [BASH, str(CHAIN)] + args + ["--submit"],
        env=env,
        capture_output=True,
        text=True,
    )
    calls = log.read_text(encoding="utf-8").splitlines() if log.is_file() else []
    return result, calls


def test_scripts_parse_with_bash() -> None:
    for script in (CHAIN, LAUNCH):
        result = subprocess.run(
            [BASH, "-n", str(script)], capture_output=True, text=True
        )
        assert result.returncode == 0, (script, result.stderr)


def test_refuses_without_sbatch_account() -> None:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"SBATCH_ACCOUNT", "SETS"}
    }

    result = _run([str(Path("scripts/slurm/tok.sbatch")), "1"], env)

    assert result.returncode == 2
    assert "SBATCH_ACCOUNT" in result.stderr


def test_dry_run_prints_the_chained_commands() -> None:
    result = _run(
        [
            "scripts/slurm/tok.sbatch",
            "3",
            "--export=ALL,RUN_ID=tok_x,STEPS=10",
            "--wall=02:00:00",
        ]
    )

    assert result.returncode == 0, result.stderr
    commands = [line for line in result.stdout.splitlines() if line.startswith("sbatch")]
    assert len(commands) == 3
    assert "DRY-RUN" in result.stdout
    assert all("--time=02:00:00" in line for line in commands)
    assert all("--export=ALL,RUN_ID=tok_x,STEPS=10" in line for line in commands)
    assert "--dependency" not in commands[0]
    assert "--dependency=afterany:<JOB_1>" in commands[1]
    assert "--dependency=afterany:<JOB_2>" in commands[2]


def test_submit_chains_with_afterany_and_prints_job_ids(tmp_path: Path) -> None:
    result, calls = _submit_commands(
        ["scripts/slurm/gen.sbatch", "2", "--export=ALL,RUN_ID=gen_x"], tmp_path
    )

    assert result.returncode == 0, result.stderr
    assert "Submitted batch job 1001 (chunk 1/2" in result.stdout
    assert "Submitted batch job 1002 (chunk 2/2" in result.stdout
    assert "CHAIN_JOB_IDS=1001 1002" in result.stdout
    assert "CHAIN_LAST_ID=1002" in result.stdout
    assert len(calls) == 2
    assert "--dependency" not in calls[0]
    assert "--dependency=afterany:1001" in calls[1]
    assert calls[0].endswith("scripts/slurm/gen.sbatch")


def test_external_dependency_is_kept_on_the_first_chunk(tmp_path: Path) -> None:
    result, calls = _submit_commands(
        [
            "scripts/slurm/tok.sbatch",
            "2",
            "--export=ALL,RUN_ID=tok_x",
            "--dependency=afterany:900",
        ],
        tmp_path,
    )

    assert result.returncode == 0, result.stderr
    assert "--dependency=afterany:900" in calls[0]
    assert "--dependency=afterany:900,afterany:1001" in calls[1]


def test_array_and_extra_sbatch_arguments_are_forwarded() -> None:
    result = _run(
        [
            "scripts/slurm/eval.sbatch",
            "1",
            "--export=ALL,RUN_ID=gen_x",
            "--array=0-0",
            "--gres=gpu:h100:1",
        ]
    )

    assert result.returncode == 0, result.stderr
    command = next(line for line in result.stdout.splitlines() if line.startswith("sbatch"))
    assert "--array=0-0" in command
    assert "--gres=gpu:h100:1" in command
    assert "--account=def-test" in command


@pytest.mark.parametrize("chunks", ["0", "-1", "two"])
def test_rejects_a_bad_chunk_count(chunks: str) -> None:
    result = _run(["scripts/slurm/tok.sbatch", chunks])

    assert result.returncode == 2
    assert "n_chunks" in result.stderr


def test_rejects_a_missing_template() -> None:
    result = _run(["scripts/slurm/nope.sbatch", "1"])

    assert result.returncode == 2
    assert "not found" in result.stderr


def test_sets_travels_inside_the_export_list_for_the_f16_sweep() -> None:
    """The F16 corruption runs go through chain.sh with SETS in --export, not as
    an exported shell variable (launch_phase.sh refuses leaked SETS, T5.7 m6)."""
    export = (
        "--export=ALL,HEAD=ce,RUN_ID=corr_p080,STEPS=2000,GEN_PPL=1,"
        "TOKENIZER=results/tok_vq_10k/ckpt/latest.pt,"
        "SETS=generator.corrupt.p_first=0.8"
    )

    result = _run(["scripts/slurm/gen.sbatch", "1", export, "--wall=00:45:00"])

    assert result.returncode == 0, result.stderr
    commands = [line for line in result.stdout.splitlines() if line.startswith("sbatch")]
    assert len(commands) == 1 and export in commands[0]


def test_usage_documents_the_wall_form_the_parser_accepts() -> None:
    """Only `--wall=HH:MM:SS` parses (as in launch_phase.sh); the usage said otherwise."""
    text = Path("scripts/slurm/chain.sh").read_text(encoding="utf-8")

    assert text.count("[--wall=HH:MM:SS]") == 2
    assert "--wall HH:MM:SS" not in text


# T4.9: --merge-after submits merge.sbatch afterok on the whole eval array.


def _merge_lines(stdout: str) -> list[str]:
    return [
        line
        for line in stdout.splitlines()
        if line.startswith("sbatch") and "merge.sbatch" in line
    ]


def test_merge_after_dry_run_prints_a_cpu_merge_afterok_on_the_last_chunk() -> None:
    result = _run(
        [
            "scripts/slurm/eval.sbatch",
            "1",
            "--export=ALL,RUN_ID=gen_x,HEAD=ce",
            "--array=0-4",
            "--dependency=afterany:900",
            "--merge-after",
        ]
    )

    assert result.returncode == 0, result.stderr
    commands = [line for line in result.stdout.splitlines() if line.startswith("sbatch")]
    merges = _merge_lines(result.stdout)
    assert len(commands) == 2 and len(merges) == 1
    assert commands[1] == merges[0]
    merge = merges[0]
    assert "--dependency=afterok:<JOB_1>" in merge
    assert "afterany" not in merge
    assert "--export=ALL,RUN_ID=gen_x,HEAD=ce" in merge
    assert "--account=def-test" in merge
    for flag in ("--gres", "--time", "--array"):
        assert flag not in merge, flag
    assert merge.endswith("scripts/slurm/merge.sbatch")
    assert "scripts/slurm/eval.sbatch" in commands[0]


def test_merge_after_waits_for_the_last_chunk_of_a_longer_chain() -> None:
    result = _run(
        ["scripts/slurm/eval.sbatch", "3", "--export=ALL,RUN_ID=gen_x", "--merge-after"]
    )

    assert result.returncode == 0, result.stderr
    merges = _merge_lines(result.stdout)
    assert len(merges) == 1
    assert "--dependency=afterok:<JOB_3>" in merges[0]


def test_merge_after_submit_keeps_chain_last_id_and_prints_the_merge_id(
    tmp_path: Path,
) -> None:
    result, calls = _submit_commands(
        [
            "scripts/slurm/eval.sbatch",
            "1",
            "--export=ALL,RUN_ID=gen_x",
            "--array=0-4",
            "--merge-after",
        ],
        tmp_path,
    )

    assert result.returncode == 0, result.stderr
    assert len(calls) == 2
    assert calls[0].endswith("scripts/slurm/eval.sbatch")
    assert "--array=0-4" in calls[0]
    assert calls[1].endswith("scripts/slurm/merge.sbatch")
    assert "--dependency=afterok:1001" in calls[1]
    assert "--export=ALL,RUN_ID=gen_x" in calls[1]
    for flag in ("--gres", "--time", "--array"):
        assert flag not in calls[1], flag
    assert "CHAIN_LAST_ID=1001" in result.stdout
    assert "MERGE_JOB_ID=1002" in result.stdout


def test_merge_after_without_run_id_in_the_export_is_refused() -> None:
    result = _run(["scripts/slurm/eval.sbatch", "1", "--export=ALL,HEAD=ce", "--merge-after"])

    assert result.returncode == 2
    assert "RUN_ID" in result.stderr
    assert not [line for line in result.stdout.splitlines() if line.startswith("sbatch")]


def test_without_merge_after_no_merge_is_planned() -> None:
    result = _run(["scripts/slurm/eval.sbatch", "1", "--export=ALL,RUN_ID=gen_x"])

    assert result.returncode == 0, result.stderr
    assert not _merge_lines(result.stdout)
    assert "MERGE_JOB_ID" not in result.stdout

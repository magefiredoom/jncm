"""T4.5: scripts/launch_phase.sh (pilots, full chunking, ablations)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest

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


def _fixture_repo(tmp_path: Path) -> Path:
    """Repo-shaped tree with its own scripts/ and results/."""
    root = tmp_path / "repo"
    shutil.copytree(
        "scripts", root / "scripts", ignore=shutil.ignore_patterns("__pycache__")
    )
    (root / "results").mkdir()
    return root


def _seed_pilot(root: Path, pilot_id: str, sec_per_step: float) -> None:
    directory = root / "results" / pilot_id
    directory.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({"step": 1, "sec_per_step": sec_per_step / 2}),
        json.dumps({"step": 2, "sec_per_step": sec_per_step}),
    ]
    (directory / "metrics.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _clean_env() -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"SBATCH_ACCOUNT", "SETS"}
    }
    env["SBATCH_ACCOUNT"] = "def-test"
    return env


def _run(
    root: Path, args: list[str], env: dict[str, str] | None = None
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [BASH, str(root / LAUNCH)] + args,
        cwd=str(root),
        env=env or _clean_env(),
        capture_output=True,
        text=True,
    )


def _sbatch_lines(stdout: str) -> list[str]:
    return [line for line in stdout.splitlines() if line.startswith("sbatch")]


def test_g0_pilot_uses_pilot_ids_steps_and_eval_settings(tmp_path: Path) -> None:
    root = _fixture_repo(tmp_path)

    result = _run(root, ["g0", "pilot"])

    assert result.returncode == 0, result.stderr
    out = result.stdout
    commands = _sbatch_lines(out)
    assert len(commands) == 3  # tokenizer, generator, eval
    assert "pilot_tok_vq" in commands[0]
    assert "STEPS=500" in commands[0]
    assert "--time=00:20:00" in commands[0]
    assert "pilot_gen_vq_ce" in commands[1]
    assert "STEPS=300" in commands[1]
    assert "GEN_PPL=1" in commands[1]
    assert "scripts/slurm/eval.sbatch" in commands[2]
    assert "N_SAMPLES=32,N_SEEDS=1" in commands[2]
    assert "--array=0-0" in commands[2]
    assert "--dependency=afterany:<LAST>" in commands[2]
    assert "tok_vq_10k" not in out
    assert "gen_vq_ce_dev" not in out


def test_g1_pilot_runs_flow_and_regress_in_parallel(tmp_path: Path) -> None:
    root = _fixture_repo(tmp_path)

    result = _run(root, ["g1", "pilot"])

    assert result.returncode == 0, result.stderr
    out = result.stdout
    gens = [line for line in _sbatch_lines(out) if "gen.sbatch" in line]
    evals = [line for line in _sbatch_lines(out) if "eval.sbatch" in line]
    assert len(gens) == 2 and len(evals) == 2
    assert any("pilot_gen_noisy_flow" in line for line in gens)
    assert any("pilot_gen_noisy_regress" in line for line in gens)
    assert all("--dependency=afterany:<LAST>" in line for line in gens)
    assert "gen_noisy_flow_dev" not in out
    assert "tok_noisy_10k" not in out


def test_config_is_forwarded_to_every_template(tmp_path: Path) -> None:
    """T5.5b: --config reaches tok, gen and eval through CONFIG."""
    root = _fixture_repo(tmp_path)

    result = _run(root, ["g0", "pilot", "--config", "configs/tinystories_lite.yaml"])

    assert result.returncode == 0, result.stderr
    commands = _sbatch_lines(result.stdout)
    assert len(commands) == 3
    assert all(
        "CONFIG=configs/tinystories_lite.yaml" in line for line in commands
    )


def test_g2_forwards_the_jepa_config(tmp_path: Path) -> None:
    root = _fixture_repo(tmp_path)

    result = _run(root, ["g2", "pilot", "--config=configs/jepa_lite.yaml"])

    assert result.returncode == 0, result.stderr
    jepa = [line for line in _sbatch_lines(result.stdout) if "jepa.sbatch" in line]
    assert jepa
    assert all("CONFIG=configs/jepa_lite.yaml" in line for line in jepa)


@pytest.mark.parametrize(
    "phase,config,expected",
    [
        ("g0", None, {"pilot_tok_vq", "pilot_gen_vq_ce"}),
        (
            "g1",
            None,
            {"pilot_tok_noisy", "pilot_gen_noisy_flow", "pilot_gen_noisy_regress"},
        ),
        ("g2", None, {"pilot_jepa_base"}),
        (
            "g0",
            "configs/tinystories_lite.yaml",
            {"pilot_tok_vq_lite", "pilot_gen_vq_ce_lite"},
        ),
        (
            "g1",
            "configs/tinystories_lite.yaml",
            {
                "pilot_tok_noisy_lite",
                "pilot_gen_noisy_flow_lite",
                "pilot_gen_noisy_regress_lite",
            },
        ),
        ("g2", "configs/jepa_lite.yaml", {"pilot_jepa_base_lite"}),
    ],
)
def test_dry_run_run_ids_follow_the_config(
    tmp_path: Path, phase: str, config: str | None, expected: set[str]
) -> None:
    """T5.5c: a non-dev config suffixes every run id, so DONE files never mix."""
    root = _fixture_repo(tmp_path)
    args = [phase, "pilot"] + (["--config", config] if config else [])

    result = _run(root, args)

    assert result.returncode == 0, result.stderr
    ids = set(re.findall(r"RUN_ID=([^,\s]+)", result.stdout))
    assert ids == expected, ids
    if config:
        assert "run-id suffix lite" in result.stdout
        assert not any(run.endswith("_dev") for run in ids), ids
        assert "_lite/ckpt/latest.pt" in result.stdout
    else:
        assert "run-id suffix none" in result.stdout
        assert not any("_lite" in line for line in _sbatch_lines(result.stdout))


def test_lite_full_run_uses_the_lite_pilot_record(tmp_path: Path) -> None:
    """T5.5c: the chunk count comes from the suffixed pilot, not the dev one."""
    root = _fixture_repo(tmp_path)
    _seed_pilot(root, "pilot_tok_vq", 8.0)
    _seed_pilot(root, "pilot_gen_vq_ce", 8.0)
    _seed_pilot(root, "pilot_tok_vq_lite", 1.0)
    _seed_pilot(root, "pilot_gen_vq_ce_lite", 1.0)

    result = _run(root, ["g0", "full", "--config", "configs/tinystories_lite.yaml"])

    assert result.returncode == 0, result.stderr
    tok = [line for line in _sbatch_lines(result.stdout) if "tok.sbatch" in line]
    gen = [line for line in _sbatch_lines(result.stdout) if "gen.sbatch" in line]
    # ceil(10000 * 1.0 / (0.9 * 10800)) = 2 chunks, from the _lite pilots.
    assert len(tok) == 2 and len(gen) == 2
    assert "(pilot pilot_tok_vq_lite)" in result.stderr
    assert "tok_vq_10k_lite" in tok[0]
    assert "gen_vq_ce_lite" in gen[0]
    assert "TOKENIZER=results/tok_vq_10k_lite/ckpt/latest.pt" in gen[0]
    assert "_dev" not in "".join(tok + gen)


def test_g2_pilot_lists_ablations_without_submitting_them(tmp_path: Path) -> None:
    root = _fixture_repo(tmp_path)

    result = _run(root, ["g2", "pilot"])

    assert result.returncode == 0, result.stderr
    out = result.stdout
    commands = _sbatch_lines(out)
    assert any("pilot_jepa_base" in line for line in commands)
    assert not any("pilot_jepa_no_vicreg" in line for line in commands)
    listed = [line for line in out.splitlines() if line.startswith("LIST ")]
    assert len(listed) == 6
    assert "pilot_jepa_no_vicreg" in out
    assert "--ablations" in out
    assert "WARNING" in result.stderr  # the pilot tokenizer checkpoint is absent
    assert "jepa_base_dev" not in out


def test_g2_pilot_with_ablations_submits_seven_arms(tmp_path: Path) -> None:
    root = _fixture_repo(tmp_path)

    result = _run(root, ["g2", "pilot", "--ablations"])

    assert result.returncode == 0, result.stderr
    out = result.stdout
    jepa = [line for line in _sbatch_lines(out) if "jepa.sbatch" in line]
    evals = [line for line in _sbatch_lines(out) if "eval.sbatch" in line]
    assert len(jepa) == 7 and len(evals) == 7
    assert 'SETS="jepa.vicreg=false"' in out
    assert 'SETS="jepa.vicreg=false jepa.decoder_through_E=false"' in out
    assert "pilot_jepa_regress" in out
    assert "STEPS=200" in jepa[0]


def test_full_g0_chunks_from_the_pilot_sec_per_step(tmp_path: Path) -> None:
    root = _fixture_repo(tmp_path)
    _seed_pilot(root, "pilot_tok_vq", 4.0)
    _seed_pilot(root, "pilot_gen_vq_ce", 4.0)

    result = _run(root, ["g0", "full"])

    assert result.returncode == 0, result.stderr
    # ceil(10000 * 4.0 / (0.9 * 10800)) = ceil(4.115...) = 5
    tok = [line for line in _sbatch_lines(result.stdout) if "tok.sbatch" in line]
    gen = [line for line in _sbatch_lines(result.stdout) if "gen.sbatch" in line]
    assert len(tok) == 5 and len(gen) == 5
    assert "5 chunk(s) (pilot pilot_tok_vq)" in result.stderr
    assert "--dependency=afterany:<JOB_4>" in tok[4]
    assert "tok_vq_10k" in tok[0] and "STEPS=10000" in tok[0]
    assert "gen_vq_ce_dev" in gen[0] and "GEN_PPL" not in gen[0]


def test_full_g0_uses_sec_per_step_flag_when_no_pilot_exists(tmp_path: Path) -> None:
    root = _fixture_repo(tmp_path)

    result = _run(root, ["g0", "full", "--sec_per_step=1.0"])

    assert result.returncode == 0, result.stderr
    tok = [line for line in _sbatch_lines(result.stdout) if "tok.sbatch" in line]
    # ceil(10000 * 1.0 / (0.9 * 10800)) = 2
    assert len(tok) == 2
    assert "(--sec_per_step)" in result.stderr


def test_full_without_pilot_or_sec_per_step_errors(tmp_path: Path) -> None:
    root = _fixture_repo(tmp_path)

    result = _run(root, ["g0", "full"])

    assert result.returncode == 1
    assert "--sec_per_step" in result.stderr
    assert not _sbatch_lines(result.stdout)


def test_full_g1_skips_a_finished_tokenizer(tmp_path: Path) -> None:
    root = _fixture_repo(tmp_path)
    (root / "results" / "tok_noisy_10k").mkdir(parents=True)
    (root / "results" / "tok_noisy_10k" / "DONE").write_text("{}\n", encoding="utf-8")
    _seed_pilot(root, "pilot_gen_noisy_flow", 1.0)
    _seed_pilot(root, "pilot_gen_noisy_regress", 1.0)

    result = _run(root, ["g1", "full"])

    assert result.returncode == 0, result.stderr
    assert "SKIP tok_noisy_10k" in result.stdout
    tok = [line for line in _sbatch_lines(result.stdout) if "tok.sbatch" in line]
    gen = [line for line in _sbatch_lines(result.stdout) if "gen.sbatch" in line]
    assert not tok
    assert len(gen) == 4  # 2 chunks x 2 generator arms
    # No tokenizer chain, so the generators wait for nothing upstream.
    assert not any("--dependency=afterany:<LAST>" in line for line in gen)


def test_submit_chains_training_then_eval(tmp_path: Path) -> None:
    root = _fixture_repo(tmp_path)
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
    env = _clean_env()
    env["FAKE_SBATCH_LOG"] = str(log)
    env["PATH"] = str(bindir) + os.pathsep + os.environ.get("PATH", "")

    result = _run(root, ["g0", "pilot", "--submit"], env)

    assert result.returncode == 0, result.stderr
    calls = log.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 3
    assert "--dependency" not in calls[0]
    assert "--dependency=afterany:1001" in calls[1]
    assert "--dependency=afterany:afterany" not in result.stdout
    assert "--dependency=afterany:1002" in calls[2]
    assert "CHAIN_LAST_ID=1003" in result.stdout


def test_requires_the_sbatch_account(tmp_path: Path) -> None:
    root = _fixture_repo(tmp_path)
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"SBATCH_ACCOUNT", "SETS"}
    }

    result = subprocess.run(
        [BASH, str(root / LAUNCH), "g0", "pilot"],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "SBATCH_ACCOUNT" in result.stderr


@pytest.mark.parametrize(
    "args", [["g9", "pilot"], ["g0", "huge"], ["g0"], ["g0", "pilot", "--wall=20:00"]]
)
def test_rejects_bad_arguments(tmp_path: Path, args: list[str]) -> None:
    root = _fixture_repo(tmp_path)

    result = _run(root, args)

    assert result.returncode == 2
    assert "ERROR" in result.stderr

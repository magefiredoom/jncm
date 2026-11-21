"""T4.5: scripts/launch_phase.sh (pilots, full chunking, ablations)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest
import yaml

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
    ignore = shutil.ignore_patterns("__pycache__")
    shutil.copytree("scripts", root / "scripts", ignore=ignore)
    shutil.copytree("configs", root / "configs", ignore=ignore)
    (root / "results").mkdir()
    return root


def _done(root: Path, run_id: str) -> None:
    directory = root / "results" / run_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "DONE").write_text("{}\n", encoding="utf-8")


def _seed_pilot(
    root: Path,
    pilot_id: str,
    sec_per_step: float,
    train_sec_per_step: float | None = None,
) -> None:
    directory = root / "results" / pilot_id
    directory.mkdir(parents=True, exist_ok=True)
    last = {"step": 2, "sec_per_step": sec_per_step}
    if train_sec_per_step is not None:
        last["train_sec_per_step"] = train_sec_per_step
    lines = [
        json.dumps({"step": 1, "sec_per_step": sec_per_step / 2}),
        json.dumps(last),
    ]
    (directory / "metrics.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


# T5.7 m6: anything upstream of chain.sh that exports these four leaks into
# every submitted job through --export=ALL, so tests must drop them here.
LEAKED_VARS = ("SETS", "GEN_PPL", "N_SAMPLES", "N_SEEDS")


def _clean_env() -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"SBATCH_ACCOUNT", *LEAKED_VARS}
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
            {"pilot_tok_vq", "pilot_gen_vq_ce_lite"},
        ),
        (
            "g1",
            "configs/tinystories_lite.yaml",
            {
                "pilot_tok_noisy",
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
        assert "/ckpt/latest.pt" in result.stdout
        # T5.6: tokenizer ids never carry the config suffix.
        assert not any("_lite" in run for run in ids if run.startswith("tok_")), ids
    else:
        assert "run-id suffix none" in result.stdout
        assert not any("_lite" in line for line in _sbatch_lines(result.stdout))


def test_lite_full_run_uses_the_lite_pilot_record_and_shared_tokenizer(
    tmp_path: Path,
) -> None:
    """T5.5c/T5.6: generator chunks from the _lite pilot; tokenizer is shared."""
    root = _fixture_repo(tmp_path)
    _seed_pilot(root, "pilot_tok_vq", 1.0)
    _seed_pilot(root, "pilot_tok_vq_lite", 1.5)
    _seed_pilot(root, "pilot_gen_vq_ce", 8.0)
    _seed_pilot(root, "pilot_gen_vq_ce_lite", 1.0)

    result = _run(root, ["g0", "full", "--config", "configs/tinystories_lite.yaml"])

    assert result.returncode == 0, result.stderr
    tok = [line for line in _sbatch_lines(result.stdout) if "tok.sbatch" in line]
    gen = [line for line in _sbatch_lines(result.stdout) if "gen.sbatch" in line]
    # T5.7 B1 (whole resume intervals, usable 9720 s per chunk): tokenizer from
    # the shared dev pilot, 5000 steps x 1.0 s = 5000 s per interval -> 1
    # interval per chunk -> 2 chunks; generator from the _lite pilot, 1000
    # steps x 1.0 s -> 9 intervals per chunk -> ceil(10000 / 9000) = 2 chunks.
    # The dev generator pilot (8.0 s/step, 8000 s per interval would give 12
    # chunks) must not be used for the lite run.
    assert len(tok) == 2 and len(gen) == 2
    assert "(pilot pilot_tok_vq)" in result.stderr
    assert "(pilot pilot_gen_vq_ce_lite)" in result.stderr
    assert "RUN_ID=tok_vq_10k" in tok[0]
    assert "gen_vq_ce_lite" in gen[0]
    assert "TOKENIZER=results/tok_vq_10k/ckpt/latest.pt" in gen[0]
    assert "tok_vq_10k_lite" not in result.stdout
    assert "_dev/ckpt" not in result.stdout


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
    _seed_pilot(root, "pilot_tok_vq", 1.0)
    _seed_pilot(root, "pilot_gen_vq_ce", 2.5)

    result = _run(root, ["g0", "full"])

    assert result.returncode == 0, result.stderr
    # T5.7 B1: a chunk advances in whole resume intervals. usable = 0.9 * 10800 =
    # 9720 s. Tokenizer (resume every 5000 steps): 5000 s per interval -> 1
    # interval = 5000 steps per chunk -> 2 chunks. Generator (resume every 10% =
    # 1000 steps): 2500 s per interval -> 3 intervals = 3000 steps per chunk ->
    # ceil(10000 / 3000) = 4 chunks (the old formula said 3 and never finished).
    tok = [line for line in _sbatch_lines(result.stdout) if "tok.sbatch" in line]
    gen = [line for line in _sbatch_lines(result.stdout) if "gen.sbatch" in line]
    assert len(tok) == 2 and len(gen) == 4
    assert "2 chunk(s) (pilot pilot_tok_vq)" in result.stderr
    assert "4 chunk(s) (pilot pilot_gen_vq_ce)" in result.stderr
    assert "afterany:<JOB_3>" in gen[3]
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


def test_lite_full_run_skips_the_shared_finished_tokenizer(tmp_path: Path) -> None:
    """T5.6: lite full runs skip the tokenizer when tok_*_10k/DONE exists."""
    root = _fixture_repo(tmp_path)
    _done(root, "tok_vq_10k")
    _seed_pilot(root, "pilot_gen_vq_ce_lite", 1.0)

    result = _run(root, ["g0", "full", "--config", "configs/tinystories_lite.yaml"])

    assert result.returncode == 0, result.stderr
    assert "SKIP tok_vq_10k" in result.stdout
    assert not [line for line in _sbatch_lines(result.stdout) if "tok.sbatch" in line]
    gen = [line for line in _sbatch_lines(result.stdout) if "gen.sbatch" in line]
    assert len(gen) == 2
    assert "gen_vq_ce_lite" in gen[0]
    assert "TOKENIZER=results/tok_vq_10k/ckpt/latest.pt" in gen[0]
    assert "tok_vq_10k_lite" not in result.stdout


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


TOKENIZER_CASES = [
    ("g0", "tok_vq_10k", "pilot_tok_vq"),
    ("g1", "tok_noisy_10k", "pilot_tok_noisy"),
]


@pytest.mark.parametrize("phase,real,pilot", TOKENIZER_CASES)
def test_pilot_submits_a_pilot_tokenizer_when_the_real_one_is_missing(
    tmp_path: Path, phase: str, real: str, pilot: str
) -> None:
    """T5.6: without results/<real>/DONE the pilot trains its own tokenizer."""
    root = _fixture_repo(tmp_path)

    result = _run(root, [phase, "pilot"])

    assert result.returncode == 0, result.stderr
    toks = [line for line in _sbatch_lines(result.stdout) if "tok.sbatch" in line]
    assert len(toks) == 1
    assert f"RUN_ID={pilot}" in toks[0]
    assert "STEPS=500" in toks[0]
    assert f"TOKENIZER=results/{pilot}/ckpt/latest.pt" in result.stdout
    assert f"results/{real}/ckpt" not in result.stdout


@pytest.mark.parametrize("phase,real,pilot", TOKENIZER_CASES)
def test_pilot_reuses_a_finished_real_tokenizer(
    tmp_path: Path, phase: str, real: str, pilot: str
) -> None:
    """T5.6: results/<real>/DONE means no pilot tokenizer and a real TOKENIZER."""
    root = _fixture_repo(tmp_path)
    _done(root, real)

    result = _run(root, [phase, "pilot"])

    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert not [line for line in _sbatch_lines(out) if "tok.sbatch" in line]
    assert f"REUSE {real}" in out
    assert pilot not in out
    downstream = [
        line
        for line in _sbatch_lines(out)
        if "gen.sbatch" in line or "eval.sbatch" in line
    ]
    assert downstream
    assert all(f"TOKENIZER=results/{real}/ckpt/latest.pt" in line for line in downstream)
    # No tokenizer chain upstream, so the generators wait for nothing; the eval
    # still waits for its generator.
    gens = [line for line in downstream if "gen.sbatch" in line]
    assert gens and not any("--dependency" in line for line in gens)


def test_lite_pilot_reuses_the_shared_real_tokenizer(tmp_path: Path) -> None:
    """T5.6: a lite pilot reads the shared dev tokenizer, unsuffixed."""
    root = _fixture_repo(tmp_path)
    _done(root, "tok_vq_10k")

    result = _run(root, ["g0", "pilot", "--config", "configs/tinystories_lite.yaml"])

    assert result.returncode == 0, result.stderr
    assert set(re.findall(r"RUN_ID=([^,\s]+)", result.stdout)) == {
        "pilot_gen_vq_ce_lite"
    }
    assert "TOKENIZER=results/tok_vq_10k/ckpt/latest.pt" in result.stdout
    assert "tok_vq_10k_lite" not in result.stdout


def test_dev_and_lite_configs_share_the_tokenizer_section() -> None:
    """T5.6: identical tokenizer sections are why tok_*_10k is shared."""
    root = Path(__file__).resolve().parents[1]
    pairs = [
        ("configs/tinystories_dev.yaml", "configs/tinystories_lite.yaml"),
        ("configs/jepa_dev.yaml", "configs/jepa_lite.yaml"),
    ]
    for dev, lite in pairs:
        dev_cfg = yaml.safe_load((root / dev).read_text(encoding="utf-8"))
        lite_cfg = yaml.safe_load((root / lite).read_text(encoding="utf-8"))
        assert dev_cfg["tokenizer"] == lite_cfg["tokenizer"], lite


def test_every_config_satisfies_batch_size_divisibility() -> None:
    """T5.6: batch_size % grad_accum == 0, micro-batch 64 in dev and lite."""
    root = Path(__file__).resolve().parents[1]
    for path in sorted((root / "configs").glob("*.yaml")):
        train = yaml.safe_load(path.read_text(encoding="utf-8"))["generator"]["train"]
        batch, accum = train["batch_size"], train["grad_accum"]
        assert accum >= 1, path.name
        assert batch % accum == 0, path.name
    expected = {
        "tinystories_dev.yaml": 64,
        "jepa_dev.yaml": 64,
        "tinystories_lite.yaml": 64,
        "jepa_lite.yaml": 64,
    }
    for name, micro in expected.items():
        train = yaml.safe_load((root / "configs" / name).read_text(encoding="utf-8"))[
            "generator"
        ]["train"]
        assert train["batch_size"] // train["grad_accum"] == micro, name


def test_plan_reports_jobs_gpu_hours_micro_batch_and_order(tmp_path: Path) -> None:
    """T5.6: the dry-run plan ends with jobs, GPU-hours, micro-batch and order."""
    root = _fixture_repo(tmp_path)

    result = _run(root, ["g0", "pilot"])

    assert result.returncode == 0, result.stderr
    out = result.stdout
    # 3 pilot jobs x 00:20:00 = 1.00 requested GPU-hour.
    assert "3 job(s), 1.00 requested GPU-hour(s)" in out
    assert (
        "micro-batch 64 (batch 1024 / grad_accum 16, configs/tinystories_dev.yaml)"
        in out
    )
    order = [
        line.strip()
        for line in out.splitlines()
        if re.match(r"\d+\. scripts/slurm/", line.strip())
    ]
    assert len(order) == 3, order
    assert order[0].startswith("1. scripts/slurm/tok.sbatch pilot_tok_vq")
    assert order[1].startswith("2. scripts/slurm/gen.sbatch pilot_gen_vq_ce")
    assert order[2].startswith("3. scripts/slurm/eval.sbatch pilot_gen_vq_ce")


def test_plan_prints_the_micro_batch_of_the_selected_config(tmp_path: Path) -> None:
    root = _fixture_repo(tmp_path)

    lite = _run(root, ["g0", "pilot", "--config", "configs/tinystories_lite.yaml"])
    jepa = _run(root, ["g2", "pilot", "--config", "configs/jepa_lite.yaml"])

    assert (
        "micro-batch 64 (batch 256 / grad_accum 4, configs/tinystories_lite.yaml)"
        in lite.stdout
    )
    assert (
        "micro-batch 64 (batch 256 / grad_accum 4, configs/jepa_lite.yaml)"
        in jepa.stdout
    )


def test_scripts_run_by_other_scripts_are_executable_in_the_index() -> None:
    """T5.6: a script another script launches by path must be mode 100755."""
    root = Path(__file__).resolve().parents[1]
    index = subprocess.run(
        ["git", "ls-files", "-s", "--", "scripts"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if index.returncode != 0 or not index.stdout.strip():
        pytest.skip("git index unavailable (not a checkout)")
    modes: dict[str, str] = {}
    for line in index.stdout.splitlines():
        meta, _, path = line.partition("\t")
        modes[path] = meta.split()[0]
    shell_script = re.compile(r"scripts/[A-Za-z0-9_./-]+\.sh")
    referenced: set[str] = set()
    for path in sorted((root / "scripts").rglob("*")):
        if path.is_file() and path.suffix in {".sh", ".sbatch", ".py"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            referenced.update(shell_script.findall(text))
    referenced &= set(modes)
    assert referenced, "no shell-script invocation found under scripts/"
    for path in sorted(referenced):
        assert modes[path] == "100755", f"{path} is {modes[path]} in the git index"
    for path in ("scripts/launch_phase.sh", "scripts/slurm/chain.sh"):
        assert modes.get(path) == "100755", path


def test_full_run_errors_when_one_resume_interval_does_not_fit_the_wall(
    tmp_path: Path,
) -> None:
    """T5.7 B1: 1000 steps x 10 s = 10000 s > 0.9 x 10800 s, so no chunk would
    ever reach its next checkpoint and the chain would restart forever."""
    root = _fixture_repo(tmp_path)
    _done(root, "tok_vq_10k")
    _seed_pilot(root, "pilot_gen_vq_ce", 10.0)

    result = _run(root, ["g0", "full"])

    assert result.returncode == 1
    assert "resume interval" in result.stderr
    assert "--wall" in result.stderr
    assert not _sbatch_lines(result.stdout)


def test_full_run_prefers_train_sec_per_step_and_says_so(tmp_path: Path) -> None:
    """T5.7 m1: pilot wall time includes diagnostics, Gen PPL and saves."""
    root = _fixture_repo(tmp_path)
    _done(root, "tok_vq_10k")
    _seed_pilot(root, "pilot_gen_vq_ce", 10.0, train_sec_per_step=1.0)

    result = _run(root, ["g0", "full"])

    assert result.returncode == 0, result.stderr
    gen = [line for line in _sbatch_lines(result.stdout) if "gen.sbatch" in line]
    # 1000 s per interval -> 9 intervals = 9000 steps per chunk -> 2 chunks.
    assert len(gen) == 2
    assert "train_sec_per_step" in result.stderr


def test_planner_constants_match_the_configs_the_launch_kit_uses() -> None:
    """The planner assumes a resume point every RESUME_FRACTION of a generator
    run and every TOK_RESUME_STEPS tokenizer steps; the configs must agree."""
    from jncm.config import load_config

    script = Path(LAUNCH).read_text(encoding="utf-8")
    fraction = float(re.search(r'^RESUME_FRACTION="([0-9.]+)"', script, re.M).group(1))
    tok_steps = int(re.search(r"^TOK_RESUME_STEPS=(\d+)", script, re.M).group(1))
    for name in (
        "tinystories_dev", "tinystories_lite", "jepa_dev", "jepa_lite",
    ):
        cfg = load_config(Path("configs") / f"{name}.yaml")
        train = cfg.generator.train
        assert train.save_milestones, name
        grid = [0.0] + sorted(train.milestones)
        gaps = [round(b - a, 6) for a, b in zip(grid, grid[1:])]
        assert max(gaps) <= fraction + 1e-9, (name, gaps)
        assert cfg.train.save_every <= tok_steps, name


def _eval_lines(stdout: str) -> list[str]:
    return [line for line in _sbatch_lines(stdout) if "eval.sbatch" in line]


def test_every_eval_job_of_a_staged_run_passes_its_head(tmp_path: Path) -> None:
    """T5.7 M1: eval.sbatch reads the dev YAML (head ce); without HEAD every
    flow and regress eval fails at startup with a head mismatch."""
    root = _fixture_repo(tmp_path)

    g0 = _eval_lines(_run(root, ["g0", "pilot"]).stdout)
    g1 = _eval_lines(_run(root, ["g1", "pilot"]).stdout)

    assert len(g0) == 1 and "HEAD=ce" in g0[0]
    flow = [line for line in g1 if "RUN_ID=pilot_gen_noisy_flow" in line]
    regress = [line for line in g1 if "RUN_ID=pilot_gen_noisy_regress" in line]
    assert len(flow) == 1 and "HEAD=flow" in flow[0]
    assert len(regress) == 1 and "HEAD=regress" in regress[0]


def test_joint_eval_jobs_carry_no_head(tmp_path: Path) -> None:
    """A joint checkpoint's own config is authoritative (run_eval.py)."""
    root = _fixture_repo(tmp_path)

    g2 = _eval_lines(_run(root, ["g2", "pilot"]).stdout)

    assert g2 and all("HEAD=" not in line for line in g2)


def test_full_g2_sizes_ablations_from_the_base_pilot(tmp_path: Path) -> None:
    """T5.7 m4: an ablation without its own pilot is sized by pilot_jepa_base."""
    root = _fixture_repo(tmp_path)
    _seed_pilot(root, "pilot_jepa_base", 1.0)

    result = _run(root, ["g2", "full", "--ablations"])

    assert result.returncode == 0, result.stderr
    jepa = [line for line in _sbatch_lines(result.stdout) if "jepa.sbatch" in line]
    # chain.sh prints one sbatch line per chunk: 7 arms x 2 chunks -> count
    # the distinct RUN_ID groups rather than the lines.
    runs = sorted({re.search(r"RUN_ID=([^,]+)", line).group(1) for line in jepa})
    assert len(runs) == 7
    # 1000-step resume interval x 1.0 s = 1000 s; floor(9720 / 1000) = 9
    # intervals per chunk -> ceil(10000 / 9000) = 2 chunks for every arm.
    assert result.stderr.count("-> 2 chunk(s)") == 7
    assert "-> 2 chunk(s) (pilot pilot_jepa_base)" in result.stderr
    assert result.stderr.count("(pilot pilot_jepa_base, ablation of jepa_base)") == 6


def test_full_g2_ablation_keeps_its_own_pilot_when_measured(tmp_path: Path) -> None:
    """T5.7 m4: the fallback never overrides an ablation's own measurement."""
    root = _fixture_repo(tmp_path)
    _seed_pilot(root, "pilot_jepa_base", 5.0)
    _seed_pilot(root, "pilot_jepa_no_vicreg", 1.0)

    result = _run(root, ["g2", "full", "--ablations"])

    assert result.returncode == 0, result.stderr
    # The base pilot (5.0 s/step): one 1000-step interval of 5000 s fits, so
    # 10 chunks; jepa_no_vicreg's own pilot (1.0 s/step) gives 2 chunks.
    assert "-> 10 chunk(s) (pilot pilot_jepa_base)" in result.stderr
    assert "-> 2 chunk(s) (pilot pilot_jepa_no_vicreg)" in result.stderr
    assert result.stderr.count("(pilot pilot_jepa_base, ablation of jepa_base)") == 5
    assert "run the pilot first" not in result.stderr


def test_full_g2_ablation_errors_without_any_measurement(tmp_path: Path) -> None:
    """T5.7 m4: the plain error still fires when no pilot and no --sec_per_step."""
    root = _fixture_repo(tmp_path)

    result = _run(root, ["g2", "full", "--ablations"])

    assert result.returncode == 1
    assert "--sec_per_step" in result.stderr
    assert not _sbatch_lines(result.stdout)


def test_full_g2_lite_sizes_ablations_from_the_lite_base_pilot(tmp_path: Path) -> None:
    """T5.7 m4: the ablation fallback carries the config suffix of the base."""
    root = _fixture_repo(tmp_path)
    _seed_pilot(root, "pilot_jepa_base_lite", 1.0)

    result = _run(root, ["g2", "full", "--ablations", "--config=configs/jepa_lite.yaml"])

    assert result.returncode == 0, result.stderr
    jepa = [line for line in _sbatch_lines(result.stdout) if "jepa.sbatch" in line]
    runs = sorted({re.search(r"RUN_ID=([^,]+)", line).group(1) for line in jepa})
    assert len(runs) == 7
    assert all(run.startswith("jepa_") and run.endswith("_lite") for run in runs)
    assert "-> 2 chunk(s) (pilot pilot_jepa_base_lite)" in result.stderr
    assert result.stderr.count("(pilot pilot_jepa_base_lite, ablation of jepa_base)") == 6
    # The dev base pilot is never touched for the lite run.
    assert not re.search(r"\(pilot pilot_jepa_base\)", result.stderr)


@pytest.mark.parametrize("var", LEAKED_VARS)
@pytest.mark.parametrize("value", ["1", ""])
def test_a_leaked_env_var_stops_the_launch(
    tmp_path: Path, var: str, value: str
) -> None:
    """T5.7 m6: --export=ALL would forward these to every job; refuse to run."""
    root = _fixture_repo(tmp_path)
    env = _clean_env()
    env[var] = value

    result = _run(root, ["g0", "pilot"], env)

    assert result.returncode == 2
    assert var in result.stderr
    assert "--export" in result.stderr
    assert not _sbatch_lines(result.stdout)


def test_launch_works_when_none_of_the_leaked_vars_is_set(tmp_path: Path) -> None:
    """T5.7 m6: a clean environment still plans the full g0 pilot."""
    root = _fixture_repo(tmp_path)
    env = _clean_env()
    for var in LEAKED_VARS:
        env.pop(var, None)

    result = _run(root, ["g0", "pilot"], env)

    assert result.returncode == 0, result.stderr
    assert len(_sbatch_lines(result.stdout)) == 3


def test_the_g2_loop_still_exports_sets_internally(tmp_path: Path) -> None:
    """T5.7 m6: the script's own `export SETS` in the G2 loop must keep working."""
    root = _fixture_repo(tmp_path)
    env = _clean_env()

    result = _run(root, ["g2", "pilot", "--ablations"], env)

    assert result.returncode == 0, result.stderr
    assert 'SETS="jepa.vicreg=false"' in result.stdout
    assert len([line for line in _sbatch_lines(result.stdout) if "jepa.sbatch" in line]) == 7


@pytest.mark.parametrize("value", ["0", "0.0", "0e3", "00.00"])
def test_a_zero_sec_per_step_flag_is_rejected_with_a_clear_message(
    tmp_path: Path, value: str
) -> None:
    """Second review MINOR-2: the validation said 'positive number' but accepted
    0, and the planner then died inside awk with 'division by zero'."""
    root = _fixture_repo(tmp_path)

    result = _run(root, ["g0", "full", f"--sec_per_step={value}"])

    assert result.returncode == 2
    assert "--sec_per_step must be a positive number" in result.stderr
    assert "awk" not in result.stderr
    assert not _sbatch_lines(result.stdout)


def test_a_zero_sec_per_step_in_a_pilot_record_is_rejected_clearly(tmp_path: Path) -> None:
    root = _fixture_repo(tmp_path)
    _done(root, "tok_vq_10k")
    _seed_pilot(root, "pilot_gen_vq_ce", 0.0)

    result = _run(root, ["g0", "full"])

    assert result.returncode == 1
    assert "sec_per_step must be positive" in result.stderr
    assert "pilot_gen_vq_ce" in result.stderr
    assert "awk" not in result.stderr
    assert not _sbatch_lines(result.stdout)


def test_usage_documents_the_option_forms_the_parser_accepts() -> None:
    """The usage once said `--wall HH:MM:SS` while only `--wall=HH:MM:SS` parses, so the
    first attempt to give a pilot a longer wall (2026-11-20) died with 'unknown option'."""
    text = Path(LAUNCH).read_text(encoding="utf-8")

    assert "[--wall=HH:MM:SS] [--sec_per_step=X]" in text
    assert "--wall HH:MM:SS" not in text

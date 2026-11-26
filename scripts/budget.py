#!/usr/bin/env python
"""Projected H100-hours for every dev run from the pilot measurements (T4.5).

Each run is projected as ``steps * sec_per_step / 3600``. ``sec_per_step`` is
the latest record of the matching pilot run (``results/pilot_*/metrics.jsonl``,
written by every trainer since T4.5); a run without a pilot is reported as
``MISSING`` and excluded from the total, so the total is only trustworthy when
nothing is missing. Eval jobs are not included: they have no ``sec_per_step``
and are short array jobs.

A single run above 12 H100-hours is flagged: with our fair-share over-used, a
job that long would wait too long, and it must be chained into shorter chunks
(``scripts/slurm/chain.sh``) instead of submitted whole.

``--config`` selects the configuration the projection is for. A non-dev config
adds its stem as a run-id suffix, exactly as ``scripts/launch_phase.sh`` does
(``configs/tinystories_lite.yaml`` -> ``gen_vq_ce_lite``), and only pilot
records carrying that suffix are used, so a dev pilot never sizes a lite run.
Tokenizer run ids are never suffixed (T5.6): the dev and lite configs share one
``tok_vq_10k`` / ``tok_noisy_10k`` pair, so those rows keep their bare ids.

A run whose results/<run>/DONE file exists (the shared tokenizers
tok_vq_10k / tok_noisy_10k, or a finished full run) has nothing left to spend,
so it is printed with DONE in the sec/step column and 0.00 hours, never
MISSING: the total is the projection of what is still to run (T5.7 m5). The six
G2 ablations never fly a pilot of their own; they are projected from the base
pilot (pilot_jepa_base, config suffix applied), exactly as
``scripts/launch_phase.sh`` sizes them.

Disk (T4.10, SPEC 7 v1.7): after the table, one ``DISK <run>:`` line per run
gives what the whole run writes under ``results/<run>/ckpt`` (GB = 1e9 B, with
GiB beside it because the cluster quota is in GiB). It comes from the parameter
counts in the pilot's metrics records (``n_params_*``, logged at step 1) and the
save cadence of the run's config family (``generator.train.milestones`` and
``save_every`` of ``configs/tinystories_<suffix>.yaml`` for staged runs,
``configs/jepa_<suffix>.yaml`` for G2): every milestone file holds 4 B per stored
parameter, and ``latest.pt`` plus every ``step_<n>.pt`` add 8 B of Adam state per
trainable parameter. A staged generator stores its backbone and head (the frozen
tokenizer is not saved); a joint run stores tokenizer, backbone and head, plus
the EMA copy of the tokenizer when its pilot ran with ``ema``. Ablations use the
base pilot's counts, an upper bound (``head: regress`` and ``ema: false`` store
less). A run without counts, or a tokenizer run (its trainer saves no
milestones), is ``n/a`` with the reason; ``n/a`` never changes the exit status.
The figure is the whole run, not what is left of a run in progress.

Usage:
    python scripts/budget.py
    python scripts/budget.py --results results
    python scripts/budget.py --config configs/tinystories_lite.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

MAX_HOURS = 12.0
ROOT = Path(__file__).resolve().parents[1]
GB = 1e9
GIB = float(2**30)
COUNT_KEYS = ("n_params_tokenizer", "n_params_backbone", "n_params_head", "n_params_total")

# (phase, run id, pilot run id, steps) -- the dev budgets the launch kit submits.
RUNS: list[tuple[str, str, str, int]] = [
    ("G0", "tok_vq_10k", "pilot_tok_vq", 10000),
    ("G0", "gen_vq_ce_dev", "pilot_gen_vq_ce", 10000),
    ("G1", "tok_noisy_10k", "pilot_tok_noisy", 10000),
    ("G1", "gen_noisy_flow_dev", "pilot_gen_noisy_flow", 10000),
    ("G1", "gen_noisy_regress_dev", "pilot_gen_noisy_regress", 10000),
    ("G2", "jepa_base_dev", "pilot_jepa_base", 10000),
    ("G2", "jepa_no_vicreg_dev", "pilot_jepa_no_vicreg", 10000),
    ("G2", "jepa_no_dec_through_E_dev", "pilot_jepa_no_dec_through_E", 10000),
    ("G2", "jepa_neither_dev", "pilot_jepa_neither", 10000),
    ("G2", "jepa_ema_clean_dev", "pilot_jepa_ema_clean", 10000),
    ("G2", "jepa_regress_dev", "pilot_jepa_regress", 10000),
    ("G2", "jepa_no_ema_dev", "pilot_jepa_no_ema", 10000),
]

HEADER = f"{'phase':<5} {'run':<27} {'steps':>7} {'sec/step':>10} {'hours':>7}"


def config_suffix(config: Path | None) -> str:
    """Run-id suffix for a config file, ``""`` for the dev defaults (T5.5c).

    ``configs/tinystories_lite.yaml`` and ``configs/jepa_lite.yaml`` both give
    ``lite``, matching the rule in ``scripts/launch_phase.sh``.
    """
    if config is None:
        return ""
    stem = config.stem
    if stem.endswith("_dev"):
        return ""
    for prefix in ("tinystories_", "jepa_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
    return stem


def with_suffix(run_id: str, suffix: str) -> str:
    """``gen_vq_ce_dev`` + ``lite`` -> ``gen_vq_ce_lite``; else append ``_lite``."""
    if not suffix:
        return run_id
    if run_id.endswith("_dev"):
        return f"{run_id[: -len('_dev')]}_{suffix}"
    return f"{run_id}_{suffix}"


def is_tokenizer(run_id: str) -> bool:
    """Tokenizer runs are shared by the dev and lite configs (T5.6)."""
    base = run_id[len("pilot_") :] if run_id.startswith("pilot_") else run_id
    return base.startswith("tok_")


def suffixed_run_id(run_id: str, suffix: str) -> str:
    """The run id the launcher builds: tokenizer ids are never suffixed."""
    return run_id if is_tokenizer(run_id) else with_suffix(run_id, suffix)


def pilot_sec_per_step(results_dir: Path, pilot_id: str) -> float | None:
    """``sec_per_step`` of the last record of a pilot run, ``None`` if absent."""
    path = results_dir / pilot_id / "metrics.jsonl"
    if not path.is_file():
        return None
    wall: object = None
    train: object = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        wall = record.get("sec_per_step", wall)
        train = record.get("train_sec_per_step", train)
    # T5.7 m1: the training step alone; wall time includes diagnostics, Gen PPL
    # and checkpoint saves and is what pilots from before T5.7 carry.
    value = train if isinstance(train, (int, float)) else wall
    if not isinstance(value, (int, float)):
        return None
    return float(value)


def pilot_param_counts(results_dir: Path, pilot_id: str) -> dict[str, object] | None:
    """The last pilot record with every ``n_params_*`` count (and ``ema``), or None."""
    path = results_dir / pilot_id / "metrics.jsonl"
    if not path.is_file():
        return None
    found: dict[str, object] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if all(isinstance(record.get(key), int) for key in COUNT_KEYS):
            found = {key: record[key] for key in COUNT_KEYS}
            if "ema" in record:
                found["ema"] = record["ema"]
    return found


def family_config(config: Path | None, suffix: str, family: str) -> Path:
    """The config a row's cadence comes from: ``family`` is tinystories or jepa."""
    if config is not None and config.stem.startswith(f"{family}_"):
        return config
    return ROOT / "configs" / f"{family}_{suffix or 'dev'}.yaml"


def path_label(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def save_cadence(config: Path) -> tuple[int, int]:
    """(number of milestone files, save_every) of ``generator.train`` in a config."""
    data = yaml.safe_load(config.read_text(encoding="utf-8"))
    try:
        train = data["generator"]["train"]
        milestones = len(train["milestones"]) if train["save_milestones"] else 0
        save_every = int(train["save_every"])
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{config}: no generator.train milestones/save_milestones/save_every") from exc
    if save_every < 1:
        raise ValueError(f"{config}: save_every must be >= 1, got {save_every}")
    return milestones, save_every


def run_disk_bytes(
    counts: dict[str, object], *, joint: bool, steps: int, milestones: int, save_every: int
) -> tuple[int, int, int]:
    """(bytes per milestone, bytes per resume file, number of resume files).

    Resume files: latest.pt plus a step_<n>.pt every save_every steps and at the
    last step (jncm/train_generator.py, jncm/train_jepa.py).
    """
    tokenizer = int(counts["n_params_tokenizer"])  # type: ignore[arg-type]
    if joint:
        trainable = int(counts["n_params_total"])  # type: ignore[arg-type]
        stored = trainable + (tokenizer if counts["ema"] else 0)
    else:
        trainable = int(counts["n_params_backbone"]) + int(counts["n_params_head"])  # type: ignore[arg-type]
        stored = trainable
    milestone = 4 * stored
    resume = 4 * stored + 8 * trainable
    periodic = steps // save_every + (1 if steps % save_every else 0)
    return milestone, resume, 1 + periodic


def disk_line(
    results_dir: Path,
    run_id: str,
    pilot_id: str,
    *,
    phase: str,
    steps: int,
    cadence_config: Path,
    from_base: bool,
) -> tuple[str, int | None]:
    """One ``DISK <run>:`` line and the whole run's bytes (None when n/a)."""
    if is_tokenizer(run_id):
        return (
            f"DISK {run_id}: n/a (tokenizer run: its trainer saves no milestones;"
            " not modelled, TASKS E8)",
            None,
        )
    counts = pilot_param_counts(results_dir, pilot_id)
    if counts is None:
        return f"DISK {run_id}: n/a (pilot {pilot_id} has no n_params_* record)", None
    joint = phase == "G2"
    if joint and not isinstance(counts.get("ema"), bool):
        return f"DISK {run_id}: n/a (pilot {pilot_id} records no ema flag)", None
    if not cadence_config.is_file():
        return f"DISK {run_id}: n/a (no config {path_label(cadence_config)} for the save cadence)", None
    milestones, save_every = save_cadence(cadence_config)
    milestone, resume, n_resume = run_disk_bytes(
        counts, joint=joint, steps=steps, milestones=milestones, save_every=save_every
    )
    total = milestones * milestone + n_resume * resume
    bound = f"; upper bound (base pilot {pilot_id})" if from_base else ""
    line = (
        f"DISK {run_id}: {total / GB:.2f} GB = {total / GIB:.2f} GiB (whole run: "
        f"{milestones} milestone(s) x {milestone / GB:.3f} GB + {n_resume} resume file(s)"
        f" x {resume / GB:.3f} GB; params from {pilot_id}{bound};"
        f" cadence {path_label(cadence_config)})"
    )
    return line, total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="config file the projection is for (default: the dev configs)",
    )
    args = parser.parse_args(argv)
    if args.config is not None and not args.config.is_file():
        parser.error(f"config file not found: {args.config}")

    suffix = config_suffix(args.config)
    print("== jncm budget projection (H100-hours = steps x pilot sec_per_step / 3600)")
    print("sec/step: train_sec_per_step (training step only) when the pilot has it, else wall sec_per_step")
    config_label = (
        args.config.as_posix() if args.config is not None else "template defaults (dev)"
    )
    print(f"config: {config_label}")
    print(f"run-id suffix: {'' if not suffix else '_' + suffix}")
    print(HEADER)
    totals: dict[str, float] = {}
    missing: list[str] = []
    flagged: list[tuple[str, float]] = []
    measured: list[float] = []
    # T4.10: the disk lines are printed after the table (its rows stay unchanged).
    disk_lines: list[str] = []
    disk_total = 0
    disk_sized = 0
    disk_na = 0
    for phase, base_run, base_pilot, steps in RUNS:
        run_id = suffixed_run_id(base_run, suffix)
        if (args.results / run_id / "DONE").is_file():
            # T5.7 m5(a): a finished run has already spent its time; it adds
            # 0.00 hours instead of being reported as MISSING.
            totals.setdefault(phase, 0.0)
            measured.append(0.0)
            print(f"{phase:<5} {run_id:<27} {steps:>7} {'DONE':>10} {0.0:>7.2f}")
            disk_lines.append(f"DISK {run_id}: DONE (nothing left to write)")
            continue
        pilot_id = suffixed_run_id(base_pilot, suffix)
        from_base = phase == "G2" and base_run != "jepa_base_dev"
        if from_base:
            # T5.7 m5(b): ablations have no pilot of their own; the base
            # pilot (same config suffix) is their measurement.
            pilot_id = suffixed_run_id("pilot_jepa_base", suffix)
        family = "jepa" if phase == "G2" else "tinystories"
        line, size = disk_line(
            args.results,
            run_id,
            pilot_id,
            phase=phase,
            steps=steps,
            cadence_config=family_config(args.config, suffix, family),
            from_base=from_base,
        )
        disk_lines.append(line)
        if size is None:
            disk_na += 1
        else:
            disk_total += size
            disk_sized += 1
        sec = pilot_sec_per_step(args.results, pilot_id)
        if sec is None:
            print(f"{phase:<5} {run_id:<27} {steps:>7} {'MISSING':>10} {'n/a':>7}")
            missing.append(f"{run_id} (pilot {pilot_id})")
            continue
        hours = steps * sec / 3600.0
        totals[phase] = totals.get(phase, 0.0) + hours
        measured.append(hours)
        if hours > MAX_HOURS:
            flagged.append((run_id, hours))
        print(f"{phase:<5} {run_id:<27} {steps:>7} {sec:>10.4f} {hours:>7.2f}")

    for phase in ("G0", "G1", "G2"):
        if phase in totals:
            print(f"{phase} subtotal: {totals[phase]:.2f} H100-hours")
    print(f"TOTAL: {sum(measured):.2f} H100-hours over {len(measured)} run(s)")

    print(
        "== disk per run (GB = 1e9 B; milestones 4 B per stored parameter,"
        " resume files 4 B + 8 B of Adam state per trainable parameter)"
    )
    for line in disk_lines:
        print(line)
    print(
        f"DISK TOTAL: {disk_total / GB:.2f} GB = {disk_total / GIB:.2f} GiB over"
        f" {disk_sized} run(s) with a size (whole runs, not DONE); {disk_na} n/a"
    )

    for run_id, hours in flagged:
        print(f"FLAG {run_id}: {hours:.2f} h > {MAX_HOURS:.1f} h (chain it into shorter chunks)")
    for entry in missing:
        print(f"MISSING {entry}: no sec_per_step; run that pilot first")
    return 1 if flagged or missing else 0


if __name__ == "__main__":
    raise SystemExit(main())

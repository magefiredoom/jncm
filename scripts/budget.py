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

Usage:
    python scripts/budget.py
    python scripts/budget.py --results results
    python scripts/budget.py --config configs/tinystories_lite.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MAX_HOURS = 12.0

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
    value: object = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        value = record.get("sec_per_step")
    if not isinstance(value, (int, float)):
        return None
    return float(value)


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
    for phase, base_run, base_pilot, steps in RUNS:
        run_id = suffixed_run_id(base_run, suffix)
        pilot_id = suffixed_run_id(base_pilot, suffix)
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

    for run_id, hours in flagged:
        print(f"FLAG {run_id}: {hours:.2f} h > {MAX_HOURS:.1f} h (chain it into shorter chunks)")
    for entry in missing:
        print(f"MISSING {entry}: no sec_per_step; run that pilot first")
    return 1 if flagged or missing else 0


if __name__ == "__main__":
    raise SystemExit(main())

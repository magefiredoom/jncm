#!/usr/bin/env python
"""Gate G0a report: exact-token accuracy of the two dev tokenizers.

Reads ``results/<run_id>/metrics.jsonl`` for the runs given by ``--runs``
(default: the 10k gate pair ``tok_vq_10k tok_noisy_10k``) and prints, for every
evaluation record, the step, ``acc_full``, the rounded ``acc_cum`` list and the
SPEC 4.5 diagnostics (codebook usage/perplexity for vq; std/effective rank for
noisy).

The final record is checked against the G0a threshold of the run's quantizer
variant: 0.999 for vq and 0.995 for noisy. The variant comes from
``results/<run_id>/config.yaml`` (``tokenizer.quantizer``) and falls back to the
run id (``..._vq_...`` / ``..._noisy_...``). A run that is missing, never
evaluated, of unknown variant, or whose final ``data_source`` is not ``real``
fails.

Usage:
    python scripts/g0a_report.py [--results results]
        [--runs tok_vq_10k tok_noisy_10k] [--last]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

# G0a thresholds per quantizer variant (TASKS gate G0a: 0.999 vq / 0.995 noisy).
THRESHOLDS = {"vq": 0.999, "noisy": 0.995}
DEFAULT_RUNS = ["tok_vq_10k", "tok_noisy_10k"]


def load_records(run_dir: Path) -> list[dict[str, Any]]:
    """Load every JSON record from ``run_dir/metrics.jsonl``."""
    path = run_dir / "metrics.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}")
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"{path} is empty")
    return records


def run_variant(run_id: str, run_dir: Path) -> str:
    """Quantizer variant of a run: its ``config.yaml``, else its run id."""
    config_path = run_dir / "config.yaml"
    if config_path.is_file():
        try:
            data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            data = {}
        tokenizer = data.get("tokenizer") if isinstance(data, dict) else None
        quantizer = tokenizer.get("quantizer") if isinstance(tokenizer, dict) else None
        if quantizer in THRESHOLDS:
            return str(quantizer)

    lowered = run_id.lower()
    if "noisy" in lowered:
        return "noisy"
    if "vq" in lowered:
        return "vq"
    raise ValueError(
        f"cannot tell the quantizer variant of run {run_id!r}: "
        f"no tokenizer.quantizer in {config_path} and no 'vq'/'noisy' in the id"
    )


def fmt_list(values: list[float], digits: int) -> str:
    """Format a short float list with a fixed number of decimals."""
    return "[" + ", ".join(f"{value:.{digits}f}" for value in values) + "]"


def format_record(record: dict[str, Any]) -> str:
    """One line: step, accuracy, then the variant diagnostics."""
    parts = [
        f"step {record.get('step')}",
        f"acc_full {float(record.get('acc_full', 0.0)):.4f}",
        f"acc_cum {fmt_list(record.get('acc_cum', []), 4)}",
        f"data_source {record.get('data_source')}",
        f"nominal_bits {float(record.get('nominal_bits', 0.0)):.3f}",
    ]
    if "codebook_usage" in record:
        parts.append(f"usage {fmt_list(record['codebook_usage'], 3)}")
        parts.append(f"perplexity {fmt_list(record['codebook_perplexity'], 1)}")
    if "std_min" in record:
        parts.append(f"std_min {fmt_list(record['std_min'], 4)}")
        parts.append(f"std_mean {fmt_list(record['std_mean'], 4)}")
        parts.append(f"eff_rank {fmt_list(record['eff_rank'], 2)}")
    return "  ".join(parts)


def report_run(run_id: str, results: Path, last_only: bool) -> bool:
    """Print one run and return whether it passes its G0a threshold."""
    run_dir = results / run_id
    try:
        records = load_records(run_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"{run_id}: MISSING ({exc})")
        return False
    try:
        variant = run_variant(run_id, run_dir)
    except ValueError as exc:
        print(f"{run_id}: FAIL ({exc})")
        return False
    threshold = THRESHOLDS[variant]

    selected = records[-1:] if last_only else records
    for record in selected:
        print(f"{run_id}: {format_record(record)}")

    final = records[-1]
    accuracy = float(final.get("acc_full", 0.0))
    source = final.get("data_source")
    passed = accuracy >= threshold and source == "real"
    verdict = "PASS" if passed else "FAIL"
    print(
        f"{verdict} {run_id} ({variant}): final acc_full {accuracy:.4f} "
        f"vs {threshold:.3f}, data_source {source}"
    )
    return passed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument(
        "--runs",
        nargs="+",
        default=list(DEFAULT_RUNS),
        help="run ids under results/ to report (default: the 10k gate pair)",
    )
    parser.add_argument(
        "--last",
        action="store_true",
        help="print only the final evaluation record of each run",
    )
    args = parser.parse_args(argv)

    outcomes = [
        report_run(run_id, args.results, args.last) for run_id in args.runs
    ]
    return 0 if all(outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Gate G0a report: exact-token accuracy of the two dev tokenizers.

Reads ``results/<run_id>/metrics.jsonl`` for ``tok_vq_dev`` and
``tok_noisy_dev`` and prints, for every evaluation record, the step,
``acc_full``, the rounded ``acc_cum`` list and the SPEC 4.5 diagnostics
(codebook usage/perplexity for vq; std/effective rank for noisy).

The final record is checked against the G0a thresholds: 0.999 for vq and
0.995 for noisy. A run that is missing, never evaluated, or whose final
``data_source`` is not ``real`` fails.

Usage:
    python scripts/g0a_report.py [--results results] [--last]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

THRESHOLDS = {"tok_vq_dev": 0.999, "tok_noisy_dev": 0.995}


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
    threshold = THRESHOLDS[run_id]
    try:
        records = load_records(results / run_id)
    except (FileNotFoundError, ValueError) as exc:
        print(f"{run_id}: MISSING ({exc})")
        return False

    selected = records[-1:] if last_only else records
    for record in selected:
        print(f"{run_id}: {format_record(record)}")

    final = records[-1]
    accuracy = float(final.get("acc_full", 0.0))
    source = final.get("data_source")
    passed = accuracy >= threshold and source == "real"
    verdict = "PASS" if passed else "FAIL"
    print(
        f"{verdict} {run_id}: final acc_full {accuracy:.4f} "
        f"vs {threshold:.3f}, data_source {source}"
    )
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument(
        "--last",
        action="store_true",
        help="print only the final evaluation record of each run",
    )
    args = parser.parse_args()

    outcomes = [
        report_run(run_id, args.results, args.last) for run_id in THRESHOLDS
    ]
    return 0 if all(outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())

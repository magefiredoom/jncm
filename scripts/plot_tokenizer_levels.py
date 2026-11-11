#!/usr/bin/env python
"""Figure C (T5.5c): the two 10k tokenizers, level by level.

Reads the last metrics record carrying ``acc_full`` from
``results/tok_vq_10k/metrics.jsonl`` and ``results/tok_noisy_10k/metrics.jsonl``
(written by ``jncm/train_tokenizer.py``) and draws two panels over the K levels:

  1. ``acc_cum`` per level: cumulative exact-token accuracy of the vq and the
     noisy tokenizer (the coarse-to-fine recovery curve of SPEC 4.5).
  2. level diagnostics: vq ``codebook_usage`` (fraction of the level codebook
     that is alive) and noisy ``eff_rank / 7`` (effective rank of the level
     codes at d_k = 8, normalised so 1.0 is a fully used space).

Saves ``figures/fig_c_tokenizers.png`` and prints the same numbers as a
markdown table, so the table can be pasted into a report.

Usage:
    python scripts/plot_tokenizer_levels.py
    python scripts/plot_tokenizer_levels.py --results results \\
        --out figures/fig_c_tokenizers.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

VQ_RUN = "tok_vq_10k"
NOISY_RUN = "tok_noisy_10k"
NOISY_RANK_SCALE = 7.0


def last_record(path: Path, key: str = "acc_full") -> dict:
    """Last JSONL record of ``path`` that contains ``key``.

    Args:
        path: metrics.jsonl path.
        key: field the record must carry (``acc_full``).

    Returns:
        The record as a dict.

    Raises:
        FileNotFoundError: ``path`` does not exist.
        ValueError: no record carries ``key``.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path}: train that tokenizer first (scripts/slurm/tok.sbatch)"
        )
    found: dict | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if isinstance(record, dict) and key in record:
            found = record
    if found is None:
        raise ValueError(f"{path} has no record with {key!r}")
    return found


def _series(record: dict, key: str, scale: float = 1.0) -> list[float] | None:
    """List field ``key`` scaled by ``scale``, ``None`` when absent/empty."""
    values = record.get(key)
    if not isinstance(values, list) or not values:
        return None
    return [float(value) / scale for value in values]


def plot_levels(vq: dict, noisy: dict, out: Path) -> Path:
    """Draw Figure C (two panels) and write ``out``.

    Args:
        vq: last metrics record of the vq tokenizer (``acc_cum``,
            ``codebook_usage``).
        noisy: last metrics record of the noisy tokenizer (``acc_cum``,
            ``eff_rank``).
        out: PNG path; parent directories are created.

    Returns:
        ``out``.

    Raises:
        ValueError: the level counts differ or no diagnostic series exists.
    """
    vq_acc = _series(vq, "acc_cum")
    noisy_acc = _series(noisy, "acc_cum")
    if vq_acc is None or noisy_acc is None:
        raise ValueError("both runs need a non-empty acc_cum list")
    if len(vq_acc) != len(noisy_acc):
        raise ValueError(
            f"level counts differ: vq {len(vq_acc)}, noisy {len(noisy_acc)}"
        )
    vq_usage = _series(vq, "codebook_usage")
    noisy_rank = _series(noisy, "eff_rank", NOISY_RANK_SCALE)
    if vq_usage is None and noisy_rank is None:
        raise ValueError("neither vq codebook_usage nor noisy eff_rank is present")

    levels = list(range(1, len(vq_acc) + 1))
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.0, 4.2))

    left.plot(levels, vq_acc, "o-", color="#1f77b4", label=f"vq ({VQ_RUN})")
    left.plot(levels, noisy_acc, "s--", color="#d62728", label=f"noisy ({NOISY_RUN})")
    left.set_title("C1  cumulative accuracy per level")
    left.set_xlabel("level (1 = coarsest)")
    left.set_ylabel("acc_cum (exact-token accuracy)")
    left.set_ylim(0.0, 1.05)
    left.grid(alpha=0.3)
    left.legend(loc="lower right")
    for level, value in enumerate(vq_acc, start=1):
        left.annotate(f"{value:.3f}", (level, value), fontsize=6, xytext=(0, 5),
                      textcoords="offset points", ha="center")

    if vq_usage is not None:
        right.plot(levels, vq_usage, "o-", color="#1f77b4", label="vq codebook_usage")
    if noisy_rank is not None:
        right.plot(
            levels,
            noisy_rank,
            "s--",
            color="#d62728",
            label=f"noisy eff_rank / {NOISY_RANK_SCALE:.0f}",
        )
    right.set_title("C2  level diagnostics")
    right.set_xlabel("level (1 = coarsest)")
    right.set_ylabel("fraction of the level code space used")
    right.set_ylim(0.0, 1.05)
    right.grid(alpha=0.3)
    right.legend(loc="lower left")
    right.axhline(0.1, color="grey", linestyle=":", linewidth=1,
                  label="dead-code reference 0.1")

    fig.suptitle(
        f"Figure C: tokenizers at step {vq.get('step')} (vq) / {noisy.get('step')} (noisy)"
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _cell(values: list[float], index: int) -> str:
    return f"{values[index]:.4f}" if index < len(values) else "n/a"


def markdown_table(vq: dict, noisy: dict) -> list[str]:
    """Markdown table of the Figure C numbers (one row per level)."""
    vq_acc = _series(vq, "acc_cum") or []
    noisy_acc = _series(noisy, "acc_cum") or []
    vq_usage = _series(vq, "codebook_usage") or []
    noisy_rank = _series(noisy, "eff_rank") or []
    scaled_rank = [value / NOISY_RANK_SCALE for value in noisy_rank]
    width = max(len(vq_acc), len(noisy_acc), len(vq_usage), len(noisy_rank))
    lines = [
        "| level | vq acc_cum | noisy acc_cum | vq codebook_usage | noisy eff_rank | noisy eff_rank/7 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for index in range(width):
        lines.append(
            f"| {index + 1} | {_cell(vq_acc, index)} | {_cell(noisy_acc, index)} | "
            f"{_cell(vq_usage, index)} | {_cell(noisy_rank, index)} | "
            f"{_cell(scaled_rank, index)} |"
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Figure C: tokenizer levels")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--vq", default=VQ_RUN, help="vq tokenizer run id")
    parser.add_argument("--noisy", default=NOISY_RUN, help="noisy tokenizer run id")
    parser.add_argument(
        "--out", type=Path, default=Path("figures/fig_c_tokenizers.png")
    )
    args = parser.parse_args(argv)

    records: dict[str, dict] = {}
    for name, run in (("vq", args.vq), ("noisy", args.noisy)):
        path = args.results / run / "metrics.jsonl"
        try:
            records[name] = last_record(path)
        except (FileNotFoundError, ValueError) as error:
            print(f"ERROR: {error}", file=sys.stderr)
            return 1

    try:
        out = plot_levels(records["vq"], records["noisy"], args.out)
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"wrote {out}")
    print(
        f"records: {args.vq} step {records['vq'].get('step')}, "
        f"{args.noisy} step {records['noisy'].get('step')}"
    )
    print()
    for line in markdown_table(records["vq"], records["noisy"]):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

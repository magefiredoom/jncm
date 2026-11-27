#!/usr/bin/env python
"""Figure A of paper section 4.3 (TASKS E7): Gen PPL against total training
positions for every arm, under both accountings of SPEC 6.2.

Left panel: ``total_positions`` (each joint sample charged once). Right panel:
``total_positions_conservative`` (both objectives charged); staged runs carry
one count and appear identically in both panels. Points are the merged milestone
evals (``results/<run_id>/eval/step_<n>.json``, as in ``gate_report.py``) with
the std over sampling seeds as a band. The dotted line is our data row (10.80).

Placeholders (rules in ``paper_results.py``; curves need 256 x 5 evals): an arm
that is a pilot, unfinished or under-sampled is drawn dashed with a dagger in
its legend, and the figure carries a PLACEHOLDER banner. Missing runs are left
out and listed on stdout.

Usage:
    python scripts/plot_fig_a.py
    python scripts/plot_fig_a.py --arm "vq + categorical=pilot_gen_vq_ce"
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter  # noqa: E402

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.gate_report import ACCOUNTINGS, REF_PPL, curve  # noqa: E402
from scripts.paper_results import (  # noqa: E402
    CURVE_SAMPLES,
    DAGGER,
    eval_reason,
    load_arm,
    parse_arms,
    run_reasons,
)

COLORS = ["#1f77b4", "#d62728", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]
TITLES = {
    "total_positions": "positions seen once",
    "total_positions_conservative": "conservative (both objectives charged)",
}
FIG_SIZE = (7.0, 2.9)
DPI = 300


def arm_series(results_dir: str | Path,
               arms: list[tuple[str, str]]) -> list[dict[str, Any]]:
    """Loaded arms with their curves under both accountings and reasons."""
    series: list[dict[str, Any]] = []
    for index, (label, run_id) in enumerate(arms):
        records = load_arm(results_dir, run_id)
        if records is None:
            series.append({"label": label, "run_id": run_id, "missing": True})
            continue
        reasons = run_reasons(results_dir, run_id, records)
        small = sorted({r for r in (eval_reason(rec, CURVE_SAMPLES)
                                    for rec in records) if r})
        reasons += small
        series.append({
            "label": label,
            "run_id": run_id,
            "missing": False,
            "color": COLORS[index % len(COLORS)],
            "reasons": reasons,
            "curves": {acc: curve(records, acc) for acc in ACCOUNTINGS},
        })
    return series


def plot(series: list[dict[str, Any]], out_path: str | Path) -> bool:
    """Draw the two panels; returns True when the figure is a placeholder."""
    present = [s for s in series if not s["missing"]]
    if not present:
        raise ValueError("no arm has merged eval records; nothing to plot")
    placeholder = any(s["reasons"] for s in present) or len(present) < len(series)
    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE, sharey=True)
    for ax, accounting in zip(axes, ACCOUNTINGS):
        for s in present:
            points = s["curves"][accounting]
            xs = [p["x"] / 1e9 for p in points]
            ys = [p["ppl"] for p in points]
            lo = [p["ppl"] - p["ppl_std"] for p in points]
            hi = [p["ppl"] + p["ppl_std"] for p in points]
            style = "--" if s["reasons"] else "-"
            label = s["label"] + (f" {DAGGER}" if s["reasons"] else "")
            ax.plot(xs, ys, style, marker="o", markersize=3, linewidth=1.4,
                    color=s["color"], label=label)
            ax.fill_between(xs, lo, hi, color=s["color"], alpha=0.15, linewidth=0)
        ax.axhline(REF_PPL, color="grey", linestyle=":", linewidth=1,
                   label=f"data row ({REF_PPL:.2f})")
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("total training positions (billions)", fontsize=8)
        ax.set_title(TITLES[accounting], fontsize=8)
        ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("Gen PPL (GPT-2-large, log)", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5),
               fontsize=7, frameon=False)
    if placeholder:
        fig.suptitle(f"PLACEHOLDER: {DAGGER} = pilot, partial or missing run; "
                     "not final results", color="#b00000", fontsize=8)
    fig.tight_layout(rect=(0, 0.1, 1, 0.95 if placeholder else 1), h_pad=0.2)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return placeholder


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--arm", action="append", metavar="LABEL=RUN_ID")
    parser.add_argument("--out", type=Path, default=Path("figures/fig_a_curves.png"))
    args = parser.parse_args(argv)
    series = arm_series(args.results, parse_arms(args.arm))
    placeholder = plot(series, args.out)
    for s in series:
        if s["missing"]:
            print(f"{s['label']} ({s['run_id']}): MISSING, not drawn")
        elif s["reasons"]:
            print(f"{s['label']} ({s['run_id']}): PLACEHOLDER: "
                  + "; ".join(s["reasons"]))
        else:
            print(f"{s['label']} ({s['run_id']}): final")
    print(f"wrote {args.out}{' (PLACEHOLDER)' if placeholder else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Figure B of paper section 4.3 (TASKS E7): level by level, the reconstruction
accuracy of each hierarchy and the explained variance of each predictor.

Left panel, ``acc_cum``: exact-token accuracy of the reconstruction from the
first k levels, from the last ``metrics.jsonl`` record carrying ``acc_cum``
(the staged tokenizers ``tok_vq_10k`` and ``tok_noisy_10k``, and any other run
given with ``--recon``). ASSUMPTION: "per-level reconstruction accuracy" in the
draft means this tokenizer metric (SPEC 4.5); a joint run appears only if its
records carry ``acc_cum``, which ``train_jepa.py`` does not log today.

Right panel, ``r2``: ``R2_k = 1 - MSE_k / Var(t_k)`` (SPEC 8) from the last
``metrics.jsonl`` record carrying ``r2``. Only the joint trainer logs it (on
its training batch, against the EMA targets). The staged generators log no
per-level R2; such arms are listed on the figure as "not logged" rather than
estimated. The regression head's coarsest-level collapse shows up here once a
staged R2 source exists.

Placeholder rules: a run id starting ``pilot_``/``corr_`` or without ``DONE``
is drawn dashed with a dagger, and the figure carries a PLACEHOLDER banner.

Usage:
    python scripts/plot_fig_b.py
    python scripts/plot_fig_b.py --r2 "JEPA (base)=pilot_jepa_base"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.paper_results import DAGGER, PILOT_PREFIXES  # noqa: E402

RECON_ARMS = [("vq", "tok_vq_10k"), ("continuous", "tok_noisy_10k")]
R2_ARMS = [
    ("continuous + flow", "gen_noisy_flow_dev"),
    ("continuous + regression", "gen_noisy_regress_dev"),
    ("JEPA (base)", "jepa_base_dev"),
]
COLORS = {"vq": "#1f77b4", "continuous": "#d62728",
          "continuous + flow": "#d62728", "continuous + regression": "#ff7f0e",
          "JEPA (base)": "#2ca02c"}
EXTRA_COLORS = ["#9467bd", "#8c564b", "#e377c2"]
FIG_SIZE = (7.0, 2.9)
DPI = 300


def override(arms: list[tuple[str, str]], specs: list[str] | None) -> list[tuple[str, str]]:
    """Apply ``label=run_id`` specs by label; new labels are appended."""
    out = list(arms)
    for spec in specs or []:
        label, sep, run_id = (part.strip() for part in spec.partition("="))
        if not sep or not label or not run_id:
            raise ValueError(f"expected 'label=run_id', got {spec!r}")
        names = [name for name, _ in out]
        if label in names:
            out[names.index(label)] = (label, run_id)
        else:
            out.append((label, run_id))
    return out


def last_with(path: Path, key: str) -> dict[str, Any] | None:
    """Last JSONL record of ``path`` carrying a list under ``key``, else None.

    A missing file returns None; a malformed line raises ``json.JSONDecodeError``.
    """
    if not path.is_file():
        return None
    found = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if isinstance(record.get(key), list):
                found = record
    return found


def reasons_for(results_dir: Path, run_id: str) -> list[str]:
    reasons = []
    if run_id.startswith(PILOT_PREFIXES):
        reasons.append("pilot or sweep run")
    if not (results_dir / run_id / "DONE").is_file():
        reasons.append("training not finished (no DONE)")
    return reasons


def collect(results_dir: Path, arms: list[tuple[str, str]], key: str) -> list[dict[str, Any]]:
    """One entry per arm: its per-level values under ``key`` or ``None``."""
    entries = []
    for index, (label, run_id) in enumerate(arms):
        record = last_with(results_dir / run_id / "metrics.jsonl", key)
        entries.append({
            "label": label,
            "run_id": run_id,
            "values": None if record is None else [float(v) for v in record[key]],
            "step": None if record is None else record.get("step"),
            "reasons": reasons_for(results_dir, run_id),
            "color": COLORS.get(label, EXTRA_COLORS[index % len(EXTRA_COLORS)]),
        })
    return entries


def _draw(ax: Any, entries: list[dict[str, Any]]) -> list[str]:
    absent = []
    for entry in entries:
        if entry["values"] is None:
            absent.append(entry["label"])
            continue
        levels = list(range(1, len(entry["values"]) + 1))
        style = "--" if entry["reasons"] else "-"
        label = entry["label"] + (f" {DAGGER}" if entry["reasons"] else "")
        ax.plot(levels, entry["values"], style, marker="o", markersize=3,
                linewidth=1.4, color=entry["color"], label=label)
    ax.set_xlabel("level k (1 = coarsest)", fontsize=8)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.tick_params(labelsize=7)
    return absent


def plot(recon: list[dict[str, Any]], r2: list[dict[str, Any]], out_path: Path) -> bool:
    """Draw both panels; returns True when the figure is a placeholder."""
    drawn = [e for e in recon + r2 if e["values"] is not None]
    if not drawn:
        raise ValueError("no run carries acc_cum or r2; nothing to plot")
    placeholder = any(e["reasons"] for e in drawn) or any(
        e["values"] is None for e in recon + r2)
    fig, (left, right) = plt.subplots(1, 2, figsize=FIG_SIZE)
    _draw(left, recon)
    left.set_ylabel("exact-token accuracy, first k levels", fontsize=8)
    left.set_ylim(0.0, 1.02)
    missing_r2 = _draw(right, r2)
    right.axhline(0.0, color="grey", linestyle=":", linewidth=1)
    right.set_ylabel(r"predictor $R^2_k$", fontsize=8)
    lowest = min([v for e in r2 if e["values"] for v in e["values"]], default=0.0)
    right.set_ylim(min(-0.05, lowest - 0.05), 1.02)
    if missing_r2:
        right.set_title("no per-level R2 logged: " + ", ".join(missing_r2),
                        fontsize=6, color="#555555")
    for ax in (left, right):
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=6, frameon=False, loc="upper left")
    if placeholder:
        fig.suptitle(f"PLACEHOLDER: {DAGGER} = pilot or unfinished run; "
                     "missing arms listed", color="#b00000", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.94 if placeholder else 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return placeholder


def markdown(title: str, entries: list[dict[str, Any]]) -> str:
    """Per-level values as a markdown table (rows: arms)."""
    width = max((len(e["values"]) for e in entries if e["values"]), default=0)
    lines = [f"{title}", "| arm | run | step | " + " | ".join(
        f"L{k}" for k in range(1, width + 1)) + " |", "|" + "---|" * (width + 3)]
    for e in entries:
        mark = f" {DAGGER}" if e["reasons"] else ""
        cells = (["not logged"] + [""] * (width - 1) if e["values"] is None
                 else [f"{v:.3f}" for v in e["values"]])
        lines.append(f"| {e['label']}{mark} | `{e['run_id']}` | {e['step'] or '-'} | "
                     + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--recon", action="append", metavar="LABEL=RUN_ID")
    parser.add_argument("--r2", action="append", metavar="LABEL=RUN_ID")
    parser.add_argument("--out", type=Path, default=Path("figures/fig_b_levels.png"))
    args = parser.parse_args(argv)
    recon = collect(args.results, override(RECON_ARMS, args.recon), "acc_cum")
    r2 = collect(args.results, override(R2_ARMS, args.r2), "r2")
    placeholder = plot(recon, r2, args.out)
    print(markdown("acc_cum", recon))
    print(markdown("r2", r2))
    for e in recon + r2:
        if e["reasons"] and e["values"] is not None:
            print(f"{DAGGER} {e['label']} ({e['run_id']}): " + "; ".join(e["reasons"]))
    print(f"wrote {args.out}{' (PLACEHOLDER)' if placeholder else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

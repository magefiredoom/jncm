#!/usr/bin/env python
"""Plot Gen PPL, entropy and diversity against total positions (SPEC 6.2, T4.2).

Reads the aggregate files written by ``scripts/run_eval.py``:
``results/<run_id>/eval/step_<n>.json``. One line per run; the x axis is
``total_positions`` from the record, which already includes the tokenizer budget
as an offset (SPEC 6.2). Per-seed shards (``step_<n>.seed_<i>.json``) are
ignored, so a merge in progress does not add half-aggregated points.

The JEPA joint runs record ``total_positions = joint_positions`` and
``total_positions_conservative = 2 * joint_positions`` (SPEC 6.2). ``--x``
selects which one is the x axis; with ``total_positions_conservative`` a staged
record (which has no joint count) falls back to its own ``total_positions``, so
both curves stay on one axis. Plot JEPA results once under each accounting.

Usage:
    python scripts/plot_curves.py --results results --out results/curves.png
    python scripts/plot_curves.py --x total_positions_conservative \\
        --out results/curves_conservative.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

AGGREGATE_RE = re.compile(r"^step_\d{6}\.json$")
X_KEYS = ("total_positions", "total_positions_conservative")


def total_positions(record: dict[str, Any]) -> int:
    """x coordinate: total positions, with the tokenizer budget as the offset."""
    if record.get("total_positions") is not None:
        return int(record["total_positions"])
    if record.get("gen_positions") is None:
        raise ValueError(
            "eval record has neither total_positions nor gen_positions"
        )
    return int(record["gen_positions"]) + int(record.get("tok_positions", 0))


def x_position(record: dict[str, Any], x_key: str = "total_positions") -> int:
    """x coordinate for one record under the selected position accounting.

    ``total_positions_conservative`` exists only on joint (JEPA) records; a
    staged record has no joint count, so it keeps its own ``total_positions``
    (SPEC 6.2).
    """
    if x_key not in X_KEYS:
        raise ValueError(f"x_key must be one of {X_KEYS}, got {x_key!r}")
    if x_key == "total_positions_conservative":
        value = record.get("total_positions_conservative")
        if value is not None:
            return int(value)
    return total_positions(record)


def collect_points(
    results_dir: str | Path, x_key: str = "total_positions"
) -> dict[str, list[dict[str, Any]]]:
    """Load every aggregate eval record, grouped by run and sorted by positions."""
    root = Path(results_dir)
    points: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(root.rglob("eval/step_*.json")):
        if not AGGREGATE_RE.match(path.name):
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        run_id = str(record.get("run_id") or path.parent.parent.name)
        point = dict(record)
        point["x"] = x_position(record, x_key)
        point["run_id"] = run_id
        points.setdefault(run_id, []).append(point)
    if not points:
        raise ValueError(f"no eval results found under {root}/**/eval/step_*.json")
    for run_points in points.values():
        run_points.sort(key=lambda point: point["x"])
    return dict(sorted(points.items()))


def _series(
    points: list[dict[str, Any]], key: str
) -> tuple[list[int], list[float]]:
    """(x, y) pairs for one metric, skipping records that do not carry it."""
    xs: list[int] = []
    ys: list[float] = []
    for point in points:
        value = point.get(key)
        if value is None:
            continue
        xs.append(point["x"])
        ys.append(float(value))
    return xs, ys


def plot_curves(
    results_dir: str | Path = "results",
    out_path: str | Path = "results/curves.png",
    x_key: str = "total_positions",
) -> Path:
    """Render the three panels and save a PNG; returns the output path."""
    by_run = collect_points(results_dir, x_key=x_key)
    fig, (ax_ppl, ax_entropy, ax_div) = plt.subplots(1, 3, figsize=(16.0, 4.5))
    colors = plt.get_cmap("tab10").colors

    for index, (run_id, points) in enumerate(by_run.items()):
        color = colors[index % len(colors)]
        xs, ys = _series(points, "ppl_mean")
        if xs:
            ax_ppl.plot(xs, ys, marker="o", color=color, label=run_id)
            stds = [
                point.get("ppl_std")
                for point in points
                if point.get("ppl_mean") is not None
            ]
            if len(stds) == len(ys) and all(std is not None for std in stds):
                if any(float(std) > 0.0 for std in stds):
                    lower = [y - float(std) for y, std in zip(ys, stds)]
                    upper = [y + float(std) for y, std in zip(ys, stds)]
                    ax_ppl.fill_between(xs, lower, upper, color=color, alpha=0.15)

        xs, ys = _series(points, "h_dataset")
        if xs:
            ax_entropy.plot(xs, ys, marker="o", color=color)
        xs, ys = _series(points, "h_gpt2")
        if xs:
            ax_entropy.plot(
                xs, ys, marker="s", linestyle="--", color=color, alpha=0.6
            )

        xs, ys = _series(points, "pairwise_agree")
        if xs:
            ax_div.plot(xs, ys, marker="o", color=color)
        xs, ys = _series(points, "unique_frac")
        if xs:
            ax_div.plot(xs, ys, marker="s", linestyle="--", color=color, alpha=0.6)

    all_x = [point["x"] for points in by_run.values() for point in points]
    if all(x > 0 for x in all_x):
        for ax in (ax_ppl, ax_entropy, ax_div):
            ax.set_xscale("log")

    ax_ppl.set_title("Gen PPL")
    ax_ppl.set_ylabel("Gen PPL (lower is better, log scale)")
    ax_ppl.set_yscale("log")
    ax_ppl.legend(fontsize=8)

    ax_entropy.set_title(
        "Entropy (solid: dataset tokens, dashed: GPT-2 tokens)"
    )
    ax_entropy.set_ylabel("H (nats)")

    ax_div.set_title("Diversity (solid: pairwise_agree, dashed: unique_frac)")
    ax_div.set_ylabel("fraction")
    ax_div.set_ylim(-0.02, 1.05)

    xlabel = (
        "total positions (conservative: 2 x joint positions)"
        if x_key == "total_positions_conservative"
        else "total positions (tokenizer budget + generator)"
    )
    for ax in (ax_ppl, ax_entropy, ax_div):
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Plot Gen PPL, entropy and diversity against total positions"
    )
    parser.add_argument("--results", type=str, default="results")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument(
        "--x",
        dest="x_key",
        choices=list(X_KEYS),
        default="total_positions",
        help=(
            "x axis: total_positions (default) or total_positions_conservative "
            "(joint JEPA runs; staged runs fall back to total_positions)"
        ),
    )
    args = parser.parse_args(argv)

    by_run = collect_points(args.results, x_key=args.x_key)
    for run_id, points in by_run.items():
        print(f"{run_id}: {len(points)} checkpoints")
    default_name = (
        "curves.png"
        if args.x_key == "total_positions"
        else "curves_conservative.png"
    )
    out = Path(args.out) if args.out else Path(args.results) / default_name
    print(f"wrote {plot_curves(args.results, out, x_key=args.x_key)}")


if __name__ == "__main__":
    main()

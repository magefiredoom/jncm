#!/usr/bin/env python
"""Table 1 of paper section 4.3 (TASKS E7), plus the arm list and placeholder
rules shared with ``plot_fig_a.py`` and ``plot_fig_b.py``.

Reads the merged eval aggregates that ``scripts/gate_report.py`` reads
(``results/<run_id>/eval/step_<n>.json``; per-seed shards ignored) and takes
each arm's last record by position. Every number comes from those files.

Placeholders: until the main runs finish, an arm can point at a pilot or a
partly trained run (``--arm "label=run_id"``). Its row is then marked with a
dagger and the reasons are printed under the table:
  - the run id starts with ``pilot_`` or ``corr_`` (a pilot or sweep run);
  - ``results/<run_id>/DONE`` is missing (training not finished);
  - the last milestone is below 100%;
  - the final eval is not 1024 samples x 5 seeds (SPEC 10.4: Table 1 uses the
    1024-sample re-evaluation; ASSUMPTION: it is read from the same
    ``eval/step_<n>.json`` and told apart by its ``n_samples``).
A missing run gets a row with no numbers.

Usage:
    python scripts/paper_results.py
    python scripts/paper_results.py --arm "vq + categorical=pilot_gen_vq_ce" \\
        --out figures/table1.md
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.gate_report import REF_H, REF_PPL, curve, load_run  # noqa: E402

# (label, default run id) in the order of the paper's Table 1 and Figure A.
DEFAULT_ARMS: list[tuple[str, str]] = [
    ("vq + categorical", "gen_vq_ce_dev"),
    ("continuous + flow", "gen_noisy_flow_dev"),
    ("continuous + regression", "gen_noisy_regress_dev"),
    ("JEPA (base)", "jepa_base_dev"),
]
PILOT_PREFIXES = ("pilot_", "corr_")
TABLE_SAMPLES = 1024
CURVE_SAMPLES = 256
N_SEEDS = 5
DAGGER = "†"


def parse_arms(specs: list[str] | None) -> list[tuple[str, str]]:
    """``label=run_id`` overrides applied onto ``DEFAULT_ARMS`` by label.

    A label that is not a default arm is appended. Raises ``ValueError`` on a
    spec without ``=`` or with an empty side.
    """
    arms = list(DEFAULT_ARMS)
    for spec in specs or []:
        label, sep, run_id = spec.partition("=")
        label, run_id = label.strip(), run_id.strip()
        if not sep or not label or not run_id:
            raise ValueError(f"--arm must be 'label=run_id', got {spec!r}")
        for index, (existing, _) in enumerate(arms):
            if existing == label:
                arms[index] = (label, run_id)
                break
        else:
            arms.append((label, run_id))
    return arms


def run_reasons(results_dir: str | Path, run_id: str,
                records: list[dict[str, Any]]) -> list[str]:
    """Why a run is not yet a final result (empty list: final training)."""
    reasons: list[str] = []
    if run_id.startswith(PILOT_PREFIXES):
        reasons.append("pilot or sweep run")
    if not (Path(results_dir) / run_id / "DONE").is_file():
        reasons.append("training not finished (no DONE)")
    fraction = records[-1].get("fraction") if records else None
    if fraction is None or float(fraction) < 1.0:
        label = "unknown" if fraction is None else f"{float(fraction) * 100:.0f}%"
        reasons.append(f"last milestone {label}")
    return reasons


def eval_reason(record: dict[str, Any], n_samples: int) -> str | None:
    """Reason text when an eval record is smaller than ``n_samples`` x 5."""
    samples = int(record.get("n_samples") or 0)
    seeds = int(record.get("n_seeds") or 0)
    if samples == n_samples and seeds == N_SEEDS:
        return None
    return f"eval {samples} x {seeds} (needs {n_samples} x {N_SEEDS})"


def curve_eval_reason(record: dict[str, Any]) -> str | None:
    """Reason text when a curve point has fewer than ``CURVE_SAMPLES`` x 5.

    More samples are fine: the SPEC 10.4 re-evaluation of the final milestone at
    ``TABLE_SAMPLES`` x 5 writes the same ``eval/step_<n>.json`` and so replaces
    that milestone's 256-sample record (review 2026-11-26, M1).
    """
    samples = int(record.get("n_samples") or 0)
    seeds = int(record.get("n_seeds") or 0)
    if samples >= CURVE_SAMPLES and seeds == N_SEEDS:
        return None
    return f"eval {samples} x {seeds} (needs at least {CURVE_SAMPLES} x {N_SEEDS})"


def load_arm(results_dir: str | Path, run_id: str) -> list[dict[str, Any]] | None:
    """Merged records of one run, ``None`` when the run has none."""
    try:
        return load_run(results_dir, run_id)
    except (FileNotFoundError, ValueError):
        return None


def model_params(record: dict[str, Any]) -> int | None:
    """Backbone + head parameters of an eval record (the frozen tokenizer,
    which ``n_params_total`` includes, is left out); ``None`` if not logged."""
    backbone, head = record.get("n_params_backbone"), record.get("n_params_head")
    if backbone is None or head is None:
        return None
    return int(backbone) + int(head)


def _num(value: float | None, digits: int, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _positions(value: int | None) -> str:
    return "-" if value is None else f"{value / 1e9:.2f}B"


def table_rows(results_dir: str | Path,
               arms: list[tuple[str, str]]) -> tuple[list[dict[str, Any]], bool]:
    """One row per arm from its last record; second value: any placeholder."""
    rows: list[dict[str, Any]] = []
    any_placeholder = False
    for label, run_id in arms:
        records = load_arm(results_dir, run_id)
        if records is None:
            rows.append({"label": label, "run_id": run_id, "point": None,
                         "reasons": ["no merged eval records"]})
            any_placeholder = True
            continue
        point = curve(records)[-1]
        reasons = run_reasons(results_dir, run_id, records)
        record = records[-1]
        extra = eval_reason(record, TABLE_SAMPLES)
        if extra:
            reasons.append(extra)
        any_placeholder = any_placeholder or bool(reasons)
        rows.append({"label": label, "run_id": run_id, "point": point,
                     "record": record, "reasons": reasons})
    return rows, any_placeholder


def render_table(rows: list[dict[str, Any]], placeholder: bool) -> str:
    """Markdown Table 1 with a dagger on every placeholder row."""
    header = ("| Method | Run | Params | Positions | Positions (cons.) | Gen PPL "
              f"| PPL / {REF_PPL:.2f} | H | H - {REF_H:.2f} | unique | agree |")
    lines = []
    if placeholder:
        lines.append(f"**PLACEHOLDER: rows marked {DAGGER} are not final "
                     "results (reasons below).**\n")
    lines += [header, "|" + "---|" * 11]
    notes: list[str] = []
    for row in rows:
        point = row["point"]
        mark = DAGGER if row["reasons"] else ""
        if row["reasons"]:
            notes.append(f"- {DAGGER} {row['label']} (`{row['run_id']}`): "
                         + "; ".join(row["reasons"]))
        if point is None:
            lines.append(f"| {row['label']}{mark} | `{row['run_id']}` |"
                         + " - |" * 9)
            continue
        h = point["h_gpt2"]
        params = model_params(row["record"])
        cons = point["total_positions_conservative"]
        lines.append(
            f"| {row['label']}{mark} | `{row['run_id']}` "
            f"| {'n/a' if params is None else f'{params / 1e6:.1f}M'} "
            f"| {_positions(point['total_positions'])} "
            f"| {_positions(cons) if cons is not None else 'same'} "
            f"| {point['ppl']:.2f} ± {point['ppl_std']:.2f} "
            f"| {point['ppl'] / REF_PPL:.3f} "
            f"| {_num(h, 2)} | {_num(None if h is None else h - REF_H, 2, True)} "
            f"| {_num(point['unique_frac'], 3)} | {_num(point['pairwise_agree'], 3)} |"
        )
    lines.append("")
    lines.append(f"Data row (real validation text, our pipeline): Gen PPL "
                 f"{REF_PPL:.2f}, H {REF_H:.2f}. PPL is mean ± std over sampling "
                 "seeds, one training seed (SPEC 10.7). Params are backbone + head "
                 "(`n_params_backbone + n_params_head`; the frozen tokenizer "
                 "is excluded). 'Positions (cons.)' charges both objectives "
                 "of a joint run (SPEC 6.2); staged runs have one count.")
    if notes:
        lines.append("")
        lines += notes
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--arm", action="append", metavar="LABEL=RUN_ID",
                        help="point an arm at another run (pilot or partial)")
    parser.add_argument("--out", type=Path, default=Path("figures/table1.md"))
    args = parser.parse_args(argv)
    rows, placeholder = table_rows(args.results, parse_arms(args.arm))
    text = render_table(rows, placeholder)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

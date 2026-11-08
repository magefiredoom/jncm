#!/usr/bin/env python
"""Gate report for G0, G1 and G2 (TASKS gates; SPEC 6.1/6.2, T4.4).

Reads the merged aggregates from ``scripts/run_eval.py``
(``results/<run_id>/eval/step_<n>.json``; shards are ignored) and prints one
table, then the verdicts: G0 (Gen PPL <= 20, H >= 3.8), G1 (noisy+flow within
15% of G0 at equal ``total_positions``, H within 0.15 of 4.03; noisy+regress
reported only, flagged when ``pairwise_agree > 0.5``) and G2 (JEPA matches or
beats the best Phase-1 run at equal positions under both accountings, each
verdict naming its accounting). Equal positions are compared by interpolation
inside the reference curve's range; outside it the line says ``not
comparable``. Exit code: 1 if a gate FAILs or a run has no merged eval records.

Usage:
    python scripts/gate_report.py
    python scripts/gate_report.py --jepa jepa_base_dev jepa_no_vicreg_dev
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

# Reference row (T1.2 protocol gate) and gate thresholds (TASKS, SPEC 6.1).
REF_PPL = 10.80
REF_H = 4.03
G0_MAX_PPL = 20.0
G0_MIN_H = 3.8
G1_PPL_RATIO = 1.15
G1_H_TOL = 0.15
FLAG_AGREE = 0.5

DEFAULT_G0 = "gen_vq_ce_dev"
DEFAULT_FLOW = "gen_noisy_flow_dev"
DEFAULT_REGRESS = "gen_noisy_regress_dev"
DEFAULT_JEPA = "jepa_base_dev"

ACCOUNTINGS = ("total_positions", "total_positions_conservative")
AGGREGATE_RE = re.compile(r"^step_\d+\.json$")

# (header, width, right-aligned) of the table columns.
COLUMNS = [
    ("gate", 4, False),
    ("run", 26, False),
    ("milestone", 6, True),
    ("total_positions", 15, True),
    ("conservative", 14, True),
    ("Gen PPL", 16, True),
    ("PPL/10.80", 9, True),
    ("H(gpt2)", 7, True),
    ("H-4.03", 7, True),
    ("H(data)", 7, True),
    ("unique", 6, True),
    ("agree", 6, True),
]


def _row(cells: list[str]) -> str:
    """Pad cells to the shared column widths."""
    return "  ".join(f"{cell:>{width}}" if right else f"{cell:<{width}}"
                     for cell, (_, width, right) in zip(cells, COLUMNS))


def _fmt(value: float | None, digits: int = 3, signed: bool = False) -> str:
    """Fixed-decimal string, ``n/a`` for a missing metric."""
    if value is None:
        return "n/a"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def x_position(record: dict[str, Any], accounting: str = "total_positions") -> int:
    """Position count under one accounting (joint records carry both).

    A staged record has no ``total_positions_conservative``, so it keeps its
    own ``total_positions`` (SPEC 6.2, as in ``scripts/plot_curves.py``).
    """
    if accounting not in ACCOUNTINGS:
        raise ValueError(f"accounting must be one of {ACCOUNTINGS}, got {accounting!r}")
    if accounting == "total_positions_conservative":
        conservative = record.get("total_positions_conservative")
        if conservative is not None:
            return int(conservative)
    if record.get("total_positions") is not None:
        return int(record["total_positions"])
    if record.get("gen_positions") is None:
        raise ValueError("eval record has neither total_positions nor gen_positions")
    return int(record["gen_positions"]) + int(record.get("tok_positions", 0))


def load_run(results_dir: str | Path, run_id: str) -> list[dict[str, Any]]:
    """Merged eval aggregates of one run in training order; raises
    ``FileNotFoundError`` without an ``eval/`` directory and ``ValueError``
    when only per-seed shards exist (``--merge`` first) or nothing is there.
    """
    directory = Path(results_dir) / run_id / "eval"
    if not directory.is_dir():
        raise FileNotFoundError(f"missing {directory}")
    records: list[dict[str, Any]] = []
    for path in sorted(directory.glob("step_*.json")):
        if not AGGREGATE_RE.match(path.name):
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("ppl_mean") is None:
            raise ValueError(f"{path} has no ppl_mean; run run_eval.py --merge first")
        records.append(record)
    if not records:
        raise ValueError(f"no merged eval aggregates under {directory}")
    records.sort(key=lambda record: x_position(record, "total_positions"))
    return records


def curve(records: list[dict[str, Any]],
          accounting: str = "total_positions") -> list[dict[str, Any]]:
    """Points of one run under one accounting, sorted by position."""
    points: list[dict[str, Any]] = []
    for record in records:
        point: dict[str, Any] = {
            "x": x_position(record, accounting),
            "ppl": float(record["ppl_mean"]),
            "ppl_std": float(record.get("ppl_std") or 0.0),
            "step": record.get("step"),
            "fraction": record.get("fraction"),
            "total_positions": x_position(record, "total_positions"),
            "total_positions_conservative": record.get("total_positions_conservative"),
        }
        for key in ("h_gpt2", "h_dataset", "unique_frac", "pairwise_agree"):
            value = record.get(f"{key}_mean")
            point[key] = None if value is None else float(value)
        points.append(point)
    points.sort(key=lambda point: point["x"])
    return points


def interpolate(points: list[dict[str, Any]], x: int) -> tuple[float, float] | None:
    """Linear ``(Gen PPL, ppl_std)`` of a curve at ``x``, or ``None`` outside.

    ``None`` covers an empty curve and any ``x`` outside the observed range:
    the caller then reports ``not comparable``.
    """
    if not points or x < points[0]["x"] or x > points[-1]["x"]:
        return None
    for left, right in zip(points, points[1:]):
        if left["x"] <= x <= right["x"]:
            span = right["x"] - left["x"]
            if span == 0:
                return (left["ppl"], left["ppl_std"])
            weight = (x - left["x"]) / span
            return (
                left["ppl"] + weight * (right["ppl"] - left["ppl"]),
                left["ppl_std"] + weight * (right["ppl_std"] - left["ppl_std"]),
            )
    return (points[-1]["ppl"], points[-1]["ppl_std"])


def observed_range(points: list[dict[str, Any]]) -> str:
    """``[low, high]`` of a curve's positions, for the not-comparable lines."""
    return f"[{points[0]['x']}, {points[-1]['x']}]" if points else "empty"


def milestone_label(point: dict[str, Any]) -> str:
    """``10%``-style milestone label, ``n/a`` without a fraction."""
    fraction = point.get("fraction")
    return "n/a" if fraction is None else f"{float(fraction) * 100:.0f}%"


def format_row(gate: str, run_id: str, point: dict[str, Any]) -> str:
    """One table row: identity, both position counts, then the eval metrics."""
    conservative = point["total_positions_conservative"]
    h_gpt2 = point["h_gpt2"]
    cells = [
        gate,
        run_id,
        milestone_label(point),
        str(point["total_positions"]),
        "-" if conservative is None else str(conservative),
        f"{point['ppl']:.2f} +/- {point['ppl_std']:.2f}",
        f"{point['ppl'] / REF_PPL:.3f}",
        _fmt(h_gpt2, 2),
        _fmt(None if h_gpt2 is None else h_gpt2 - REF_H, 3, signed=True),
        _fmt(point["h_dataset"], 2),
        _fmt(point["unique_frac"]),
        _fmt(point["pairwise_agree"]),
    ]
    return _row(cells)


def h_line(point: dict[str, Any]) -> tuple[str, bool]:
    """``|H - 4.03| <= 0.15`` verdict text and boolean for one milestone."""
    h_gpt2 = point["h_gpt2"]
    if h_gpt2 is None:
        return "H n/a FAIL", False
    delta = h_gpt2 - REF_H
    passed = abs(delta) <= G1_H_TOL
    verb = "PASS" if passed else "FAIL"
    return f"H {h_gpt2:.2f} ({delta:+.3f}) {verb} (|dH| <= {G1_H_TOL:.2f})", passed


def g0_verdict(records: list[dict[str, Any]], run_id: str) -> str:
    """G0: Gen PPL <= 20 and H >= 3.8 at every milestone."""
    print(f"G0 ({run_id}): Gen PPL <= {G0_MAX_PPL:.0f}, H(gpt2) >= {G0_MIN_H:.1f} "
          "at every milestone")
    passed = True
    for point in curve(records):
        ppl_ok = point["ppl"] <= G0_MAX_PPL
        h_ok = point["h_gpt2"] is not None and point["h_gpt2"] >= G0_MIN_H
        ok = ppl_ok and h_ok
        passed = passed and ok
        print(f"G0 step {point['step']} ({milestone_label(point)}): "
              f"{'PASS' if ok else 'FAIL'} "
              f"ppl {point['ppl']:.2f} {'<=' if ppl_ok else '>'} {G0_MAX_PPL:.0f}, "
              f"H {_fmt(point['h_gpt2'], 2)} {'>=' if h_ok else '<'} {G0_MIN_H:.1f}")
    verdict = "PASS" if passed else "FAIL"
    print(f"G0: {verdict}")
    return verdict


def g1_flow_verdict(g0_points: list[dict[str, Any]],
                    flow_points: list[dict[str, Any]],
                    g0_id: str, flow_id: str) -> str:
    """G1 flow: PPL ratio to G0 at equal positions, and H within 0.15."""
    print(f"G1 ({flow_id} vs {g0_id}): Gen PPL ratio <= {G1_PPL_RATIO:.2f} at "
          f"equal total_positions, |H - {REF_H:.2f}| <= {G1_H_TOL:.2f}")
    compared = 0
    passed = True
    for point in flow_points:
        h_text, h_ok = h_line(point)
        label = f"G1 {flow_id} step {point['step']} ({milestone_label(point)})"
        reference = interpolate(g0_points, point["x"])
        if reference is None:
            print(f"{label}: not comparable at total_positions {point['x']} "
                  f"(outside {g0_id} {observed_range(g0_points)}); {h_text}")
            passed = passed and h_ok
            continue
        ref_ppl, _ = reference
        ratio = point["ppl"] / ref_ppl
        ppl_ok = ratio <= G1_PPL_RATIO
        compared += 1
        passed = passed and ppl_ok and h_ok
        print(f"{label}: {'PASS' if ppl_ok and h_ok else 'FAIL'} "
              f"ppl {point['ppl']:.2f} vs {g0_id} {ref_ppl:.2f} (interpolated at "
              f"{point['x']}) ratio {ratio:.3f} "
              f"{'<=' if ppl_ok else '>'} {G1_PPL_RATIO:.3f}; {h_text}")
    if compared == 0:
        print(f"G1 ({flow_id}): not comparable (no milestone inside the "
              f"{g0_id} position range {observed_range(g0_points)})")
        return "not comparable"
    verdict = "PASS" if passed else "FAIL"
    print(f"G1 ({flow_id}): {verdict} ({compared} compared milestones)")
    return verdict


def g1_regress_report(regress_points: list[dict[str, Any]], regress_id: str) -> None:
    """Report noisy+regress without a verdict; FLAG high pairwise_agree."""
    print(f"G1 ({regress_id}): reported only, no pass/fail")
    flags = 0
    for point in regress_points:
        agree = point["pairwise_agree"]
        flagged = agree is not None and agree > FLAG_AGREE
        flags += int(flagged)
        note = " FLAG pairwise_agree > 0.5" if flagged else ""
        print(f"G1 {regress_id} step {point['step']} "
              f"({milestone_label(point)}): ppl {point['ppl']:.2f} "
              f"H {_fmt(point['h_gpt2'], 2)} "
              f"unique_frac {_fmt(point['unique_frac'])} "
              f"pairwise_agree {_fmt(agree)}{note}")
    print(f"G1 ({regress_id}): reported only ({flags} flagged milestones)")


def g2_verdict(jepa_id: str, jepa_points: list[dict[str, Any]],
               references: list[tuple[str, list[dict[str, Any]]]],
               accounting: str) -> str:
    """G2: JEPA matches or beats the best Phase-1 run at equal positions."""
    print(f"G2 (accounting: {accounting})")
    compared = 0
    passed = True
    for point in jepa_points:
        candidates = [
            (run_id, values[0], values[1])
            for run_id, ref_points in references
            if (values := interpolate(ref_points, point["x"])) is not None
        ]
        label = f"G2 {jepa_id} step {point['step']} ({milestone_label(point)})"
        if not candidates:
            print(f"{label}: not comparable at {accounting} {point['x']} "
                  "(outside every Phase-1 curve)")
            continue
        best_run, best_ppl, best_std = min(candidates, key=lambda item: item[1])
        limit = best_ppl + best_std
        ok = point["ppl"] <= limit
        compared += 1
        passed = passed and ok
        print(f"{label}: {'PASS' if ok else 'FAIL'} ppl {point['ppl']:.2f} vs "
              f"best Phase-1 {best_run} {best_ppl:.2f} +/- {best_std:.2f} "
              f"(limit {limit:.2f}) at {accounting} {point['x']}")
    if compared == 0:
        print(f"G2 (accounting: {accounting}): not comparable "
              "(no JEPA milestone inside a Phase-1 range)")
        return "not comparable"
    verdict = "PASS" if passed else "FAIL"
    print(f"G2 (accounting: {accounting}): {verdict} ({compared} compared)")
    return verdict


def _combine(verdicts: list[str]) -> str:
    """One verdict for a list of arms: FAIL if any fails, else PASS if any."""
    if "FAIL" in verdicts:
        return "FAIL"
    return "PASS" if "PASS" in verdicts else "not comparable"


def report(results_dir: str | Path, g0_id: str, flow_id: str,
           regress_id: str, jepa_ids: list[str]) -> int:
    """Print the table and every gate verdict; returns the process exit code."""
    entries = [("G0", g0_id), ("G1", flow_id), ("G1", regress_id)]
    entries += [("G2", jepa_id) for jepa_id in jepa_ids]
    loaded: dict[str, list[dict[str, Any]] | None] = {}
    incomplete = False
    print("== Eval table (merged eval/step_<n>.json; PPL is mean +/- std over seeds)")
    print(_row([name for name, _, _ in COLUMNS]))
    for gate, run_id in entries:
        if run_id not in loaded:
            try:
                loaded[run_id] = load_run(results_dir, run_id)
            except (FileNotFoundError, ValueError) as exc:
                print(f"{gate} {run_id}: MISSING ({exc})")
                loaded[run_id] = None
                incomplete = True
        records = loaded[run_id]
        for point in curve(records) if records is not None else []:
            print(format_row(gate, run_id, point))

    verdicts: list[str] = []
    print("\n== Gate verdicts")
    g0_records = loaded[g0_id]
    flow_records = loaded[flow_id]
    regress_records = loaded[regress_id]
    g0_points = curve(g0_records) if g0_records is not None else []
    verdicts.append("MISSING" if g0_records is None else g0_verdict(g0_records, g0_id))
    if flow_records is None:
        print(f"G1 ({flow_id}): MISSING")
        verdicts.append("MISSING")
    else:
        verdicts.append(g1_flow_verdict(g0_points, curve(flow_records), g0_id, flow_id))
    if regress_records is None:
        print(f"G1 ({regress_id}): MISSING")
    else:
        g1_regress_report(curve(regress_records), regress_id)

    phase1 = [
        (run_id, records)
        for run_id in (flow_id, regress_id)
        if (records := loaded[run_id]) is not None
    ]
    accounting_verdicts: dict[str, list[str]] = {}
    for jepa_id in jepa_ids:
        jepa_records = loaded[jepa_id]
        if jepa_records is None:
            print(f"G2 ({jepa_id}): MISSING")
            verdicts.append("MISSING")
            continue
        for accounting in ACCOUNTINGS:
            references = [
                (run_id, curve(records, accounting)) for run_id, records in phase1
            ]
            verdict = g2_verdict(
                jepa_id, curve(jepa_records, accounting), references, accounting
            )
            accounting_verdicts.setdefault(accounting, []).append(verdict)
            verdicts.append(verdict)
    if accounting_verdicts:
        parts = [
            f"{accounting} {_combine(accounting_verdicts[accounting])}"
            for accounting in ACCOUNTINGS
        ]
        print("G2: " + ", ".join(parts))

    failed = incomplete or "MISSING" in verdicts or "FAIL" in verdicts
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--g0", default=DEFAULT_G0, help="Phase 0 vq+ce run id")
    parser.add_argument("--flow", default=DEFAULT_FLOW, help="Phase 1 flow run id")
    parser.add_argument("--regress", default=DEFAULT_REGRESS,
                        help="Phase 1 regress run id")
    parser.add_argument("--jepa", nargs="+", default=[DEFAULT_JEPA],
                        help="joint JEPA run id(s) judged vs the best Phase 1 run")
    args = parser.parse_args(argv)
    return report(args.results, args.g0, args.flow, args.regress, args.jepa)


if __name__ == "__main__":
    raise SystemExit(main())

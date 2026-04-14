#!/usr/bin/env python3
"""
Aggregate statistics across multiple ``run_evaluation.py`` output directories.

Each run should contain ``aggregate_results.json`` (e.g. ``.../run_01/aggregate_results.json``).
Writes a JSON summary with mean, sample standard deviation, min, and max per metric.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _stdev(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(statistics.stdev(values))


def _collect_numeric_fields(
    obj: Dict[str, Any], skip_keys: Tuple[str, ...] = ("label",)
) -> List[str]:
    keys: List[str] = []
    for k, v in obj.items():
        if k in skip_keys:
            continue
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            keys.append(k)
    return sorted(keys)


def find_aggregate_files(base: Path) -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = []
    if not base.is_dir():
        return out
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        agg = child / "aggregate_results.json"
        if agg.is_file():
            out.append((child.name, agg))
    return out


def load_aggregates(
    base: Path,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Return (run_names, parsed JSON list) for each run with aggregate_results.json."""
    runs: List[str] = []
    data: List[Dict[str, Any]] = []
    for name, path in find_aggregate_files(base):
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        runs.append(name)
        data.append(payload)
    return runs, data


def summarize_runs(
    payloads: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not payloads:
        return {
            "num_runs": 0,
            "conditions": {},
            "avg_ranks": {},
        }

    # Union of condition keys and dimension keys from first payload with aggregate_scores
    first_scores: Optional[Dict[str, Any]] = None
    for p in payloads:
        agg = p.get("aggregate_scores")
        if isinstance(agg, dict) and agg:
            first_scores = agg
            break
    if not first_scores:
        return {
            "num_runs": len(payloads),
            "conditions": {},
            "avg_ranks": {},
            "warning": "No aggregate_scores found in any run (all failed or empty).",
        }

    condition_keys = list(first_scores.keys())
    summary_conditions: Dict[str, Any] = {}

    for ck in condition_keys:
        per_field: Dict[str, Any] = {}
        # Label from first non-missing
        label: Optional[str] = None
        for p in payloads:
            agg = p.get("aggregate_scores") or {}
            block = agg.get(ck)
            if isinstance(block, dict) and "label" in block:
                label = block.get("label")
                break
        if label is not None:
            per_field["label"] = label

        field_names: Optional[List[str]] = None
        for p in payloads:
            agg = p.get("aggregate_scores") or {}
            block = agg.get(ck)
            if not isinstance(block, dict):
                continue
            field_names = _collect_numeric_fields(block)
            if field_names:
                break
        if not field_names:
            summary_conditions[ck] = per_field
            continue

        for field in field_names:
            values: List[float] = []
            for p in payloads:
                agg = p.get("aggregate_scores") or {}
                block = agg.get(ck)
                if not isinstance(block, dict):
                    continue
                v = block.get(field)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    values.append(float(v))
            if not values:
                continue
            per_field[field] = {
                "mean": float(statistics.mean(values)),
                "std": _stdev(values),
                "min": float(min(values)),
                "max": float(max(values)),
                "values": values,
            }
        summary_conditions[ck] = per_field

    # avg_ranks across runs
    rank_keys: Optional[List[str]] = None
    for p in payloads:
        ar = p.get("avg_ranks")
        if isinstance(ar, dict) and ar:
            rank_keys = sorted(ar.keys())
            break
    summary_ranks: Dict[str, Any] = {}
    if rank_keys:
        for rk in rank_keys:
            values: List[float] = []
            for p in payloads:
                ar = p.get("avg_ranks") or {}
                v = ar.get(rk)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    values.append(float(v))
            if not values:
                continue
            summary_ranks[rk] = {
                "mean": float(statistics.mean(values)),
                "std": _stdev(values),
                "min": float(min(values)),
                "max": float(max(values)),
                "values": values,
            }

    out: Dict[str, Any] = {
        "num_runs": len(payloads),
        "conditions": summary_conditions,
        "avg_ranks": summary_ranks,
    }
    return out


def print_summary_table(summary: Dict[str, Any]) -> None:
    cond = summary.get("conditions") or {}
    if not cond:
        print("No condition summaries to print.")
        return

    # Prefer weighted_avg for a compact table
    print("\n" + "=" * 72)
    print("WEIGHTED AVERAGE (mean ± std across runs)")
    print("=" * 72)
    print(f"{'Condition':<40} {'mean':>8} {'std':>8} {'min':>8} {'max':>8}")
    print("-" * 72)
    for ck in sorted(cond.keys()):
        block = cond[ck] or {}
        label = block.get("label", ck)
        wt = block.get("weighted_avg")
        if not isinstance(wt, dict):
            continue
        m = wt.get("mean", 0.0)
        s = wt.get("std", 0.0)
        mn = wt.get("min", 0.0)
        mx = wt.get("max", 0.0)
        print(f"{str(label)[:40]:<40} {m:>8.3f} {s:>8.3f} {mn:>8.3f} {mx:>8.3f}")

    ranks = summary.get("avg_ranks") or {}
    if ranks:
        print("\n" + "=" * 72)
        print("AVERAGE RANK (mean ± std; lower is better)")
        print("=" * 72)
        print(f"{'Condition':<40} {'mean':>8} {'std':>8}")
        print("-" * 72)
        for rk in sorted(ranks.keys()):
            rr = ranks[rk]
            # find label from conditions
            label = (cond.get(rk) or {}).get("label", rk)
            m = rr.get("mean", 0.0)
            s = rr.get("std", 0.0)
            print(f"{str(label)[:40]:<40} {m:>8.3f} {s:>8.3f}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize repeated run_evaluation.py outputs (mean, std, min, max)."
    )
    parser.add_argument(
        "runs_dir",
        type=Path,
        help="Directory containing run_* subfolders with aggregate_results.json",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write summary JSON to this path (default: <runs_dir>/summary.json)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Do not print the text table",
    )
    args = parser.parse_args()

    runs_dir = args.runs_dir.resolve()
    run_names, payloads = load_aggregates(runs_dir)
    if not payloads:
        print(f"ERROR: No aggregate_results.json found under subdirectories of {runs_dir}", file=sys.stderr)
        sys.exit(1)

    summary = summarize_runs(payloads)
    summary["run_names"] = run_names
    summary["runs_directory"] = str(runs_dir)

    out_path = args.output
    if out_path is None:
        out_path = runs_dir / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if not args.quiet:
        print_summary_table(summary)
    print(f"Wrote summary: {out_path}")


if __name__ == "__main__":
    main()

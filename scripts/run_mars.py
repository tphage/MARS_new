#!/usr/bin/env python3
"""Run the full MARS pipeline for one or more benchmark queries.

Usage:
    python scripts/run_mars.py                        # all queries
    python scripts/run_mars.py --queries Query1,Query2  # specific subset
    python scripts/run_mars.py --output-dir results    # custom output root
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.utils.ablation_utils import load_ablation_queries
from src.runner import initialize, run_query


def main():
    parser = argparse.ArgumentParser(description="Run the full MARS pipeline")
    parser.add_argument(
        "--queries", default=None,
        help="Comma-separated query names (default: all queries in config/queries.yaml)",
    )
    parser.add_argument(
        "--output-dir", default="results",
        help="Root output directory (default: results)",
    )
    args = parser.parse_args()

    config = load_config()
    queries = load_ablation_queries()

    if args.queries:
        selected = {q.strip() for q in args.queries.split(",")}
        queries = [q for q in queries if q["name"] in selected]
        if not queries:
            print(f"ERROR: None of the specified queries found: {args.queries}")
            available = [q["name"] for q in load_ablation_queries()]
            print(f"  Available: {', '.join(available)}")
            sys.exit(1)

    print(f"MARS Full Pipeline")
    print(f"Queries: {', '.join(q['name'] for q in queries)}")
    print(f"Output:  {args.output_dir}/")
    print()

    components = initialize(config)
    print()

    for i, query in enumerate(queries, 1):
        name = query["name"]
        output_dir = str(PROJECT_ROOT / args.output_dir / name)
        print(f"\n{'='*70}")
        print(f"[{i}/{len(queries)}] {name}")
        print(f"{'='*70}")
        run_query(components, query, output_dir)

    print(f"\nAll {len(queries)} queries complete.")


if __name__ == "__main__":
    main()

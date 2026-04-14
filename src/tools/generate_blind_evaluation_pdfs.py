"""
Generate blind expert-evaluation PDFs for all pipeline conditions across all
query directories.

For each query directory the script locates the four JSON files (full MARS
pipeline + three ablation conditions), randomly assigns neutral labels
("Response A" .. "Response D"), renders a blind LaTeX PDF for each, and writes
a key file that maps labels back to conditions.

**Legacy layout** (``pipeline_logs_Query*`` at ``--queries-dir``):

    python -m src.tools.generate_blind_evaluation_pdfs --layout legacy \
        [--queries-dir .] [--output-dir evaluation_results/blind_pdfs] [--seed 42]

**Results layout** (``results/<QueryName>/`` with ``mars.json`` and ablation JSONs):

    python -m src.tools.generate_blind_evaluation_pdfs --layout results \
        [--results-root results] [--only-queries Query1,Query2] \
        [--output-dir results/blind_pdfs] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

from src.tools.generate_evaluation_latex_pdf import (
    load_evaluation,
    render_pdf_with_latex,
)

CONDITION_LABELS = ["Response A", "Response B", "Response C", "Response D"]

CONDITION_NAMES = {
    "full_mars": "full_mars",
    "1agent_no_rag": "1agent_no_rag",
    "1agent_rag": "1agent_rag",
    "3agent": "3agent",
}


def _find_query_dirs(root: Path) -> List[Path]:
    """Return sorted ``pipeline_logs_Query*`` directories under *root*."""
    dirs = sorted(
        p for p in root.iterdir()
        if p.is_dir() and re.match(r"pipeline_logs_Query\d+", p.name)
    )
    return dirs


def _find_json(directory: Path, prefix: str) -> Optional[Path]:
    """Return the first JSON file matching ``<prefix>*.json`` in *directory*."""
    matches = sorted(directory.glob(f"{prefix}*.json"))
    return matches[0] if matches else None


def _collect_conditions(query_dir: Path) -> Dict[str, Path]:
    """Map condition names to their JSON file paths for one query directory."""
    conditions: Dict[str, Path] = {}

    evaluation = _find_json(query_dir, "evaluation_")
    if evaluation:
        conditions["full_mars"] = evaluation

    for ablation_key in ("1agent_no_rag", "1agent_rag", "3agent"):
        path = _find_json(query_dir, f"ablation_{ablation_key}_")
        if path:
            conditions[ablation_key] = path

    return conditions


def _find_condition_file(query_dir: Path, exact_name: str, prefix: str) -> Path:
    """Prefer *exact_name*; else newest ``prefix*.json`` in *query_dir* (like run_evaluation)."""
    exact = query_dir / exact_name
    if exact.is_file():
        return exact
    candidates = sorted(
        [
            f
            for f in query_dir.iterdir()
            if f.is_file() and f.name.startswith(prefix) and f.suffix == ".json"
        ],
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No {exact_name} or {prefix}*.json in {query_dir}"
        )
    return candidates[0]


def _find_mars_or_evaluation_json(query_dir: Path) -> Optional[Path]:
    """Resolve MARS baseline JSON: mars.json, then evaluation_*.json, then artifacts/evaluation_*.json."""
    mars = query_dir / "mars.json"
    if mars.is_file():
        return mars
    try:
        return _find_condition_file(query_dir, "mars.json", "evaluation_")
    except FileNotFoundError:
        pass
    art = query_dir / "artifacts"
    if art.is_dir():
        matches = sorted(
            art.glob("evaluation_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if matches:
            return matches[0]
    return None


def _collect_conditions_results(query_dir: Path) -> Dict[str, Path]:
    """
    Map condition keys to JSON paths under ``results/<QueryName>/`` (same keys as legacy).
    """
    conditions: Dict[str, Path] = {}
    mars = _find_mars_or_evaluation_json(query_dir)
    if mars:
        conditions["full_mars"] = mars

    ablation_specs = (
        ("ablation_3agent.json", "ablation_3agent_", "3agent"),
        ("ablation_1agent_rag.json", "ablation_1agent_rag_", "1agent_rag"),
        ("ablation_1agent_no_rag.json", "ablation_1agent_no_rag_", "1agent_no_rag"),
    )
    for exact, prefix, key in ablation_specs:
        try:
            conditions[key] = _find_condition_file(query_dir, exact, prefix)
        except FileNotFoundError:
            continue

    return conditions


def _results_query_dir_complete(query_dir: Path) -> bool:
    return len(_collect_conditions_results(query_dir)) == 4


def _find_results_query_dirs(
    results_root: Path, only_names: Optional[Set[str]]
) -> List[Path]:
    """Subdirectories of *results_root* that contain all four condition files."""
    dirs: List[Path] = []
    if not results_root.is_dir():
        raise SystemExit(f"Results root is not a directory: {results_root}")
    for p in sorted(results_root.iterdir()):
        if not p.is_dir() or p.name == "evaluation":
            continue
        if only_names is not None and p.name not in only_names:
            continue
        if _results_query_dir_complete(p):
            dirs.append(p)
    return dirs


def _render_blind_for_query(
    query_name: str,
    conditions: Dict[str, Path],
    rng: random.Random,
    output_dir: Path,
    key: Dict[str, Dict[str, str]],
) -> None:
    condition_keys = list(conditions.keys())
    rng.shuffle(condition_keys)

    labels = CONDITION_LABELS[: len(condition_keys)]
    mapping: Dict[str, str] = {}

    out_subdir = output_dir / query_name
    out_subdir.mkdir(parents=True, exist_ok=True)

    for label, cond_name in zip(labels, condition_keys):
        json_path = conditions[cond_name]
        evaluation = load_evaluation(json_path)

        pdf_name = label.replace(" ", "_") + ".pdf"
        pdf_path = out_subdir / pdf_name

        render_pdf_with_latex(
            evaluation,
            output_path=pdf_path,
            label=label,
            blind=True,
            skip_rejected=True,
            skip_hard_constraints=True,
        )
        mapping[label] = cond_name
        print(f"  {query_name}/{pdf_name}  <-  {cond_name}")

    key[query_name] = mapping


def generate_blind_pdfs(
    queries_dir: Path,
    output_dir: Path,
    seed: Optional[int] = None,
) -> Path:
    """
    Generate blind PDFs for every legacy ``pipeline_logs_Query*`` directory and write a key file.

    Returns the path to the key file.
    """
    rng = random.Random(seed)
    query_dirs = _find_query_dirs(queries_dir)
    if not query_dirs:
        raise SystemExit(
            f"No pipeline_logs_Query* directories found under {queries_dir}"
        )

    key: Dict[str, Dict[str, str]] = {}

    for qdir in query_dirs:
        query_name = qdir.name.replace("pipeline_logs_", "")
        conditions = _collect_conditions(qdir)

        if not conditions:
            print(f"  [skip] {query_name}: no JSON files found")
            continue

        _render_blind_for_query(query_name, conditions, rng, output_dir, key)

    return _write_blind_key(output_dir, key)


def generate_blind_pdfs_results(
    results_root: Path,
    output_dir: Path,
    seed: Optional[int] = None,
    only_queries: Optional[List[str]] = None,
) -> Path:
    """
    Generate blind PDFs for each ``results/<QueryName>/`` directory that has all
    four JSON exports. If *only_queries* is set, require those names to exist and
    be complete, or exit with an error.
    """
    rng = random.Random(seed)
    only_set: Optional[Set[str]] = set(only_queries) if only_queries else None
    query_dirs = _find_results_query_dirs(results_root, only_set)

    if only_queries:
        missing = set(only_queries) - {p.name for p in query_dirs}
        for name in sorted(missing):
            qpath = results_root / name
            if not qpath.is_dir():
                raise SystemExit(f"Query directory not found: {qpath}")
            if not _results_query_dir_complete(qpath):
                raise SystemExit(
                    f"Query {name!r} is missing one or more condition JSON files "
                    f"(need mars.json or evaluation export + three ablation_*.json) under {qpath}"
                )

    if not query_dirs:
        scope = f"matching {sorted(only_queries)!r} " if only_queries else ""
        raise SystemExit(
            f"No complete query directories {scope}under {results_root} "
            "(need all four: MARS baseline + three ablations)."
        )

    key: Dict[str, Dict[str, str]] = {}

    for qdir in query_dirs:
        query_name = qdir.name
        conditions = _collect_conditions_results(qdir)
        _render_blind_for_query(query_name, conditions, rng, output_dir, key)

    return _write_blind_key(output_dir, key)


def _write_blind_key(
    output_dir: Path, key: Dict[str, Dict[str, str]]
) -> Path:
    key_path = output_dir / "blind_key.json"
    key_path.parent.mkdir(parents=True, exist_ok=True)
    with key_path.open("w", encoding="utf-8") as f:
        json.dump(key, f, indent=2, ensure_ascii=False)
    print(f"\nKey file written to: {key_path}")
    return key_path


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate blind expert-evaluation PDFs for all pipeline conditions "
            "across all query directories."
        ),
    )
    parser.add_argument(
        "--layout",
        choices=("legacy", "results"),
        default="legacy",
        help=(
            "Directory layout: legacy = pipeline_logs_Query* under --queries-dir; "
            "results = results/<QueryName>/ under --results-root (default: legacy)."
        ),
    )
    parser.add_argument(
        "--queries-dir",
        default=".",
        help=(
            "Root directory containing pipeline_logs_Query* folders "
            "(--layout legacy only; default: current directory)"
        ),
    )
    parser.add_argument(
        "--results-root",
        default="results",
        help=(
            "Directory containing Query* folders (--layout results only; default: results)"
        ),
    )
    parser.add_argument(
        "--only-queries",
        default=None,
        help=(
            "Comma-separated query folder names to include, e.g. Query1,Query2 "
            "(--layout results only). Each must exist and have all four JSON exports."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation_results/blind_pdfs",
        help="Directory to write blind PDFs and key file (default: evaluation_results/blind_pdfs)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible label assignment",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    output_dir = Path(args.output_dir).resolve()

    if args.layout == "legacy":
        queries_dir = Path(args.queries_dir).resolve()
        print(f"Layout:  legacy")
        print(f"Scanning: {queries_dir}")
        print(f"Output:   {output_dir}\n")
        generate_blind_pdfs(queries_dir, output_dir, seed=args.seed)
        return

    results_root = Path(args.results_root).resolve()
    only_list: Optional[List[str]] = None
    if args.only_queries:
        only_list = [x.strip() for x in args.only_queries.split(",") if x.strip()]

    print(f"Layout:  results")
    print(f"Scanning: {results_root}")
    print(f"Output:   {output_dir}\n")

    generate_blind_pdfs_results(
        results_root, output_dir, seed=args.seed, only_queries=only_list
    )


if __name__ == "__main__":
    main()

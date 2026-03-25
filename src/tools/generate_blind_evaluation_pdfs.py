"""
Generate blind expert-evaluation PDFs for all pipeline conditions across all
query directories.

For each ``pipeline_logs_Query*`` directory the script locates the four JSON
files (full MARS pipeline + three ablation conditions), randomly assigns
neutral labels ("Response A" .. "Response D"), renders a blind LaTeX PDF for
each, and writes a key file that maps labels back to conditions.

CLI usage:

    python -m src.tools.generate_blind_evaluation_pdfs [--queries-dir .] \
        [--output-dir evaluation_results/blind_pdfs] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional

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


def generate_blind_pdfs(
    queries_dir: Path,
    output_dir: Path,
    seed: Optional[int] = None,
) -> Path:
    """
    Generate blind PDFs for every query directory and write a key file.

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
        "--queries-dir",
        default=".",
        help=(
            "Root directory containing pipeline_logs_Query* folders "
            "(default: current directory)"
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
    queries_dir = Path(args.queries_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    print(f"Scanning: {queries_dir}")
    print(f"Output:   {output_dir}\n")

    generate_blind_pdfs(queries_dir, output_dir, seed=args.seed)


if __name__ == "__main__":
    main()

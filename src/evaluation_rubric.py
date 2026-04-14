"""Shared loading and helpers for ``config/evaluation_rubric.yaml``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

DEFAULT_RELATIVE_PATH = Path("config") / "evaluation_rubric.yaml"


def default_evaluation_rubric_path() -> Path:
    """Project root ``config/evaluation_rubric.yaml``."""
    return Path(__file__).resolve().parents[1] / DEFAULT_RELATIVE_PATH


def load_evaluation_rubric(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load the judge / expert rubric YAML."""
    rubric_path = path or default_evaluation_rubric_path()
    with rubric_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "dimensions" not in data:
        raise ValueError(f"Invalid evaluation rubric YAML: {rubric_path}")
    return data


def dimension_keys_in_order(rubric: Dict[str, Any]) -> List[str]:
    return list(rubric["dimensions"].keys())


def iter_subsystem_dimensions(
    rubric: Dict[str, Any], subsystem: int
) -> List[Tuple[str, Dict[str, Any]]]:
    """Dimensions for System *subsystem* (1, 2, or 3), in YAML order."""
    prefix = f"system{subsystem}_"
    return [
        (k, v)
        for k, v in rubric["dimensions"].items()
        if k.startswith(prefix)
    ]


def subsystem_criterion_labels(rubric: Dict[str, Any], subsystem: int) -> List[str]:
    """Short labels for PDF bullet lists (``criterion_label`` field)."""
    out: List[str] = []
    for _k, meta in iter_subsystem_dimensions(rubric, subsystem):
        lbl = meta.get("criterion_label")
        if not lbl:
            raise KeyError(
                f"Dimension missing criterion_label: {_k} in evaluation_rubric.yaml"
            )
        out.append(str(lbl))
    return out


def ordinal_scale_lines(rubric: Dict[str, Any]) -> List[str]:
    lines = rubric.get("ordinal_scale_lines")
    if not lines:
        return []
    return [str(x) for x in lines]


def rubric_column_header(dim_meta: Dict[str, Any]) -> str:
    """Compact header for CLI tables."""
    if dim_meta.get("short_name"):
        return str(dim_meta["short_name"])[:14]
    name = str(dim_meta.get("name", ""))
    return name[:14]

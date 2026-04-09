"""Shared utilities for MARS ablation study notebooks.

Provides JSON response parsing, evaluation payload construction, and query loading.
"""

import json
import os
import re
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_ablation_queries(config_path: Optional[str] = None) -> List[Dict[str, str]]:
    """Load benchmark queries from config/queries.yaml."""
    if config_path is None:
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "config" / "queries.yaml"
        if not config_path.exists():
            config_path = project_root / "config" / "ablation_queries.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["queries"]


def extract_json_from_response(text: str) -> Optional[Dict[str, Any]]:
    """Extract a JSON object from an LLM response, handling markdown fences.

    Tries in order:
    1. Extract from ```json ... ``` fenced block
    2. Extract from ``` ... ``` fenced block
    3. Find the outermost { ... } in the raw text
    4. Direct json.loads on the full text
    """
    if not text or not text.strip():
        return None

    # 1. ```json ... ```
    match = re.search(r"```json\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 2. ``` ... ```
    match = re.search(r"```\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 3. Outermost { ... }
    first_brace = text.find("{")
    if first_brace != -1:
        last_brace = text.rfind("}")
        if last_brace > first_brace:
            try:
                return json.loads(text[first_brace : last_brace + 1])
            except json.JSONDecodeError:
                pass

    # 4. Direct parse
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None


def build_ablation_evaluation(
    query: Dict[str, str],
    properties: List[str],
    constraints: List[str],
    candidate: Optional[Dict[str, Any]],
    manufacturing: Dict[str, Any],
    condition_name: str,
    run_id: str,
    duration_seconds: float = 0.0,
    raw_responses: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Build an evaluation payload matching the MARS evaluation_*.json format.

    Args:
        query: Dict with sentence, material_X, application_Y.
        properties: List of extracted property strings.
        constraints: List of extracted constraint strings.
        candidate: Dict with material_name, material_class, justification (or None).
        manufacturing: Dict with status, process_recipe, blocking_constraints, feedback_to_system2.
        condition_name: Ablation condition identifier (e.g. "3agent", "1agent_rag", "1agent_no_rag").
        run_id: Run identifier string.
        duration_seconds: Total wall-clock seconds for the ablation run.
        raw_responses: Optional dict mapping step names to raw LLM response text.
    """
    candidate_entry = None
    if candidate:
        candidate_entry = {
            "material_name": candidate.get("material_name"),
            "material_class": candidate.get("material_class"),
            "material_id": candidate.get("material_id"),
            "justification": candidate.get("justification"),
            "properties": candidate.get("properties"),
        }

    mfg = {
        "status": manufacturing.get("status", "unknown"),
        "process_recipe": manufacturing.get("process_recipe"),
        "blocking_constraints": manufacturing.get("blocking_constraints", []),
        "feedback_to_system2": manufacturing.get("feedback_to_system2", ""),
    }

    payload = {
        "query": {
            "sentence": query.get("sentence", ""),
            "material_X": query.get("material_X", ""),
            "application_Y": query.get("application_Y", ""),
        },
        "required_material_properties": {
            "properties": properties,
            "constraints": constraints,
        },
        "candidate_selection": {
            "final_candidate": candidate_entry,
            "rejected_candidates": [],
        },
        "manufacturing_process": mfg,
        "metadata": {
            "pipeline_run_id": run_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "final_outcome_status": manufacturing.get("status", "unknown"),
            "total_iterations": 1,
            "total_rejected_candidates": 0,
            "ablation_condition": condition_name,
            "duration_seconds": duration_seconds,
        },
    }

    if raw_responses:
        payload["raw_responses"] = raw_responses

    return payload


def save_ablation_result(
    payload: Dict[str, Any],
    output_dir: str,
    condition_name: str,
    run_id: str,
) -> str:
    """Save an ablation evaluation payload to disk.

    Returns the path to the saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = f"ablation_{condition_name}_{run_id}.json"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    return filepath


def format_rag_results_for_prompt(
    rag_results: List[Dict[str, Any]],
    source_name: str,
    max_chars_per_result: int = 3000,
) -> str:
    """Format RAG results into a text block suitable for inclusion in an LLM prompt."""
    if not rag_results:
        return f"[{source_name}]: No results retrieved.\n"

    lines = [f"--- {source_name} ({len(rag_results)} documents) ---"]
    for i, doc in enumerate(rag_results, 1):
        content = doc.get("content", "")
        if len(content) > max_chars_per_result:
            content = content[:max_chars_per_result] + "..."
        lines.append(f"[{source_name} Doc {i}]:\n{content}\n")
    return "\n".join(lines)


def format_kg_results_for_prompt(
    kg_result: Dict[str, Any],
    kg_name: str,
) -> str:
    """Format knowledge graph connection results into a text block for an LLM prompt."""
    if not kg_result:
        return f"[{kg_name} KG]: No connections found.\n"

    lines = [f"--- {kg_name} Knowledge Graph ---"]

    summary = kg_result.get("summary") or kg_result.get("connections_text", "")
    if summary:
        if len(summary) > 5000:
            summary = summary[:5000] + "..."
        lines.append(f"Connection summary:\n{summary}\n")

    found_paths = kg_result.get("found_paths", [])
    if found_paths:
        lines.append(f"Found {len(found_paths)} paths between keywords.")

    matched = kg_result.get("matched_node_ids", [])
    if matched:
        preview = matched[:20]
        lines.append(f"Matched nodes ({len(matched)} total): {', '.join(str(n) for n in preview)}")

    return "\n".join(lines)

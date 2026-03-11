"""Evaluation export utility for SG_MIT_EvalPlan framework.

Builds a self-contained, evaluation-ready JSON payload from pipeline_run
and saved system outputs. Used to create evaluation documents for material scientists.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    """Load JSON file if it exists."""
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _merge_rejected_candidates(
    iteration_history: List[Dict[str, Any]],
    tracker_details: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge rejection sources: iteration_history (System 2) + tracker (System 2 + System 3)."""
    # Build map from iteration_history for System 2 feasibility rejections (richer reasoning)
    hist_by_candidate: Dict[str, Dict[str, Any]] = {}
    for h in iteration_history or []:
        if h.get("feasible") is False and h.get("candidate"):
            c = str(h.get("candidate", "")).strip()
            if c:
                hist_by_candidate[c.lower()] = {
                    "candidate": c,
                    "reasoning": h.get("reasoning", ""),
                    "constraints_violated": h.get("constraints_violated", []),
                    "source": "feasibility",
                }

    # Use tracker as primary; enrich with iteration_history when available
    seen: set = set()
    merged: List[Dict[str, Any]] = []
    for entry in tracker_details or []:
        c = str(entry.get("candidate", "")).strip()
        if not c or c.lower() in seen:
            continue
        seen.add(c.lower())
        source = entry.get("source", "feasibility")
        hist = hist_by_candidate.get(c.lower())
        if hist and source != "manufacturability":
            merged.append(hist)
        else:
            merged.append({
                "candidate": c,
                "reasoning": entry.get("reason", ""),
                "constraints_violated": entry.get("constraints", []),
                "source": source or "feasibility",
            })
    return merged


def build_evaluation_payload(
    pipeline_run: Dict[str, Any],
    output_dir: str,
) -> Dict[str, Any]:
    """
    Build a self-contained evaluation payload from pipeline_run and loaded system outputs.

    Args:
        pipeline_run: The pipeline_run dict (after completion).
        output_dir: Directory containing system1/2/3 JSON files (typically pipeline_logs).

    Returns:
        Evaluation-ready dict with query, required_material_properties,
        candidate_selection, manufacturing_process, and metadata.
    """
    pipeline_run_id = pipeline_run.get("pipeline_run_id", "unknown")
    final_outcome = pipeline_run.get("final_outcome", {})
    system1 = pipeline_run.get("system1", {})
    loop_data = pipeline_run.get("system2_system3_loop", {})
    iterations = loop_data.get("iterations", [])

    # Load System 1 output
    system1_data: Optional[Dict[str, Any]] = None
    s1_path = system1.get("result_path")
    if s1_path:
        system1_data = _load_json(s1_path)
    if not system1_data and output_dir:
        s1_alt = os.path.join(output_dir, f"system1_{system1.get('run_id', '')}.json")
        system1_data = _load_json(s1_alt)

    # Query and required material properties from System 1
    query = {
        "sentence": "",
        "material_X": "",
        "application_Y": "",
    }
    required_material_properties = {
        "properties": [],
        "constraints": [],
    }
    if system1_data:
        query["sentence"] = system1_data.get("sentence", "")
        query["material_X"] = system1_data.get("material_X", "")
        query["application_Y"] = system1_data.get("application_Y", "")
        props = system1_data.get("properties_W", {})
        required_material_properties["properties"] = props.get("required", []) or system1_data.get("extracted_keywords", [])
        required_material_properties["constraints"] = system1_data.get("extracted_constraints", [])

    # Candidate selection and manufacturing process from last iteration
    candidate_selection: Dict[str, Any] = {
        "final_candidate": None,
        "rejected_candidates": [],
    }
    manufacturing_process: Dict[str, Any] = {
        "status": "not_run",
        "process_recipe": None,
        "blocking_constraints": [],
        "feedback_to_system2": "",
    }

    if iterations:
        last_iter = iterations[-1]
        s2_path = last_iter.get("system2", {}).get("result_path")
        s3_path = last_iter.get("system3", {}).get("result_path")

        system2_data = _load_json(s2_path) if s2_path else None
        system3_data = _load_json(s3_path) if s3_path else None

        # Candidate selection from System 2
        if system2_data:
            candidate = system2_data.get("candidate")
            if candidate:
                candidate_selection["final_candidate"] = {
                    "material_name": candidate.get("material_name"),
                    "material_class": candidate.get("material_class"),
                    "material_id": candidate.get("material_id"),
                    "justification": candidate.get("justification"),
                    "properties": candidate.get("properties"),
                }
            # Merge rejected candidates from iteration_history + tracker
            iteration_history = system2_data.get("iteration_history", [])
            tracker_path = os.path.join(output_dir, "rejected_candidates.json")
            tracker_data = _load_json(tracker_path)
            tracker_details = (tracker_data or {}).get("rejected_candidates", [])
            candidate_selection["rejected_candidates"] = _merge_rejected_candidates(
                iteration_history, tracker_details
            )

        # Manufacturing process from System 3
        if system3_data:
            manufacturing_process["status"] = system3_data.get("status", "unknown")
            manufacturing_process["process_recipe"] = system3_data.get("process_recipe")
            manufacturing_process["blocking_constraints"] = system3_data.get("blocking_constraints", [])
            manufacturing_process["feedback_to_system2"] = system3_data.get("feedback_to_system2", "")

    metadata = {
        "pipeline_run_id": pipeline_run_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "final_outcome_status": final_outcome.get("status"),
        "total_iterations": loop_data.get("total_iterations", 0),
        "total_rejected_candidates": final_outcome.get("total_rejected_candidates", 0),
    }

    return {
        "query": query,
        "required_material_properties": required_material_properties,
        "candidate_selection": candidate_selection,
        "manufacturing_process": manufacturing_process,
        "metadata": metadata,
    }


def save_evaluation_export(
    pipeline_run: Dict[str, Any],
    output_dir: str,
) -> str:
    """
    Build and save the evaluation payload to pipeline_logs.

    Args:
        pipeline_run: The pipeline_run dict (after completion).
        output_dir: Directory to save the file (typically pipeline_logs).

    Returns:
        Path to the saved file, or empty string on failure.
    """
    try:
        payload = build_evaluation_payload(pipeline_run, output_dir)
        pipeline_run_id = pipeline_run.get("pipeline_run_id", "unknown")
        filepath = os.path.join(output_dir, f"evaluation_{pipeline_run_id}.json")
        os.makedirs(output_dir, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
        return filepath
    except Exception as e:
        print(f"Warning: Failed to save evaluation export: {e}")
        return ""

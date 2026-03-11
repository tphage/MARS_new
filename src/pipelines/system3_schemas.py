"""Pydantic schemas for System 3 (Manufacturability Assessment and Process Design)."""

from typing import List, Optional, Any, Dict, Literal
from pydantic import BaseModel, Field


def _get_schemas_config() -> dict:
    """Load schemas config from config.yaml, returning empty dict on failure."""
    try:
        from ..config import load_config
        return load_config().get("schemas", {})
    except (ImportError, FileNotFoundError, KeyError):
        return {}


# Constraint type and severity literals for BlockingConstraint
CONSTRAINT_TYPE_LITERAL = [
    "precursor_availability",
    "equipment",
    "temperature_pressure",
    "safety_regulatory",
    "ip_proprietary",
    "missing_critical_info",
    "scale_infeasible",
]
SEVERITY_LITERAL = ["hard", "soft"]


class System3Input(BaseModel):
    """Input contract for System 3 pipeline. All fields optional for validation flexibility."""

    system2_result: Optional[Dict[str, Any]] = None
    initial_query: Optional[str] = None
    material_X: Optional[str] = None
    application_Y: Optional[str] = None
    constraints_U: Optional[List[str]] = None
    properties_W: Optional[Dict[str, Any]] = None

    class Config:
        extra = "allow"


class ProcessStep(BaseModel):
    """High-level step in a lab-scale process recipe."""

    step_index: int = Field(..., description="1-based step order")
    description: str = Field(..., description="Short description of the step")
    conditions: Optional[str] = Field(None, description="Temperature, pressure, atmosphere, etc.")
    equipment_class: Optional[str] = Field(None, description="Class of equipment required")
    inputs: Optional[List[str]] = Field(None, description="Required inputs or precursors")

    class Config:
        extra = "allow"


class BlockingConstraint(BaseModel):
    """A single blocking constraint when status is blocked."""

    type: str = Field(
        ...,
        description="One of: precursor_availability, equipment, temperature_pressure, "
        "safety_regulatory, ip_proprietary, missing_critical_info, scale_infeasible",
    )
    severity: str = Field(..., description="hard or soft")
    description: str = Field(..., description="Human-readable description")
    suggested_mitigation: Optional[str] = Field(None, description="Optional mitigation")
    evidence_pointers: Optional[List[str]] = Field(None, description="IDs or refs to evidence chunks")

    class Config:
        extra = "allow"


class MaterialDecomposition(BaseModel):
    """Structured decomposition of a candidate into existing constituents."""
    is_composite: bool = Field(..., description="Whether the candidate is a composite or blend")
    constituents: List[str] = Field(default_factory=list, description="Constituent material names")
    composition_notes: str = Field("", description="Ratios/loadings or uncertainty notes")
    combination_modes: List[str] = Field(default_factory=list, description="Likely combination/processing modes")

    class Config:
        extra = "forbid"


class DecompositionQuery(BaseModel):
    """Structured retrieval query generated from decomposition."""
    query: str = Field(..., description="Retrieval query text")
    query_type: Literal["constituent", "combination"] = Field(..., description="Query category")
    constituent: str = Field("", description="Constituent name for constituent queries")
    is_combination_query: bool = Field(False, description="True when query targets combination process")

    class Config:
        extra = "forbid"


class System3OutputManufacturable(BaseModel):
    """Output when status is manufacturable."""

    status: Literal["manufacturable"] = "manufacturable"
    candidate: Optional[Dict[str, Any]] = Field(None, description="Candidate from System 2")
    process_recipe: List[ProcessStep] = Field(default_factory=list)
    evidence: Optional[List[Any]] = Field(None, description="Citations/IDs to retrieved chunks or nodes")

    class Config:
        extra = "allow"


class System3OutputBlocked(BaseModel):
    """Output when status is blocked."""

    status: Literal["blocked"] = "blocked"
    candidate: Optional[Dict[str, Any]] = Field(None, description="Candidate from System 2")
    blocking_constraints: List[BlockingConstraint] = Field(default_factory=list)
    feedback_to_system2: str = Field("", description="Concise constraint summary for System 2")

    class Config:
        extra = "allow"


def system3_output_to_dict(obj: BaseModel) -> dict:
    """Serialize System3Output (manufacturable or blocked) to dict, with legacy keys."""
    d = obj.model_dump()
    if isinstance(obj, System3OutputManufacturable):
        d["manufacturable"] = True
        d["info_text"] = _recipe_to_info_text(obj.process_recipe, obj.evidence)
        d["rejected_candidate"] = None
    elif isinstance(obj, System3OutputBlocked):
        d["manufacturable"] = False
        d["info_text"] = None
        # Extract material_name from candidate - raise KeyError if missing (no defaults)
        candidate_name = ""
        if obj.candidate:
            try:
                candidate_name = obj.candidate["material_name"]
            except KeyError:
                raise KeyError(
                    f"Missing required field 'material_name' in candidate dict. "
                    f"Available keys: {list(obj.candidate.keys())}"
                )
        d["rejected_candidate"] = {
            "candidate": candidate_name,
            "constraints": [c.description for c in obj.blocking_constraints],
            "reason": obj.feedback_to_system2,
        }
    return d


def _recipe_to_info_text(recipe: List[ProcessStep], evidence: Optional[List[Any]]) -> str:
    """Build a single info_text string from process recipe and evidence."""
    parts = []
    for step in recipe:
        parts.append(f"Step {step.step_index}: {step.description}")
        if step.conditions:
            parts.append(f"  Conditions: {step.conditions}")
        if step.equipment_class:
            parts.append(f"  Equipment: {step.equipment_class}")
        if step.inputs:
            parts.append(f"  Inputs: {', '.join(step.inputs)}")
    if evidence:
        cfg = _get_schemas_config()
        max_content = cfg.get("max_evidence_content_length", 200)
        max_items = cfg.get("max_evidence_items_before_truncation", 10)
        parts.append("\nEvidence:")
        for idx, ev in enumerate(evidence, 1):
            if isinstance(ev, dict):
                ev_id = ev.get("id", f"source_{idx}")
                ev_source = ev.get("source", ev.get("metadata", {}).get("source", "unknown"))
                ev_content = str(ev.get("content", ""))[:max_content]
                parts.append(f"  [{idx}] id={ev_id}, source={ev_source}: {ev_content}...")
            else:
                parts.append(f"  [{idx}] {str(ev)[:max_content]}")
            if idx >= max_items:
                parts.append(f"  ... and {len(evidence) - max_items} more evidence items")
                break
    return "\n".join(parts) if parts else ""

"""
Generate a structured PDF report from an evaluation_*.json file using LaTeX.

This mirrors the content and structure of `generate_evaluation_pdf.py`, but
instead of using ReportLab directly it:

- Renders the report as a LaTeX document
- Invokes `pdflatex` to compile it to a PDF

CLI usage:

    python -m src.tools.generate_evaluation_latex_pdf -i pipeline_logs/evaluation_XXXX.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from dataclasses import dataclass, field
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.evaluation_rubric import (
    load_evaluation_rubric,
    ordinal_scale_lines,
    subsystem_criterion_labels,
)


# ---------------------------
# Data model (mirrors generate_evaluation_pdf)
# ---------------------------


@dataclass
class RequiredMaterialProperties:
    properties: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)


@dataclass
class Candidate:
    candidate: str
    reasoning: str
    constraints_violated: List[str] = field(default_factory=list)
    source: str = ""


@dataclass
class FinalCandidate:
    material_name: str
    material_class: Optional[str] = None
    material_id: Optional[str] = None
    justification: str = ""
    properties: Optional[Dict[str, Any]] = None


@dataclass
class ManufacturingBlockingConstraint:
    type: str = ""
    severity: str = ""
    description: str = ""
    suggested_mitigation: str = ""
    evidence_pointers: List[str] = field(default_factory=list)


@dataclass
class ManufacturingProcess:
    status: str = "N/A"
    process_recipe: Optional[Any] = None
    blocking_constraints: List[ManufacturingBlockingConstraint] = field(
        default_factory=list
    )
    feedback_to_system2: Optional[str] = None


@dataclass
class EvaluationMetadata:
    pipeline_run_id: str = ""
    timestamp: str = ""
    final_outcome_status: str = ""
    total_iterations: Optional[int] = None
    total_rejected_candidates: Optional[int] = None


@dataclass
class EvaluationRun:
    sentence: str
    material_x: str
    application_y: str
    required_material_properties: RequiredMaterialProperties
    final_candidate: Optional[FinalCandidate]
    rejected_candidates: List[Candidate]
    manufacturing_process: ManufacturingProcess
    metadata: EvaluationMetadata


# ---------------------------
# JSON loading (copied from generate_evaluation_pdf to stay in sync)
# ---------------------------

def _latex_process_recipe_block(process_recipe: Any) -> str:
    if not process_recipe:
        return _latex_multiline(
            "No explicit process recipe was produced in this evaluation."
        ) + "\\\\\n"

    if isinstance(process_recipe, list):
        blocks = []
        for i, step in enumerate(process_recipe, start=1):
            if not isinstance(step, dict):
                blocks.append(
                    f"\\paragraph{{Step {i}}}\n"
                    + _latex_multiline(str(step))
                    + "\\\\\n"
                )
                continue

            step_idx = step.get("step_index", i)
            description = step.get("description", "")
            conditions = step.get("conditions", "")
            equipment = step.get("equipment_class", "")
            inputs = step.get("inputs", []) or []

            parts = [f"\\paragraph{{Step {step_idx}}}"]

            if description:
                parts.append(
                    f"\\textbf{{Description}}: {_latex_multiline(str(description))}\\\\"
                )
            if conditions:
                parts.append(
                    f"\\textbf{{Conditions}}: {_latex_multiline(str(conditions))}\\\\"
                )
            if equipment:
                parts.append(
                    f"\\textbf{{Equipment}}: {_latex_multiline(str(equipment))}\\\\"
                )
            if inputs:
                parts.append("\\textbf{Inputs}\\\\")
                parts.append(
                    "\\begin{itemize}\n"
                    + "\n".join(
                        f"  \\item {_latex_multiline(str(x))}" for x in inputs
                    )
                    + "\n\\end{itemize}\n"
                )

            blocks.append("\n".join(parts))

        return "\n\n".join(blocks) + "\n"

    return _latex_multiline(str(process_recipe)) + "\\\\\n"


def load_evaluation(path: Path) -> EvaluationRun:
    """
    Load an evaluation_*.json file into a structured EvaluationRun.

    The loader is intentionally tolerant of missing keys so that older
    or partial logs still produce a usable report.
    """
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    query = raw.get("query", {}) or {}
    sentence = query.get("sentence", "")
    material_x = query.get("material_X", "")
    application_y = query.get("application_Y", "")

    req_raw = raw.get("required_material_properties", {}) or {}
    req_props = req_raw.get("properties", []) or []
    req_constraints = req_raw.get("constraints", []) or []
    required_material_properties = RequiredMaterialProperties(
        properties=list(req_props),
        constraints=list(req_constraints),
    )

    cand_sel = raw.get("candidate_selection", {}) or {}
    final_raw = cand_sel.get("final_candidate") or None
    final_candidate: Optional[FinalCandidate]
    if final_raw:
        final_candidate = FinalCandidate(
            material_name=final_raw.get("material_name", "<unknown>"),
            material_class=final_raw.get("material_class"),
            material_id=final_raw.get("material_id"),
            justification=final_raw.get("justification", ""),
            properties=final_raw.get("properties"),
        )
    else:
        final_candidate = None

    rejected_raw = cand_sel.get("rejected_candidates", []) or []
    rejected_candidates: List[Candidate] = []
    for r in rejected_raw:
        rejected_candidates.append(
            Candidate(
                candidate=r.get("candidate", "<unknown>"),
                reasoning=(r.get("reasoning") or "").strip(),
                constraints_violated=list(r.get("constraints_violated", []) or []),
                source=(r.get("source") or "").strip(),
            )
        )

    mfg_raw = raw.get("manufacturing_process", {}) or {}
    mfg_status = mfg_raw.get("status", "N/A") or "N/A"
    mfg_recipe = mfg_raw.get("process_recipe")
    mfg_blocking_raw = mfg_raw.get("blocking_constraints", []) or []
    blocking_constraints: List[ManufacturingBlockingConstraint] = []
    for b in mfg_blocking_raw:
        blocking_constraints.append(
            ManufacturingBlockingConstraint(
                type=b.get("type", "") or "",
                severity=b.get("severity", "") or "",
                description=b.get("description", "") or "",
                suggested_mitigation=b.get("suggested_mitigation", "") or "",
                evidence_pointers=list(b.get("evidence_pointers", []) or []),
            )
        )
    mfg_feedback = mfg_raw.get("feedback_to_system2")
    manufacturing_process = ManufacturingProcess(
        status=mfg_status,
        process_recipe=mfg_recipe if mfg_recipe else None,
        blocking_constraints=blocking_constraints,
        feedback_to_system2=mfg_feedback,
    )

    meta_raw = raw.get("metadata", {}) or {}
    metadata = EvaluationMetadata(
        pipeline_run_id=meta_raw.get("pipeline_run_id", ""),
        timestamp=meta_raw.get("timestamp", ""),
        final_outcome_status=meta_raw.get("final_outcome_status", ""),
        total_iterations=meta_raw.get("total_iterations"),
        total_rejected_candidates=meta_raw.get("total_rejected_candidates"),
    )

    return EvaluationRun(
        sentence=sentence,
        material_x=material_x,
        application_y=application_y,
        required_material_properties=required_material_properties,
        final_candidate=final_candidate,
        rejected_candidates=rejected_candidates,
        manufacturing_process=manufacturing_process,
        metadata=metadata,
    )


# ---------------------------
# LaTeX helpers
# ---------------------------


def _latex_escape(text: str) -> str:
    """Escape LaTeX special characters in plain text."""
    if not text:
        return ""
    # Order matters: escape backslash first
    replacements = [
        ("\\", r"\\"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("#", r"\#"),
        ("$", r"\$"),
        ("%", r"\%"),
        ("&", r"\&"),
        ("_", r"\_"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ]
    out = text
    for src, dst in replacements:
        out = out.replace(src, dst)
    return out


def _latex_multiline(text: str) -> str:
    """Convert plain text with newlines into LaTeX-safe content."""
    if not text:
        return ""
    # Normalize unicode similar to the ReportLab version for consistency
    normalized = (
        text.replace("\u202f", " ")
        .replace("\u00a0", " ")
        .replace("\u2212", "-")
        .replace("\u2010", "-")
        .replace("\u2011", "-")
        .replace("\u2012", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u207b", "-")
        .replace("\u2070", "0")
        .replace("\u00b9", "1")
        .replace("\u00b2", "2")
        .replace("\u00b3", "3")
        .replace("\u2074", "4")
        .replace("\u2075", "5")
        .replace("\u2076", "6")
        .replace("\u2077", "7")
        .replace("\u2078", "8")
        .replace("\u2079", "9")
        # Replace a few common comparison symbols with ASCII equivalents
        .replace("\u2248", "approx.")
        .replace("\u2265", ">=")
    )
    lines = [_latex_escape(line) for line in normalized.splitlines()]
    return r" \\ ".join(line for line in lines if line.strip())


def _latex_inline_markdown(text: str) -> str:
    """
    Convert a small subset of markdown-like formatting to LaTeX-friendly markup.
    Supports **bold** only.
    """
    if not text:
        return ""
    # Reuse the unicode normalization from _latex_multiline
    normalized = (
        text.replace("\u202f", " ")
        .replace("\u00a0", " ")
        .replace("\u2212", "-")
        .replace("\u2010", "-")
        .replace("\u2011", "-")
        .replace("\u2012", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u207b", "-")
        .replace("\u2070", "0")
        .replace("\u00b9", "1")
        .replace("\u00b2", "2")
        .replace("\u00b3", "3")
        .replace("\u2074", "4")
        .replace("\u2075", "5")
        .replace("\u2076", "6")
        .replace("\u2077", "7")
        .replace("\u2078", "8")
        .replace("\u2079", "9")
        .replace("\u2248", "approx.")
        .replace("\u2265", ">=")
    )
    escaped = _latex_escape(normalized)
    # Convert **bold** to \textbf{...}
    return re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", escaped)


_BULLET_RE = re.compile(r"\s*(?:[•\-\*])\s+(.*)")
_NUMBERED_RE = re.compile(r"\s*\d+\.\s+(.*)")


def _latex_rich_text_block(text: str) -> str:
    """
    Render a markdown-like text block into LaTeX with support for:
    - Normal paragraphs
    - Bullet lists (•, -, *) rendered as an itemize block
    - Numbered lists (1., 2., ...) rendered as an enumerate block
    - **bold** via _latex_inline_markdown
    """
    if not text or not text.strip():
        return ""

    raw = text.replace("\r\n", "\n").strip()
    lines = raw.splitlines()

    blocks: List[str] = []
    para_lines: List[str] = []
    bullet_items: List[str] = []
    numbered_items: List[str] = []

    def flush_paragraph() -> None:
        nonlocal para_lines
        if not para_lines:
            return
        joined = " ".join(ln.strip() for ln in para_lines if ln.strip())
        if joined:
            blocks.append(_latex_inline_markdown(joined) + r"\\")
        para_lines = []

    def flush_bullets() -> None:
        nonlocal bullet_items
        if not bullet_items:
            return
        blocks.append(
            "\\begin{itemize}\n"
            + "\n".join(bullet_items)
            + "\n\\end{itemize}"
        )
        bullet_items = []

    def flush_numbered() -> None:
        nonlocal numbered_items
        if not numbered_items:
            return
        blocks.append(
            "\\begin{enumerate}\n"
            + "\n".join(numbered_items)
            + "\n\\end{enumerate}"
        )
        numbered_items = []

    for ln in lines:
        stripped = ln.strip()

        if not stripped:
            flush_paragraph()
            continue

        bm = _BULLET_RE.match(stripped)
        nm = _NUMBERED_RE.match(stripped)

        if bm:
            flush_paragraph()
            flush_numbered()
            bullet_items.append("  \\item " + _latex_inline_markdown(bm.group(1)))
        elif nm:
            flush_paragraph()
            flush_bullets()
            numbered_items.append("  \\item " + _latex_inline_markdown(nm.group(1)))
        else:
            if bullet_items:
                bullet_items[-1] += " " + _latex_inline_markdown(stripped)
            elif numbered_items:
                numbered_items[-1] += " " + _latex_inline_markdown(stripped)
            else:
                para_lines.append(ln)

    flush_paragraph()
    flush_bullets()
    flush_numbered()

    return "\n\n".join(blocks)


def _latex_json_block(data: Dict[str, Any]) -> str:
    """Render a JSON dict as a LaTeX verbatim-like block."""
    pretty = json.dumps(data, indent=2, ensure_ascii=False)
    escaped_lines = [_latex_escape(line) for line in pretty.splitlines()]
    body = "\n".join(escaped_lines)
    return "\\begin{verbatim}\n" + body + "\n\\end{verbatim}\n"


def _latex_itemize(items: List[str]) -> str:
    if not items:
        return ""
    body = "\n".join(f"  \\item {_latex_multiline(item)}" for item in items)
    return "\\begin{itemize}\n" + body + "\n\\end{itemize}\n"


def _scoring_scale_block_from_rubric(rubric: Dict[str, Any]) -> str:
    lines = [
        "Scoring scale (1--5), same as the LLM judge (config/evaluation\\_rubric.yaml):",
    ]
    lines.extend(ordinal_scale_lines(rubric))
    return _latex_itemize(lines)


def _subsystem_expert_evaluation_block(
    system_label: str, criteria: List[str], rubric: Dict[str, Any]
) -> str:
    """Per-criterion 1--5 scores; criterion list comes from evaluation_rubric.yaml."""
    criteria_block = _latex_itemize(criteria)
    scale_block = _scoring_scale_block_from_rubric(rubric)
    score_lines: List[str] = []
    for c in criteria:
        score_lines.append(
            "\\noindent\\textbf{"
            + _latex_escape(c)
            + "}\\\\[0.15cm]\n"
            "\\noindent\\textcolor{blue}{SCORE (1--5):} "
            "\\rule{2.2cm}{0.3mm}\\quad "
            "\\textcolor{blue}{Comment:} "
            "\\rule{10cm}{0.3mm}\\\\[0.55cm]\n"
        )
    scores_tex = "".join(score_lines)
    return (
        "\\vspace{0.6cm}\n"
        "\\noindent{\\Large\\bfseries\\textcolor{blue}{"
        + _latex_escape(system_label)
        + "}}\\\\[0.4cm]\n"
        "\\textit{Rate each criterion independently using the shared ordinal scale.}\\\\[0.35cm]\n"
        "Assess:\n\n"
        f"{criteria_block}\n"
        f"{scale_block}\n"
        "\\vspace{0.5cm}\n"
        f"{scores_tex}\n"
        "\\clearpage\n"
    )


# ---------------------------
# LaTeX document rendering
# ---------------------------


def render_latex(
    evaluation: EvaluationRun,
    label: Optional[str] = None,
    blind: bool = False,
    skip_rejected: bool = False,
    skip_hard_constraints: bool = False,
    rubric_path: Optional[Path] = None,
) -> str:
    """Render the full LaTeX document as a string."""
    rubric = load_evaluation_rubric(rubric_path)
    title_label = label or "PFAS Replacement Evaluation"

    meta_parts: List[str] = []
    if not blind:
        if evaluation.metadata.pipeline_run_id:
            meta_parts.append(f"Pipeline run ID: {evaluation.metadata.pipeline_run_id}")
        if evaluation.metadata.timestamp:
            meta_parts.append(f"Timestamp: {evaluation.metadata.timestamp}")
        if evaluation.metadata.final_outcome_status:
            meta_parts.append(f"Outcome: {evaluation.metadata.final_outcome_status}")
        if evaluation.metadata.total_iterations is not None:
            meta_parts.append(f"Iterations: {evaluation.metadata.total_iterations}")
        if evaluation.metadata.total_rejected_candidates is not None:
            meta_parts.append(
                f"Rejected candidates: {evaluation.metadata.total_rejected_candidates}"
            )

    meta_line = " | ".join(meta_parts)

    # System 1 content
    if evaluation.required_material_properties.properties:
        props_block = (
            "\\subsubsection*{Properties}\n"
            f"{_latex_itemize(evaluation.required_material_properties.properties)}\n"
        )
    else:
        props_block = (
            "\\subsubsection*{Properties}\n"
            f"{_latex_multiline('No structured required properties recorded.')}\\\\\n"
        )

    if skip_hard_constraints:
        constraints_block = ""
    else:
        if evaluation.required_material_properties.constraints:
            constraints_block = (
                "\\subsubsection*{Hard constraints}\n"
                f"{_latex_itemize(evaluation.required_material_properties.constraints)}\n"
            )
        else:
            constraints_block = (
                "\\subsubsection*{Hard constraints}\n"
                f"{_latex_multiline('No explicit hard constraints recorded.')}\\\\\n"
            )

    # System 2 - final candidate
    if evaluation.final_candidate:
        fc = evaluation.final_candidate
        fc_lines = [
            f"\\textbf{{Name}}: {_latex_multiline(fc.material_name)}\\\\",
        ]
        if fc.material_id:
            fc_lines.append(
                f"\\textbf{{ID}}: {_latex_multiline(fc.material_id)}\\\\"
            )
        fc_block = "\n".join(fc_lines) + "\n"

        if fc.properties:
            fc_block += "\\medskip\n\\textbf{Structured properties}\\\\\n"
            fc_block += _latex_json_block(fc.properties)

        if fc.justification:
            fc_block += "\\medskip\n\\textbf{Justification}\\\\\n"
            fc_block += _latex_rich_text_block(fc.justification) + "\n"
    else:
        fc_block = _latex_multiline(
            "No final candidate present in this evaluation log."
        ) + "\\\\\n"

    # System 2 - rejected candidates
    if evaluation.rejected_candidates:
        rc_blocks: List[str] = []
        for idx, cand in enumerate(evaluation.rejected_candidates, start=1):
            parts: List[str] = [
                f"\\paragraph{{Candidate {idx}}}",
                f"\\textbf{{Name}}: {_latex_multiline(cand.candidate)}\\\\",
                f"\\textbf{{Source}}: {_latex_multiline(cand.source or '-') }\\\\",
            ]
            if cand.reasoning:
                parts.append(
                    f"\\textbf{{Reasoning}}: {_latex_multiline(cand.reasoning)}\\\\"
                )
            if cand.constraints_violated:
                items = [
                    f"Constraint violated: {c}" for c in cand.constraints_violated
                ]
                parts.append(_latex_itemize(items))
            else:
                parts.append(
                    _latex_multiline(
                        "No explicit violated constraints recorded."
                    )
                    + "\\\\"
                )
            rc_blocks.append("\n".join(parts))
        rejected_block = "\n\n".join(rc_blocks)
    else:
        rejected_block = (
            _latex_multiline(
                "No rejected candidates recorded in this evaluation."
            )
            + "\\\\\n"
        )

    # System 3 - manufacturing
    mfg_lines = [
        f"\\textbf{{Status}}: {_latex_multiline(evaluation.manufacturing_process.status)}\\\\"
    ]
    mfg_block = "\n".join(mfg_lines) + "\n"

    if evaluation.manufacturing_process.process_recipe:
        mfg_block += "\\medskip\n\\textbf{Process recipe (summary)}\\\\\n"
        mfg_block += _latex_process_recipe_block(
            evaluation.manufacturing_process.process_recipe
        )
    else:
        mfg_block += (
            _latex_multiline(
                "No explicit process recipe was produced in this evaluation."
            )
            + "\\\\\n"
        )

    if evaluation.manufacturing_process.blocking_constraints:
        bc_blocks: List[str] = []
        for idx, bc in enumerate(
            evaluation.manufacturing_process.blocking_constraints, start=1
        ):
            parts: List[str] = [
                f"\\paragraph{{Blocking constraint {idx}}}",
                f"\\textbf{{Type}}: {_latex_multiline(bc.type or '-')}\\\\",
                f"\\textbf{{Severity}}: {_latex_multiline(bc.severity or '-')}\\\\",
                f"\\textbf{{Description}}: {_latex_multiline(bc.description or '-')}\\\\",
                f"\\textbf{{Suggested mitigation}}: {_latex_multiline(bc.suggested_mitigation or '-')}\\\\",
            ]
            if bc.evidence_pointers:
                items = [f"Evidence: {p}" for p in bc.evidence_pointers]
                parts.append(_latex_itemize(items))
            bc_blocks.append("\n".join(parts))
        mfg_block += "\n\n" + "\n\n".join(bc_blocks)
    else:
        mfg_block += (
            _latex_multiline(
                "No blocking manufacturing constraints recorded."
            )
            + "\\\\\n"
        )

    if evaluation.manufacturing_process.feedback_to_system2:
        mfg_block += (
            "\\medskip\n\\textbf{Feedback to System 2}\\\\\n"
            + _latex_multiline(
                evaluation.manufacturing_process.feedback_to_system2
            )
            + "\\\\\n"
        )

    # Assemble full document
    meta_tex = _latex_multiline(meta_line) if meta_line else ""
    if skip_rejected:
        query_block = (
            f"{_latex_multiline(evaluation.sentence or '-')}\\\\"
        )
    else:
        query_block = "\n".join(
            [
                f"\\textbf{{Sentence}}: {_latex_multiline(evaluation.sentence or '-')}\\\\",
                f"\\textbf{{Material X}}: {_latex_multiline(evaluation.material_x or '-')}\\\\",
                f"\\textbf{{Application Y}}: {_latex_multiline(evaluation.application_y or '-')}\\\\",
            ]
        )

    if skip_rejected:
        rejected_section = ""
    else:
        rejected_section = (
            "\\subsection*{Rejected candidates}\n" + rejected_block
        )

    s1_criteria = subsystem_criterion_labels(rubric, 1)
    s2_criteria = subsystem_criterion_labels(rubric, 2)
    s3_criteria = subsystem_criterion_labels(rubric, 3)

    system1_eval_block = _subsystem_expert_evaluation_block(
        "System 1 -- Expert evaluation section", s1_criteria, rubric
    )
    system2_eval_block = _subsystem_expert_evaluation_block(
        "System 2 -- Expert evaluation section", s2_criteria, rubric
    )
    system3_eval_block = _subsystem_expert_evaluation_block(
        "System 3 -- Expert evaluation section", s3_criteria, rubric
    )

    # Basic article class; user can post-process as needed
    document = f"""\\documentclass[11pt]{{article}}
\\usepackage[margin=1in]{{geometry}}
\\usepackage[T1]{{fontenc}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{lmodern}}
\\usepackage{{hyperref}}
\\usepackage{{xcolor}}

\\begin{{document}}

\\begin{{center}}
    \\LARGE {_latex_escape(title_label)}\\\\[0.5cm]
    {meta_tex}
\\end{{center}}

\\section*{{Query}}
{query_block}

\\begin{{center}}
\\Huge\\bfseries System 1\\\\[0.15cm]
\\Large\\bfseries Required Material Properties
\\end{{center}}
\\vspace{{0.3cm}}
\\section*{{System 1 -- Results}}
{props_block}
{constraints_block}

{system1_eval_block}

\\begin{{center}}
\\Huge\\bfseries System 2\\\\[0.15cm]
\\Large\\bfseries Candidate Selection
\\end{{center}}
\\vspace{{0.3cm}}
\\section*{{System 2 -- Results}}

\\subsection*{{Final candidate}}
{fc_block}

{rejected_section}

{system2_eval_block}

\\begin{{center}}
\\Huge\\bfseries System 3\\\\[0.15cm]
\\Large\\bfseries Manufacturing Process
\\end{{center}}
\\vspace{{0.3cm}}
\\section*{{System 3 -- Results}}
{mfg_block}

{system3_eval_block}

\\end{{document}}
"""
    return document


def render_pdf_with_latex(
    evaluation: EvaluationRun,
    output_path: Path,
    label: Optional[str] = None,
    blind: bool = False,
    skip_rejected: bool = False,
    skip_hard_constraints: bool = False,
    rubric_path: Optional[Path] = None,
) -> None:
    """
    Render a PDF report at output_path by generating LaTeX and running pdflatex.

    Requires `pdflatex` to be available on the system PATH.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    latex_source = render_latex(
        evaluation,
        label=label,
        blind=blind,
        skip_rejected=skip_rejected,
        skip_hard_constraints=skip_hard_constraints,
        rubric_path=rubric_path,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        tex_path = tmpdir_path / "evaluation.tex"
        with tex_path.open("w", encoding="utf-8") as f:
            f.write(latex_source)

        cmd = ["pdflatex", "-interaction=nonstopmode", tex_path.name]
        try:
            # We don't use check=True here because LaTeX may return a non-zero
            # exit code even when a PDF was produced (e.g. for minor warnings).
            result = subprocess.run(
                cmd,
                cwd=tmpdir_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "pdflatex not found on PATH. Please install a LaTeX distribution "
                "such as TeX Live or MacTeX, and ensure `pdflatex` is available."
            ) from exc

        pdf_tmp = tmpdir_path / "evaluation.pdf"
        if not pdf_tmp.is_file():
            # Surface LaTeX output to help debugging when no PDF is present
            log = result.stdout.decode("utf-8", errors="ignore")
            raise RuntimeError(
                "pdflatex did not produce evaluation.pdf. "
                "LaTeX output follows:\\n" + log
            )

        output_path.write_bytes(pdf_tmp.read_bytes())


# ---------------------------
# Public API / CLI
# ---------------------------


def generate_evaluation_pdf_latex(
    input_path: str | Path,
    output_path: Optional[str | Path] = None,
    label: Optional[str] = None,
    blind: bool = False,
    skip_rejected: bool = False,
    rubric_path: Optional[str | Path] = None,
) -> Path:
    """
    High-level helper to generate a LaTeX-based PDF from one evaluation JSON.
    """
    input_path = Path(input_path)
    evaluation = load_evaluation(input_path)

    if output_path is None:
        base_dir = input_path.parent
        reports_dir = base_dir / "reports"
        run_id = evaluation.metadata.pipeline_run_id or input_path.stem.replace(
            "evaluation_", ""
        )
        output_path = reports_dir / f"evaluation_{run_id}_latex.pdf"
    else:
        output_path = Path(output_path)

    rp = Path(rubric_path) if rubric_path else None
    render_pdf_with_latex(
        evaluation,
        output_path,
        label=label,
        blind=blind,
        skip_rejected=skip_rejected,
        rubric_path=rp,
    )
    return output_path


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a LaTeX-based PDF report from an evaluation_*.json file.",
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to evaluation_*.json",
    )
    parser.add_argument(
        "-o",
        "--output",
        help=(
            "Optional path to output PDF "
            "(default: pipeline_logs/reports/evaluation_<id>_latex.pdf)"
        ),
    )
    parser.add_argument(
        "--label",
        help="Optional blind-evaluation label to show in the PDF header (e.g. 'Response A').",
    )
    parser.add_argument(
        "--blind",
        action="store_true",
        help="Hide pipeline IDs and timestamps inside the PDF (for blind expert evaluation).",
    )
    parser.add_argument(
        "--rubric",
        type=Path,
        default=None,
        help="Path to evaluation_rubric.yaml (default: config/evaluation_rubric.yaml).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    input_path = Path(args.input)
    if not input_path.is_file():
        raise SystemExit(f"Input JSON not found: {input_path}")

    output_path = generate_evaluation_pdf_latex(
        input_path=input_path,
        output_path=args.output,
        label=args.label,
        blind=bool(args.blind),
        rubric_path=args.rubric,
    )
    print(f"Wrote LaTeX evaluation PDF to: {output_path}")


if __name__ == "__main__":
    main()


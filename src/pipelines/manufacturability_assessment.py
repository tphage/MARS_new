"""Manufacturing Assessment Pipeline (System 3)

Assesses whether a candidate material proposed by System 2 can be manufactured at lab scale.
If manufacturable: returns process recipe, evidence.
If blocked: returns blocking constraints, feedback for System 2, and optionally updates tracker.
"""

import json
import logging
import hashlib
from collections import defaultdict
from typing import Any, Dict, List, Optional

from ..agents.tracker import RejectedCandidateTracker
from ..config import load_config
from .system3_schemas import (
    System3Input,
    System3OutputManufacturable,
    System3OutputBlocked,
    ProcessStep,
    BlockingConstraint,
    MaterialDecomposition,
    DecompositionQuery,
    system3_output_to_dict,
)

logger = logging.getLogger(__name__)


def run_manufacturability_assessment_pipeline(
    system2_result: Dict[str, Any],
    initial_query: Optional[str] = None,
    material_X: Optional[str] = None,
    application_Y: Optional[str] = None,
    constraints_U: Optional[List[str]] = None,
    tracker: Optional[RejectedCandidateTracker] = None,
    process_analyst=None,
    manager=None,
    properties_W: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    temperature: Optional[float] = None,
    chat_logger: Optional[Any] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Manufacturing Assessment: evaluates lab-scale manufacturability of the System 2 candidate.

    If manufacturable: returns status=manufacturable, process_recipe, evidence.
    If blocked: returns status=blocked, blocking_constraints, feedback_to_system2; optionally updates tracker.

    This function follows a fail-loud philosophy: it raises exceptions instead of failing quietly
    when critical inputs are missing or operations fail unexpectedly.

    Args:
        system2_result: Full output from run_material_discovery_pipeline (required)
            Must have success=True and contain a valid candidate with material_name
        initial_query: User's original query/sentence (optional)
        material_X: Material to replace (from user) (optional)
        application_Y: Target application (from user) (optional)
        constraints_U: User constraints (optional)
        tracker: Optional RejectedCandidateTracker for closed-loop (add rejections when blocked)
        process_analyst: MultiAnalyst instance for process retrieval (required)
        manager: ResearchManager instance for feasibility and recipe synthesis (required)
        properties_W: Requirements W from System 1 (optional)
        config: Config dict for max_process_families, n_results_per_source (optional)
        temperature: LLM temperature (default 0)
        chat_logger: Optional ChatLogger for tracking LLM calls (same naming as S1/S2: system3_chat_log_manufacturability_assessment_{run_id}.json)
        **kwargs: Additional arguments passed to manager methods

    Returns:
        Dict with status, manufacturable, candidate, process_recipe or blocking_constraints,
        feedback_to_system2, and legacy keys (info_text, rejected_candidate).

    Raises:
        ValueError: If system2_result is invalid, candidate is missing/invalid, candidate material_name
            is missing/empty/"Unknown", process_analyst is None, or manager is None
        RuntimeError: If query generation fails or returns empty list, or if LLM operations fail
        ValueError: If JSON parsing fails in any LLM operation
    """
    # Load config
    yaml_config = load_config()
    pipeline_config = yaml_config.get("pipelines", {}).get("manufacturability_assessment", {})
    
    # Use config default if not provided
    if temperature is None:
        temperature = pipeline_config.get("temperature", 0)
    
    logger.info("System 3: Manufacturability assessment starting.")

    # Strict input validation - raise ValueError for missing critical inputs
    if system2_result is None:
        raise ValueError("system2_result is required but was None")
    
    if not isinstance(system2_result, dict):
        raise ValueError(f"system2_result must be a dict, got {type(system2_result)}")
    
    if not system2_result.get("success"):
        raise ValueError(f"system2_result must have success=True, got success={system2_result.get('success')}")
    
    candidate = system2_result.get("candidate")
    if candidate is None:
        raise ValueError("system2_result must contain a 'candidate' field")
    
    if not isinstance(candidate, dict):
        raise ValueError(f"candidate must be a dict, got {type(candidate)}")
    
    material_name = candidate.get("material_name")
    if not material_name or material_name.strip() == "" or material_name.strip().lower() == "unknown":
        raise ValueError(
            f"candidate must have a valid 'material_name' field (non-empty and not 'Unknown'), "
            f"got: {repr(material_name)}"
        )
    
    if process_analyst is None:
        raise ValueError("process_analyst is required but was None")
    
    if manager is None:
        raise ValueError("manager is required but was None")
    
    # Extract optional fields (these can be None/empty, but we validate they exist)
    material_class = candidate.get("material_class", "")
    justification = candidate.get("justification", "")
    application_Y = application_Y or ""
    constraints_U = constraints_U or []
    properties_W = properties_W or {}

    cfg = config or {}
    runtime_pipeline_cfg = cfg.get("pipelines", {}).get("manufacturability_assessment", {})
    pipeline_cfg = {**pipeline_config, **runtime_pipeline_cfg}
    max_process_families = pipeline_cfg.get("max_process_families", 20)

    # Verbose pipeline header
    print("=" * 70)
    print("Manufacturability Assessment Pipeline: Starting Process")
    print("=" * 70)
    print("\nPipeline Configuration:")
    print(f"  [OK] Candidate material: {material_name}")
    if material_class:
        print(f"  [OK] Material class: {material_class}")
    if application_Y:
        print(f"  [OK] Application: {application_Y}")
    print(f"  [OK] Constraints: {len(constraints_U)} constraint(s)")
    if constraints_U:
        for constraint in constraints_U:
            print(f"      - {constraint}")
    print(f"  [OK] Max process families: {max_process_families}")
    print(f"  [OK] Temperature: {temperature}")
    if properties_W and properties_W.get("required"):
        print(f"  [OK] Required properties: {len(properties_W.get('required', []))} property(ies)")
    print(f"  [OK] Tracker: {'Available' if tracker else 'Not provided'}")

    # Process retrieval: subqueries from LLM, then MultiAnalyst
    print("\n" + "-" * 70)
    print("Step 1: Process Retrieval")
    print("-" * 70)
    print("→ Extracting constituent materials from candidate...")
    decomposition_raw = manager.extract_material_constituents_for_manufacturing(
        candidate_Z=candidate,
        application_Y=application_Y,
        constraints_U=constraints_U,
        temperature=temperature,
        **kwargs
    )
    decomposition = MaterialDecomposition(**decomposition_raw)
    decomposition_data = decomposition.model_dump()
    print(f"✓ Constituents identified: {len(decomposition.constituents)}")
    for idx, constituent in enumerate(decomposition.constituents, 1):
        print(f"  {idx}. {constituent}")
    if decomposition.is_composite:
        print("  → Composite candidate detected; combination-process retrieval enabled")

    print("→ Generating decomposition-aware retrieval queries...")
    query_plan_raw = manager.generate_decomposition_process_queries(
        decomposition=decomposition_data,
        candidate_Z=candidate,
        application_Y=application_Y,
        temperature=temperature,
        **kwargs
    )
    query_plan = [DecompositionQuery(**q).model_dump() for q in query_plan_raw]
    if not query_plan:
        raise RuntimeError("generate_decomposition_process_queries returned an empty query plan")

    constituent_query_count = sum(1 for q in query_plan if q["query_type"] == "constituent")
    combination_query_count = sum(1 for q in query_plan if q["query_type"] == "combination")
    print(f"✓ Generated {len(query_plan)} retrieval queries "
          f"({constituent_query_count} constituent, {combination_query_count} combination)")
    for i, qspec in enumerate(query_plan, 1):
        query_preview = qspec["query"][:80] + "..." if len(qspec["query"]) > 80 else qspec["query"]
        q_label = f"{qspec['query_type']}:{qspec.get('constituent', '')}" if qspec["query_type"] == "constituent" else "combination"
        print(f"  {i}. [{q_label}] {query_preview}")

    print(f"\n→ Retrieving evidence from RAG sources for {len(query_plan)} queries...")
    all_rag_results = []
    constituent_with_evidence = set()
    combination_queries_with_evidence = 0
    for i, qspec in enumerate(query_plan, 1):
        q = qspec["query"]
        print(f"  [{i}/{len(query_plan)}] Processing query: \"{q[:70]}{'...' if len(q) > 70 else ''}\"")
        ret = process_analyst.analyze_question(q)
        rag_results = ret.get("rag_results", [])
        if qspec["query_type"] == "constituent" and rag_results:
            constituent_with_evidence.add(qspec.get("constituent", ""))
        if qspec["query_type"] == "combination" and rag_results:
            combination_queries_with_evidence += 1
        for r in rag_results:
            item = dict(r)
            item["retrieval_query"] = q
            item["query_type"] = qspec["query_type"]
            item["constituent"] = qspec.get("constituent", "")
            item["is_combination_query"] = qspec.get("is_combination_query", False)
            all_rag_results.append(item)
        print(f"      ✓ Retrieved {len(rag_results)} documents from RAG")

    # Deduplicate with source-aware stable keys; then source-balance before final cap.
    print(f"\n→ Deduplicating results...")
    print(f"  → Total documents before deduplication: {len(all_rag_results)}")
    sorted_results = sorted(
        all_rag_results,
        key=lambda x: (
            x.get("distance", float("inf")),
            str(x.get("source", "")),
            str(x.get("id", "")),
        ),
    )
    seen = set()
    unique_results = []
    for r in sorted_results:
        source_norm = str(r.get("source") or "unknown").strip().lower()
        id_norm = str(r.get("id") or "").strip().lower()
        content_norm = " ".join(str(r.get("content", "")).lower().split())[:500]
        content_hash = hashlib.sha1(content_norm.encode("utf-8")).hexdigest()
        key = (source_norm, id_norm, content_hash)
        if key in seen:
            continue
        seen.add(key)
        unique_results.append(r)

    by_source = defaultdict(list)
    for item in unique_results:
        by_source[str(item.get("source") or "unknown")].append(item)

    source_count = len(by_source) if by_source else 1
    per_source_cap = max(1, max_process_families // source_count)
    source_balanced = []
    selected_keys = set()
    for source_name in sorted(by_source.keys()):
        for item in by_source[source_name][:per_source_cap]:
            source_norm = str(item.get("source") or "unknown").strip().lower()
            id_norm = str(item.get("id") or "").strip().lower()
            content_norm = " ".join(str(item.get("content", "")).lower().split())[:500]
            item_key = (source_norm, id_norm, hashlib.sha1(content_norm.encode("utf-8")).hexdigest())
            if item_key in selected_keys:
                continue
            selected_keys.add(item_key)
            source_balanced.append(item)
            if len(source_balanced) >= max_process_families:
                break
        if len(source_balanced) >= max_process_families:
            break

    if len(source_balanced) < max_process_families:
        for item in unique_results:
            source_norm = str(item.get("source") or "unknown").strip().lower()
            id_norm = str(item.get("id") or "").strip().lower()
            content_norm = " ".join(str(item.get("content", "")).lower().split())[:500]
            item_key = (source_norm, id_norm, hashlib.sha1(content_norm.encode("utf-8")).hexdigest())
            if item_key in selected_keys:
                continue
            selected_keys.add(item_key)
            source_balanced.append(item)
            if len(source_balanced) >= max_process_families:
                break

    retrieved_rag_results = source_balanced
    constituent_with_evidence_lower = {c.lower() for c in constituent_with_evidence if c}
    constituents_without_evidence = [
        c for c in decomposition.constituents if c.lower() not in constituent_with_evidence_lower
    ]
    combination_queries_total = sum(1 for q in query_plan if q["query_type"] == "combination")
    evidence_coverage = {
        "constituents_total": len(decomposition.constituents),
        "constituents_with_evidence": sorted(list(constituent_with_evidence)),
        "constituents_without_evidence": constituents_without_evidence,
        "combination_queries_total": combination_queries_total,
        "combination_queries_with_evidence": combination_queries_with_evidence,
        "combination_queries_without_evidence": max(0, combination_queries_total - combination_queries_with_evidence),
        "query_plan_total": len(query_plan),
        "query_plan_constituent": constituent_query_count,
        "query_plan_combination": combination_query_count,
    }
    print(f"  → Unique documents after deduplication: {len(retrieved_rag_results)}")
    if len(all_rag_results) > len(retrieved_rag_results):
        print(f"  → Removed {len(all_rag_results) - len(retrieved_rag_results)} duplicate(s)")
    if len(retrieved_rag_results) >= max_process_families:
        print(f"  → Capped at max_process_families={max_process_families}")
    print(f"✓ Collected {len(retrieved_rag_results)} evidence items")
    logger.info("System 3: Process retrieval collected %d evidence items.", len(retrieved_rag_results))

    if not retrieved_rag_results:
        print("\n⚠ No evidence retrieved; returning blocked (missing critical info).")
        logger.warning("System 3: No evidence retrieved; returning blocked (missing critical info).")
        out = System3OutputBlocked(
            candidate=candidate,
            blocking_constraints=[
                BlockingConstraint(
                    type="missing_critical_info",
                    severity="hard",
                    description="No manufacturing evidence found in textbooks, patents, or spec sheets.",
                    suggested_mitigation="Check ChromaDB corpora or broaden the candidate/material class.",
                    evidence_pointers=None,
                )
            ],
            feedback_to_system2="No manufacturing evidence found for this candidate.",
        )
        result = system3_output_to_dict(out)
        result["initial_query"] = initial_query
        result["system2_result"] = system2_result
        result["decomposition"] = decomposition_data
        result["retrieval_query_plan"] = query_plan
        result["evidence_coverage"] = evidence_coverage
        if tracker:
            tracker.add_rejected(
                candidate=material_name,
                constraints=[out.feedback_to_system2],
                reason="RAG returned no results.",
                source="manufacturability",
            )
            print(f"✓ Added rejection to tracker (source=manufacturability)")
        if chat_logger is not None:
            try:
                chat_log_path = chat_logger.save()
                if chat_log_path:
                    result["chat_log_path"] = chat_log_path
                    print(f"\n✓ Saved chat log to {chat_log_path}")
            except Exception as e:
                logger.warning("Failed to save chat log: %s", e)
        print("\n" + "=" * 70)
        print("Manufacturability Assessment: Process Complete")
        print("=" * 70)
        print("Summary:")
        print("  - Status: BLOCKED")
        print("  - Evidence items retrieved: 0")
        print("  - Blocking constraints: 1")
        print("=" * 70)
        return result

    # ----------------------------------------------------------------
    # Step 2: Generate Feasibility Questions
    # ----------------------------------------------------------------
    num_feasibility_questions = pipeline_cfg.get("num_feasibility_questions", pipeline_config.get("num_feasibility_questions", 4))
    print("\n" + "-" * 70)
    print("Step 2: Generate Feasibility Questions")
    print("-" * 70)
    print(f"→ Generating {num_feasibility_questions} feasibility assessment questions...")
    print(f"  → Using {len(retrieved_rag_results)} evidence items from Step 1 as context")

    # No try-except: let exceptions propagate - failures should crash the pipeline
    feasibility_questions = manager.generate_feasibility_questions(
        candidate_Z=candidate,
        properties_W=properties_W,
        application_Y=application_Y,
        constraints_U=constraints_U,
        retrieved_rag_results=retrieved_rag_results,
        evidence_coverage=evidence_coverage,
        num_questions=num_feasibility_questions,
        temperature=temperature,
        **kwargs
    )

    print(f"✓ Generated {len(feasibility_questions)} feasibility questions:")
    for i, question in enumerate(feasibility_questions, 1):
        q_preview = question[:80] + "..." if len(question) > 80 else question
        print(f"  {i}. {q_preview}")

    # ----------------------------------------------------------------
    # Step 3: Answer Each Question (1 LLM call per question + per-question RAG)
    # ----------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Step 3: Answer Feasibility Questions")
    print("-" * 70)
    print(f"→ Answering {len(feasibility_questions)} questions (1 LLM call per question with per-question RAG retrieval)...")

    question_answers = []
    for i, question in enumerate(feasibility_questions, 1):
        print(f"\n  [{i}/{len(feasibility_questions)}] Question: \"{question[:80]}{'...' if len(question) > 80 else ''}\"")

        # Per-question RAG retrieval
        print(f"      → Retrieving evidence for this question...")
        ret = process_analyst.analyze_question(question)
        q_rag_results = ret.get("rag_results", [])
        print(f"      ✓ Retrieved {len(q_rag_results)} documents from RAG")

        # Answer the question using 1 LLM call
        print(f"      → Generating answer using LLM...")
        answer_result = manager.answer_feasibility_question(
            question=question,
            rag_results=q_rag_results,
            candidate_Z=candidate,
            temperature=temperature,
            **kwargs
        )

        qa_pair = {
            "question": question,
            "answer": answer_result["answer"],
            "confidence": answer_result["confidence"],
            "evidence_used": answer_result["evidence_used"],
            "num_rag_documents": len(q_rag_results),
        }
        question_answers.append(qa_pair)

        # Print summary
        answer_preview = answer_result["answer"][:150] + "..." if len(answer_result["answer"]) > 150 else answer_result["answer"]
        print(f"      ✓ Answer (confidence={answer_result['confidence']}): {answer_preview}")

    print(f"\n✓ All {len(question_answers)} questions answered")
    confidence_counts = {}
    for qa in question_answers:
        c = qa["confidence"]
        confidence_counts[c] = confidence_counts.get(c, 0) + 1
    print(f"  Confidence breakdown: {confidence_counts}")

    # ----------------------------------------------------------------
    # Step 4: Evaluate Feasibility (1 LLM call aggregating all Q&A pairs)
    # ----------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Step 4: Feasibility Evaluation")
    print("-" * 70)
    print("→ Evaluating overall manufacturability feasibility from Q&A pairs...")
    print(f"  → Candidate: {material_name}")
    if material_class:
        print(f"  → Material class: {material_class}")
    if application_Y:
        print(f"  → Application: {application_Y}")
    print(f"  → Q&A pairs: {len(question_answers)}")
    if properties_W and properties_W.get("required"):
        print(f"  → Required properties: {len(properties_W.get('required', []))} property(ies)")
    if constraints_U:
        print(f"  → Constraints: {len(constraints_U)} constraint(s)")

    # No try-except: let exceptions propagate - failures should crash the pipeline
    feasibility = manager.assess_manufacturability_feasibility(
        question_answers=question_answers,
        candidate_Z=candidate,
        properties_W=properties_W,
        application_Y=application_Y,
        constraints_U=constraints_U,
        evidence_coverage=evidence_coverage,
        temperature=temperature,
        **kwargs
    )

    # Extract required fields from feasibility result - raise KeyError if missing (no defaults)
    try:
        feasible = feasibility["feasible"]
        blocking_constraints_raw = feasibility["blocking_constraints"]
        feedback_to_system2 = feasibility["feedback_to_system2"]
    except KeyError as e:
        raise KeyError(
            f"Missing required field '{e.args[0]}' in feasibility assessment result. "
            f"Available keys: {list(feasibility.keys())}"
        ) from e

    # Normalize blocking constraints for downstream logic
    normalized_constraints: List[Dict[str, Any]] = []
    for c in blocking_constraints_raw or []:
        if isinstance(c, dict):
            c_type = c.get("type", "missing_critical_info")
            c_severity = c.get("severity", "hard")
            c_desc = c.get("description", "")
            normalized_constraints.append(
                {
                    "type": c_type,
                    "severity": c_severity,
                    "description": c_desc,
                    "suggested_mitigation": c.get("suggested_mitigation"),
                    "evidence_pointers": c.get("evidence_pointers"),
                }
            )
        else:
            normalized_constraints.append(
                {
                    "type": "missing_critical_info",
                    "severity": "hard",
                    "description": str(c),
                    "suggested_mitigation": None,
                    "evidence_pointers": None,
                }
            )

    # Composite-specific guardrail: if combination retrieval got no evidence, bias to missing_critical_info,
    # but only hard-block when Q&A pairs are also dominated by insufficient evidence on process aspects.
    if decomposition.is_composite and evidence_coverage.get("combination_queries_total", 0) > 0:
        if evidence_coverage.get("combination_queries_with_evidence", 0) == 0:
            insufficient_count = sum(
                1 for qa in question_answers if qa.get("confidence") == "insufficient_evidence"
            )
            total_q = len(question_answers) or 1
            insufficient_majority = insufficient_count * 2 >= total_q

            # Ensure we have a composite-related constraint in the list
            composite_constraint = {
                "type": "missing_critical_info",
                "severity": "soft",
                "description": (
                    "No evidence was retrieved for constituent-combination process queries. "
                    "Composite manufacturability should be treated as conditionally established and "
                    "validated via targeted mixing/compounding experiments."
                ),
                "suggested_mitigation": (
                    "Prioritize candidates with documented mixing/compounding routes and interfacial "
                    "compatibility evidence for the selected constituent materials, or plan lab-scale "
                    "compounding trials to validate the process window."
                ),
                "evidence_pointers": None,
            }
            normalized_constraints.append(composite_constraint)

            if insufficient_majority:
                # In this stricter case, escalate to a hard block on missing critical combination info.
                feasible = False
                composite_constraint["severity"] = "hard"
                if not feedback_to_system2:
                    feedback_to_system2 = (
                        "Missing composite combination-process evidence; require documented constituent "
                        "compatibility and compounding route before acceptance."
                    )

    # Decide how hard vs soft constraints affect overall feasibility.
    hard_non_missing = [
        c for c in normalized_constraints
        if c.get("severity") == "hard" and c.get("type") != "missing_critical_info"
    ]

    # If the LLM judged the candidate feasible but only soft or missing_critical_info constraints are present,
    # treat it as manufacturable and surface constraints as advisory in the final result.
    advisory_constraints: List[Dict[str, Any]] = []
    if feasible:
        if hard_non_missing:
            # There is at least one hard, non-missing_critical_info constraint: override to blocked.
            feasible = False
        else:
            advisory_constraints = normalized_constraints


    print(f"\n  Feasibility Evaluation:")
    print(f"     Feasible: {'[YES]' if feasible else '[NO]'}")

    if feasible:
        # ----------------------------------------------------------------
        # Step 5: Recipe Synthesis
        # ----------------------------------------------------------------
        print("\n" + "-" * 70)
        print("Step 5: Recipe Synthesis")
        print("-" * 70)
        print("→ Synthesizing process recipe...")
        logger.info("System 3: Candidate judged manufacturable; synthesizing recipe.")
        # No try-except: let exceptions propagate - failures should crash the pipeline
        recipe_result = manager.synthesize_process_recipe(
            retrieved_rag_results=retrieved_rag_results,
            candidate_Z=candidate,
            application_Y=application_Y,
            temperature=temperature,
            **kwargs
        )

        # Extract required fields from recipe_result - raise KeyError if missing (no defaults)
        try:
            raw_steps = recipe_result["process_recipe"]
            evidence = recipe_result["evidence"]
        except KeyError as e:
            raise KeyError(
                f"Missing required field '{e.args[0]}' in recipe synthesis result. "
                f"Available keys: {list(recipe_result.keys())}"
            ) from e
        
        steps = []
        for s in raw_steps:
            if isinstance(s, dict):
                # Extract required step fields - raise KeyError if missing
                try:
                    step_index = s["step_index"]
                    description = s["description"]
                except KeyError as e:
                    raise KeyError(
                        f"Missing required field '{e.args[0]}' in process_recipe step. "
                        f"Available keys: {list(s.keys())}"
                    ) from e
                
                # Convert conditions dict/list to a human-readable string
                conditions = s.get("conditions")  # Optional field
                if isinstance(conditions, dict):
                    # Format as "key: value" pairs for readability
                    conditions = "; ".join(
                        f"{k}: {v}" for k, v in conditions.items() if v is not None
                    )
                elif isinstance(conditions, list):
                    conditions = "; ".join(str(c) for c in conditions)
                elif conditions is not None and not isinstance(conditions, str):
                    conditions = str(conditions)
                
                steps.append(ProcessStep(
                    step_index=step_index,
                    description=description,
                    conditions=conditions,
                    equipment_class=s.get("equipment_class"),  # Optional field
                    inputs=s.get("inputs"),  # Optional field
                ))
            else:
                raise ValueError(
                    f"Expected process_recipe step to be a dict, got {type(s)}: {s}"
                )

        print(f"✓ Generated {len(steps)} process steps")
        if evidence:
            print(f"✓ Evidence items: {len(evidence)}")
        if steps:
            print(f"\n  Process Recipe Preview:")
            for step in steps[:3]:  # Show first 3 steps
                step_desc = step.description[:70] + "..." if len(step.description) > 70 else step.description
                print(f"    Step {step.step_index}: {step_desc}")
            if len(steps) > 3:
                print(f"    ... and {len(steps) - 3} more step(s)")

        out = System3OutputManufacturable(
            candidate=candidate,
            process_recipe=steps,
            evidence=evidence,
        )
        logger.info("System 3: Returning manufacturable with %d recipe steps.", len(steps))
        result = system3_output_to_dict(out)
        if advisory_constraints:
            # Surface any soft or missing_critical_info constraints as advisory notes rather than blockers.
            result["advisory_constraints"] = advisory_constraints
        result["initial_query"] = initial_query
        result["system2_result"] = system2_result
        result["question_answers"] = question_answers
        result["decomposition"] = decomposition_data
        result["retrieval_query_plan"] = query_plan
        result["evidence_coverage"] = evidence_coverage
        if chat_logger is not None:
            try:
                chat_log_path = chat_logger.save()
                if chat_log_path:
                    result["chat_log_path"] = chat_log_path
                    print(f"\n✓ Saved chat log to {chat_log_path}")
            except Exception as e:
                logger.warning("Failed to save chat log: %s", e)
        
        print("\n" + "=" * 70)
        print("Manufacturability Assessment: Process Complete")
        print("=" * 70)
        print("Summary:")
        print("  - Status: MANUFACTURABLE")
        print(f"  - Evidence items retrieved (Step 1): {len(retrieved_rag_results)}")
        print(f"  - Feasibility questions asked: {len(question_answers)}")
        print(f"  - Process recipe steps: {len(steps)}")
        print(f"  - Evidence items in recipe: {len(evidence)}")
        print("=" * 70)
        return result

    # ----------------------------------------------------------------
    # Step 5 (blocked path): Blocking Constraints
    # ----------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Step 5: Blocking Constraints")
    print("-" * 70)
    print("→ Processing blocking constraints...")
    constraints = []
    for c in normalized_constraints:
        if isinstance(c, dict):
            # Extract required constraint fields - raise KeyError if missing
            try:
                constraints.append(BlockingConstraint(
                    type=c["type"],
                    severity=c["severity"],
                    description=c["description"],
                    suggested_mitigation=c.get("suggested_mitigation"),  # Optional field
                    evidence_pointers=c.get("evidence_pointers"),  # Optional field
                ))
            except KeyError as e:
                raise KeyError(
                    f"Missing required field '{e.args[0]}' in blocking_constraint dict. "
                    f"Available keys: {list(c.keys())}"
                ) from e
        else:
            constraints.append(BlockingConstraint(
                type="missing_critical_info",
                severity="hard",
                description=str(c),
            ))
    if not constraints:
        constraints.append(BlockingConstraint(
            type="missing_critical_info",
            severity="hard",
            description=feedback_to_system2 or "Manufacturability assessment returned not feasible.",
        ))

    print(f"✓ Identified {len(constraints)} blocking constraint(s):")
    for i, constraint in enumerate(constraints, 1):
        constraint_type = constraint.type if hasattr(constraint, 'type') else 'unknown'
        constraint_desc = constraint.description if hasattr(constraint, 'description') else str(constraint)
        constraint_desc_preview = constraint_desc[:70] + "..." if len(constraint_desc) > 70 else constraint_desc
        print(f"  {i}. [{constraint_type}] {constraint_desc_preview}")
    
    if feedback_to_system2:
        feedback_preview = feedback_to_system2[:100] + "..." if len(feedback_to_system2) > 100 else feedback_to_system2
        print(f"\n  Feedback to System 2: {feedback_preview}")

    out = System3OutputBlocked(
        candidate=candidate,
        blocking_constraints=constraints,
        feedback_to_system2=feedback_to_system2 or "; ".join(c.description for c in constraints[:3]),
    )
    if tracker:
        constraint_list = [feedback_to_system2] if feedback_to_system2 else [c.description for c in constraints]
        tracker.add_rejected(
            candidate=material_name,
            constraints=constraint_list,
            reason=feedback_to_system2[:500] if feedback_to_system2 else "",
            source="manufacturability",
        )
        print(f"\n✓ Added rejection to tracker (source=manufacturability)")
        logger.info("System 3: Added rejection to tracker (source=manufacturability).")
    logger.info("System 3: Returning blocked with %d constraints.", len(constraints))
    result = system3_output_to_dict(out)
    result["initial_query"] = initial_query
    result["system2_result"] = system2_result
    result["question_answers"] = question_answers
    result["decomposition"] = decomposition_data
    result["retrieval_query_plan"] = query_plan
    result["evidence_coverage"] = evidence_coverage
    if chat_logger is not None:
        try:
            chat_log_path = chat_logger.save()
            if chat_log_path:
                result["chat_log_path"] = chat_log_path
                print(f"\n✓ Saved chat log to {chat_log_path}")
        except Exception as e:
            logger.warning("Failed to save chat log: %s", e)
    
    print("\n" + "=" * 70)
    print("Manufacturability Assessment: Process Complete")
    print("=" * 70)
    print("Summary:")
    print("  - Status: BLOCKED")
    print(f"  - Evidence items retrieved (Step 1): {len(retrieved_rag_results)}")
    print(f"  - Feasibility questions asked: {len(question_answers)}")
    print(f"  - Blocking constraints: {len(constraints)}")
    print("=" * 70)
    return result

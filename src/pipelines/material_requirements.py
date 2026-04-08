"""Material Requirements Analysis Pipeline (Substitute Material Requirements Analysis)"""

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
from ..agents import ResearchAnalyst, ResearchManager, ResearchAssistant, ResearchScientist
from ..config import load_config

logger = logging.getLogger(__name__)


def _clean_extracted_keywords(keywords: List[str]) -> List[str]:
    """
    Post-process LLM-extracted keywords to remove formatting artifacts.

    Removes:
    - Markdown bold/italic wrappers (** / *)
    - Lines that are headers or meta-commentary (e.g. "Key material attributes…")
    - Trailing summary sentences (e.g. "These keywords capture…")
    - Leading/trailing whitespace

    Args:
        keywords: Raw keyword list from LLM extraction.

    Returns:
        Cleaned keyword list with artifacts stripped.
    """
    if not keywords:
        return keywords

    # Patterns that indicate meta-commentary rather than actual properties
    _META_PATTERNS = [
        re.compile(r"^(these|the above|in summary|overall|note:)", re.IGNORECASE),
        re.compile(r"capture the essential|summarize|listed above|key material attributes and constraints", re.IGNORECASE),
    ]

    min_len = load_config().get("pipelines", {}).get("material_requirements", {}).get("min_keyword_length", 3)
    cleaned = []
    for kw in keywords:
        if not isinstance(kw, str):
            continue
        # Strip markdown bold / italic wrappers
        kw = re.sub(r"\*{1,2}(.+?)\*{1,2}", r"\1", kw).strip()
        # Skip empty after stripping
        if not kw:
            continue
        # Skip meta-commentary lines
        if any(pat.search(kw) for pat in _META_PATTERNS):
            continue
        # Skip very short non-informative tokens
        if len(kw) < min_len:
            continue
        cleaned.append(kw)

    return cleaned


def _extract_keywords_from_question(question: str, max_keywords: Optional[int] = None) -> List[str]:
    """
    Extract keywords from a question text for knowledge graph querying.
    
    Uses simple NLP: removes stop words, extracts noun phrases and important terms.
    
    Args:
        question: Question text to extract keywords from
        max_keywords: Maximum number of keywords to return (default: None, uses config value)
        
    Returns:
        List of keyword strings suitable for KG querying
    """
    # Load config if max_keywords not provided
    if max_keywords is None:
        config = load_config()
        pipeline_config = config.get("pipelines", {}).get("material_requirements", {})
        max_keywords = pipeline_config.get("max_keywords", 5)
    
    if not question or not isinstance(question, str):
        return []
    
    # Common stop words to remove
    stop_words = {
        'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
        'could', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those',
        'what', 'which', 'who', 'whom', 'where', 'when', 'why', 'how',
        'for', 'to', 'of', 'in', 'on', 'at', 'by', 'with', 'from', 'as',
        'and', 'or', 'but', 'if', 'then', 'than', 'so', 'because', 'since',
        'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below'
    }
    
    # Remove punctuation and split into words
    text = re.sub(r'[^\w\s]', ' ', question.lower())
    words = text.split()
    
    # Filter out stop words and short words (< 3 chars), keep meaningful terms
    keywords = []
    for word in words:
        word = word.strip()
        if len(word) >= 3 and word not in stop_words:
            keywords.append(word)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_keywords = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique_keywords.append(kw)
    
    # Return top keywords, prioritizing longer/more specific terms
    # Sort by length (descending) to prioritize more specific terms
    unique_keywords.sort(key=lambda x: len(x), reverse=True)
    
    return unique_keywords[:max_keywords]


def run_fixed_pipeline(
    sentence: str,
    analyst: ResearchAnalyst,
    manager: ResearchManager,
    research_assistant: ResearchAssistant,
    scientist: ResearchScientist,
    pfas_scientist: ResearchScientist,
    keywords: List[str] = None,
    include_rag_context: bool = None,
    max_items: int = None,
    temperature: float = None,
    n_results: int = None,
    chat_logger=None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run the material requirements analysis pipeline (Substitute Material Requirements Analysis).
    
    This pipeline extracts material properties W required for a substitute material.
    
    Workflow:
    1. RAG retrieval (sentence + keywords) → ResearchAnalyst
    2. Generate research questions → ResearchManager
    3. Process each question via RAG → ResearchAnalyst
    4. Query PFAS KG for each question → ResearchScientist
    5. Answer each question → ResearchManager (with RAG + KG context)
    6. Extract keywords → ResearchAssistant
    
    Args:
        sentence: Input sentence/question to process
        analyst: ResearchAnalyst instance (required)
        manager: ResearchManager instance (required)
        research_assistant: ResearchAssistant instance (required)
        scientist: ResearchScientist instance (unused; retained for interface consistency)
        pfas_scientist: ResearchScientist instance (required, for Step 4 - PFAS KG querying during question answering)
        keywords: List of keywords (optional)
        include_rag_context: If True, include RAG results in manager's prompt (default: None, uses config value)
        max_items: Maximum number of items for manager to return (default: None, uses config value)
        temperature: Temperature for LLM generation (default: None, uses config value)
        n_results: Number of RAG results to retrieve (default: None, uses config value)
        **kwargs: Additional arguments passed to manager methods
        
    Returns:
        Dict containing:
            - "sentence": Original input sentence
            - "keywords": Provided keywords
            - "analyst_result": Full result from ResearchAnalyst
            - "manager_result": List output from ResearchManager (limited to max_items)
            - "question_results": List of RAG results for each question (with KG results if available)
            - "question_answers": List of dicts with 'question', 'answer', 'num_rag_documents', 'is_answered'
            - "extracted_keywords": List of keywords/key phrases extracted from answered questions
            - "num_rag_results": Number of RAG documents retrieved
            - "num_manager_items": Number of items generated by manager
            - "num_question_results": Number of questions processed
            - "num_question_answers": Number of questions answered
            - "num_extracted_keywords": Number of keywords extracted
            - "connections_found": Always False (System 1 does not generate graphs)
    """
    print("=" * 70)
    print("Material Requirements Analysis: Starting Process")
    print("=" * 70)
    
    # Load config
    config = load_config()
    pipeline_config = config.get("pipelines", {}).get("material_requirements", {})
    
    # Use config defaults if not provided
    if include_rag_context is None:
        include_rag_context = pipeline_config.get("include_rag_context", True)
    if max_items is None:
        max_items = pipeline_config.get("max_items", 6)
    if temperature is None:
        temperature = pipeline_config.get("temperature", 0)
    if n_results is None:
        n_results = pipeline_config.get("n_results", 5)
    
    if keywords is None:
        keywords = []
    
    # Step 1: Use ResearchAnalyst to perform RAG
    print("\n" + "-" * 70)
    print("Step 1: ResearchAnalyst - Performing RAG Retrieval")
    print("-" * 70)
    print(f"→ Querying ChromaDB with sentence and keywords...")
    analyst_result = analyst.analyze(sentence, keywords)
    num_rag_results = len(analyst_result.get('rag_results', []))
    print(f"✓ RAG retrieval complete: {num_rag_results} documents retrieved")

    if num_rag_results == 0:
        print("✗ No RAG results found, using analyze_question instead (i.e., no keywords used in retrieval)")
        analyst_result = analyst.analyze_question(sentence)
        num_rag_results = len(analyst_result.get('rag_results', []))
        print(f"✓ RAG retrieval complete: {num_rag_results} documents retrieved")
    
    if num_rag_results > 0:
        print(f"  Document previews:")
        for i, result in enumerate(analyst_result.get('rag_results', [])[:3], 1):
            content_preview = result.get('content', '')[:150]
            print(f"    {i}. {content_preview}{'...' if len(result.get('content', '')) > 150 else ''}")
    
    # Step 2: Prepare analysis_result dictionary for ResearchManager
    print("\n" + "-" * 70)
    print("Step 2: Preparing Analysis Result for ResearchManager")
    print("-" * 70)
    manager_analysis_result = {
        "sentence": sentence,
        "keywords": keywords,
        "rag_results": analyst_result.get('rag_results', []) if include_rag_context else [],
        "num_results": len(analyst_result.get('rag_results', []) if include_rag_context else [])
    }
    print(f"✓ Prepared analysis result with {manager_analysis_result['num_results']} RAG documents")
    
    # Step 3: Use ResearchManager to generate questions
    print("\n" + "-" * 70)
    print("Step 3: ResearchManager - Generating Research Questions")
    print("-" * 70)
    print(f"→ Calling ResearchManager.process() with max_items={max_items}, temperature={temperature}...")
    manager_result = manager.process(
        analysis_result=manager_analysis_result,
        temperature=temperature,
        max_items=max_items,
        **kwargs
    )
    num_questions = len(manager_result) if manager_result else 0
    print(f"✓ ResearchManager complete: {num_questions} questions generated")
    if manager_result:
        print(f"  Generated questions:")
        for i, question in enumerate(manager_result, 1):
            print(f"    {i}. {question[:100]}{'...' if len(question) > 100 else ''}")
    
    # Step 4: Process questions automatically
    print("\n" + "-" * 70)
    print("Step 4: Processing Questions - Collecting Information via RAG and PFAS KG")
    print("-" * 70)
    question_results = []

    # Helper to process a single question (RAG + KG)
    def _process_single_question(question: str) -> Dict[str, Any]:
        """Retrieve RAG docs and query PFAS KG for one question."""
        question_result = analyst.analyze_question(question=question)
        kg_results = None
        question_keywords = _extract_keywords_from_question(question)
        min_question_keywords = config.get("pipelines", {}).get("material_requirements", {}).get("min_question_keywords", 2)
        if question_keywords and len(question_keywords) >= min_question_keywords:
            kg_results = pfas_scientist.find_connections(
                keywords=question_keywords,
                use_best_match_only=True,
            )
        question_result["kg_results"] = kg_results
        question_result["question_keywords"] = question_keywords
        return question_result

    # Load parallelization config
    pipeline_config = config.get("pipelines", {}).get("material_requirements", {})
    parallel_workers = pipeline_config.get("parallel_rag_workers", 1)

    if manager_result and len(manager_result) > 0:
        valid_questions = [(i, q) for i, q in enumerate(manager_result, 1) if q and q.strip()]
        print(f"→ Processing {len(valid_questions)} questions (workers={parallel_workers})...")

        if parallel_workers > 1 and len(valid_questions) > 1:
            # Parallel execution of RAG + KG queries
            ordered_results = [None] * len(valid_questions)
            with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                future_to_idx = {}
                for idx, (i, question) in enumerate(valid_questions):
                    future = executor.submit(_process_single_question, question)
                    future_to_idx[future] = (idx, i, question)

                for future in as_completed(future_to_idx):
                    idx, i, question = future_to_idx[future]
                    try:
                        qr = future.result()
                        ordered_results[idx] = qr
                        num_docs = len(qr.get("rag_results", []))
                        kg_res = qr.get("kg_results")
                        kg_found = kg_res and kg_res.get("summary", {}).get("connections_found", False)
                        print(f"  [{i}/{len(manager_result)}] {question[:60]}... → {num_docs} docs, KG={'yes' if kg_found else 'no'}")
                    except Exception as e:
                        raise RuntimeError(f"Question processing failed for Q{i}: {e}") from e

            question_results = [r for r in ordered_results if r is not None]
        else:
            # Sequential execution (original behaviour, with improved logging)
            for i, question in valid_questions:
                print(f"  [{i}/{len(manager_result)}] Processing: {question[:80]}{'...' if len(question) > 80 else ''}")
                question_result = _process_single_question(question)
                num_docs = len(question_result.get("rag_results", []))
                print(f"      ✓ Retrieved {num_docs} documents from RAG")
                kg_res = question_result.get("kg_results")
                if kg_res:
                    kg_summary = kg_res.get("summary", {})
                    if kg_summary.get("connections_found"):
                        print(f"      ✓ PFAS KG: Found {kg_summary.get('num_paths_found', 0)} paths, {kg_summary.get('subgraph_nodes', 0)} nodes")
                    else:
                        print(f"      → PFAS KG: No connections found")
                question_results.append(question_result)
    else:
        print("  No questions to process")
    
    print(f"✓ Question processing complete: {len(question_results)} questions processed")
    
    # Step 5: Answer each question using ResearchManager
    print("\n" + "-" * 70)
    print("Step 5: ResearchManager - Answering Questions")
    print("-" * 70)
    question_answers = []
    if manager_result and len(manager_result) > 0 and question_results:
        print(f"→ Answering {len(question_results)} questions based on retrieved information...")
        for i, (question, q_result) in enumerate(zip(manager_result, question_results), 1):
            if question and question.strip():
                print(f"  [{i}/{len(question_results)}] Answering: {question[:80]}{'...' if len(question) > 80 else ''}")
                rag_results = q_result.get('rag_results', [])
                num_docs = len(rag_results)
                
                # Extract KG context from question result
                kg_context = q_result.get('kg_results')
                has_kg_context = kg_context is not None and kg_context.get('summary', {}).get('connections_found', False)
                if has_kg_context:
                    print(f"      → Including PFAS KG context in answer")
                
                # Use ResearchManager to answer the question
                answer = manager.answer_question(
                    question=question,
                    rag_results=rag_results,
                    kg_context=kg_context,
                    temperature=temperature,
                    **kwargs
                )
                
                # Determine if the question was answered using LLM evaluation
                print(f"      → Evaluating if question was answered using LLM...")
                is_answered = manager._is_question_answered(question, answer, num_docs, temperature)
                
                # Store question-answer pair
                qa_pair = {
                    "question": question,
                    "answer": answer,
                    "num_rag_documents": num_docs,
                    "is_answered": is_answered,
                    "has_kg_context": has_kg_context
                }
                question_answers.append(qa_pair)
                
                answer_preview = answer[:150] if answer else "No answer generated"
                status = "✓ Answered" if is_answered else "✗ Not answered"
                kg_indicator = " (with KG)" if has_kg_context else ""
                print(f"      {status}{kg_indicator} - Generated answer ({len(answer)} chars): {answer_preview}{'...' if len(answer) > 150 else ''}")
    else:
        print("  No questions to answer")
    
    print(f"✓ Question answering complete: {len(question_answers)} questions answered")
    
    # Step 6: Extract keywords from answered questions
    extracted_keywords = []
    print("\n" + "-" * 70)
    print("Step 6: ResearchAssistant - Extracting Material Property Keywords")
    print("-" * 70)
    
    # Filter to only answered questions
    answered_qa_pairs = [qa for qa in question_answers if qa.get('is_answered', False)]
    num_answered = len(answered_qa_pairs)
    
    if num_answered > 0:
        print(f"→ Extracting keywords from original question and {num_answered} answered questions...")
        extracted_keywords = research_assistant.extract_keywords(
            original_question=sentence,
            question_answers=question_answers,
            temperature=temperature,
            **kwargs
        )
        # Clean formatting artifacts (markdown bold, summary sentences, etc.)
        raw_count = len(extracted_keywords)
        extracted_keywords = _clean_extracted_keywords(extracted_keywords)
        if raw_count != len(extracted_keywords):
            print(f"  → Cleaned {raw_count - len(extracted_keywords)} formatting artifacts from keywords")
        print(f"✓ Extracted {len(extracted_keywords)} keywords/key phrases")
        if extracted_keywords:
            print(f"  Sample keywords: {', '.join(extracted_keywords[:5])}{'...' if len(extracted_keywords) > 5 else ''}")
    else:
        print("  No answered questions available for keyword extraction")
    
    # Step 6b: Extract hard constraints from answered questions
    extracted_constraints = []
    print("\n" + "-" * 70)
    print("Step 6b: ResearchAssistant - Extracting Hard Constraints")
    print("-" * 70)
    if num_answered > 0:
        print(f"→ Extracting constraints from original question and {num_answered} answered questions...")
        try:
            extracted_constraints = research_assistant.extract_constraints(
                original_question=sentence,
                question_answers=question_answers,
                temperature=temperature,
                **kwargs
            )
            # Clean formatting artifacts
            raw_count = len(extracted_constraints)
            extracted_constraints = _clean_extracted_keywords(extracted_constraints)
            if raw_count != len(extracted_constraints):
                print(f"  → Cleaned {raw_count - len(extracted_constraints)} formatting artifacts from constraints")
            print(f"✓ Extracted {len(extracted_constraints)} hard constraints")
            if extracted_constraints:
                for i, c in enumerate(extracted_constraints[:5], 1):
                    print(f"  {i}. {c}")
                if len(extracted_constraints) > 5:
                    print(f"  ... and {len(extracted_constraints) - 5} more")
        except Exception as e:
            print(f"  Warning: Constraint extraction failed: {e}")
            extracted_constraints = []
    else:
        print("  No answered questions available for constraint extraction")

    # Step 7 intentionally removed: System 1 does not generate graphs.
    connections_found = False
    
    print("\n" + "=" * 70)
    print("Material Requirements Analysis: Process Complete")
    print("=" * 70)
    print(f"Summary:")
    print(f"  - RAG documents retrieved: {num_rag_results}")
    print(f"  - Questions generated: {num_questions}")
    print(f"  - Questions processed: {len(question_results)}")
    print(f"  - Questions answered: {len(question_answers)}")
    if extracted_keywords:
        print(f"  - Keywords extracted: {len(extracted_keywords)}")
    print("=" * 70 + "\n")
    
    result = {
        "sentence": sentence,
        "keywords": keywords,
        "analyst_result": analyst_result,
        "manager_result": manager_result,
        "question_results": question_results,
        "question_answers": question_answers,
        "extracted_keywords": extracted_keywords,
        "extracted_constraints": extracted_constraints,
        "num_rag_results": num_rag_results,
        "num_manager_items": num_questions,
        "num_question_results": len(question_results),
        "num_question_answers": len(question_answers),
        "num_extracted_keywords": len(extracted_keywords),
        "num_extracted_constraints": len(extracted_constraints),
        "connections_found": connections_found
    }
    
    # Save chat log if chat_logger is provided
    if chat_logger is not None:
        try:
            chat_log_path = chat_logger.save()
            if chat_log_path:
                result["chat_log_path"] = chat_log_path
        except Exception as e:
            logger.warning("Failed to save chat log: %s", e)
    
    return result


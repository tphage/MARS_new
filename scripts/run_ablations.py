#!/usr/bin/env python3
"""Run ablation conditions for one or more benchmark queries.

Three conditions are supported:
  - 3agent         : 3 sequential LLM calls (no RAG / no KG)
  - 1agent_rag     : 1 LLM call with pre-retrieved RAG + KG context
  - 1agent_no_rag  : 1 LLM call, purely parametric

Usage:
    python scripts/run_ablations.py                                 # all conditions, all queries
    python scripts/run_ablations.py --queries Query1,Query2         # subset of queries
    python scripts/run_ablations.py --condition 3agent              # single condition
    python scripts/run_ablations.py --condition 1agent_rag --queries Query1
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, load_prompts
from src.utils import (
    llm,
    TransformerEmbeddingFunction,
    MaterialDatabase,
    PropertyMapper,
    load_ablation_queries,
    extract_json_from_response,
    build_ablation_evaluation,
    save_ablation_result,
    format_rag_results_for_prompt,
    format_kg_results_for_prompt,
)
from src.agents import ResearchAnalyst, ResearchScientist

ALL_CONDITIONS = ["3agent", "1agent_rag", "1agent_no_rag"]


# ---------------------------------------------------------------------------
# Condition 1: 3-agent sequential (no RAG/KG)
# ---------------------------------------------------------------------------
def run_3agent_ablation(query, generate_fn, ablation_prompts, temperature=0):
    """Three separate LLM calls (property extraction -> candidate -> mfg)."""
    raw_responses = {}
    start_time = time.time()

    # Agent 1 — property extraction
    print("  Agent 1: Property Extraction…")
    a1_sys = ablation_prompts["agent1_properties"]
    a1_usr = ablation_prompts["agent1_properties_user_prompt"].format(
        sentence=query["sentence"],
        material_X=query["material_X"],
        application_Y=query["application_Y"],
    )
    a1_raw = generate_fn(system_prompt=a1_sys, prompt=a1_usr, temperature=temperature)
    raw_responses["agent1_properties"] = a1_raw
    a1_parsed = extract_json_from_response(a1_raw)
    if a1_parsed is None:
        a1_parsed = {"properties": [], "constraints": []}
    properties = a1_parsed.get("properties", [])
    constraints = a1_parsed.get("constraints", [])
    print(f"    {len(properties)} properties, {len(constraints)} constraints")

    # Agent 2 — material discovery
    print("  Agent 2: Material Discovery…")
    a2_sys = ablation_prompts["agent2_candidate"]
    a2_usr = ablation_prompts["agent2_candidate_user_prompt"].format(
        material_X=query["material_X"],
        application_Y=query["application_Y"],
        properties_json=json.dumps(properties, indent=2),
        constraints_json=json.dumps(constraints, indent=2),
    )
    a2_raw = generate_fn(system_prompt=a2_sys, prompt=a2_usr, temperature=temperature)
    raw_responses["agent2_candidate"] = a2_raw
    a2_parsed = extract_json_from_response(a2_raw)
    if a2_parsed is None:
        a2_parsed = {"material_name": "UNKNOWN", "material_class": "unknown", "justification": a2_raw}
    candidate = a2_parsed
    print(f"    Candidate: {candidate.get('material_name', 'UNKNOWN')}")

    # Agent 3 — manufacturability
    print("  Agent 3: Manufacturability…")
    a3_sys = ablation_prompts["agent3_manufacturing"]
    a3_usr = ablation_prompts["agent3_manufacturing_user_prompt"].format(
        material_name=candidate.get("material_name", "UNKNOWN"),
        material_class=candidate.get("material_class", "unknown"),
        application_Y=query["application_Y"],
        justification=candidate.get("justification", ""),
        properties_json=json.dumps(properties, indent=2),
        constraints_json=json.dumps(constraints, indent=2),
    )
    a3_raw = generate_fn(system_prompt=a3_sys, prompt=a3_usr, temperature=temperature)
    raw_responses["agent3_manufacturing"] = a3_raw
    a3_parsed = extract_json_from_response(a3_raw)
    if a3_parsed is None:
        a3_parsed = {
            "status": "unknown", "process_recipe": None,
            "blocking_constraints": [], "feedback_to_system2": a3_raw,
        }
    manufacturing = a3_parsed
    print(f"    Status: {manufacturing.get('status', 'unknown')}")

    return build_ablation_evaluation(
        query=query, properties=properties, constraints=constraints,
        candidate=candidate, manufacturing=manufacturing,
        condition_name="3agent",
        run_id=datetime.now().strftime("%Y%m%d%H"),
        duration_seconds=time.time() - start_time,
        raw_responses=raw_responses,
    )


# ---------------------------------------------------------------------------
# Condition 2: 1-agent + RAG/KG
# ---------------------------------------------------------------------------
def _pre_retrieve_context(query, rag_analysts, scientists, material_db):
    """Retrieve RAG + KG context using the raw query."""
    context_parts = []
    sentence = query["sentence"]
    keywords = [query["material_X"], query["application_Y"]]

    for source_name, analyst in rag_analysts.items():
        try:
            result = analyst.analyze_question(sentence)
            rag_results = result.get("rag_results", [])
            context_parts.append(format_rag_results_for_prompt(rag_results, source_name))
            print(f"    RAG [{source_name}]: {len(rag_results)} documents")
        except Exception as e:
            print(f"    RAG [{source_name}]: Error — {e}")
            context_parts.append(f"[{source_name}]: Retrieval failed.\n")

    for kg_name, scientist in scientists.items():
        try:
            kg_result = scientist.find_connections(keywords)
            context_parts.append(format_kg_results_for_prompt(kg_result, kg_name))
            print(f"    KG [{kg_name}]: {len(kg_result.get('matched_node_ids', []))} nodes, "
                  f"{len(kg_result.get('found_paths', []))} paths")
        except Exception as e:
            print(f"    KG [{kg_name}]: Error — {e}")
            context_parts.append(f"[{kg_name} KG]: Connection search failed.\n")

    mat_lines = ["--- Available Materials Database ---"]
    for mat in material_db.materials:
        mat_name = mat.get("material_name", mat.get("material_id", "unknown"))
        mat_class = mat.get("material_class", "")
        mat_lines.append(f"- {mat_name} ({mat_class})")
    context_parts.append("\n".join(mat_lines))
    return "\n\n".join(context_parts)


def run_1agent_rag_ablation(query, generate_fn, ablation_prompts, rag_analysts,
                             scientists, material_db, temperature=0):
    """Single LLM call with pre-retrieved RAG + KG context."""
    raw_responses = {}
    start_time = time.time()

    print("  Pre-retrieving context…")
    context = _pre_retrieve_context(query, rag_analysts, scientists, material_db)
    print(f"  Context retrieved ({len(context)} chars, {time.time() - start_time:.1f}s)")

    print("  Running single-agent LLM call…")
    sys_prompt = ablation_prompts["single_agent_with_context"]
    usr_prompt = ablation_prompts["single_agent_with_context_user_prompt"].format(
        sentence=query["sentence"],
        material_X=query["material_X"],
        application_Y=query["application_Y"],
        retrieved_context=context,
    )
    raw = generate_fn(system_prompt=sys_prompt, prompt=usr_prompt, temperature=temperature)
    raw_responses["single_agent"] = raw
    raw_responses["retrieved_context_chars"] = len(context)

    parsed = extract_json_from_response(raw) or {}
    rmp = parsed.get("required_material_properties", {})
    properties = rmp.get("properties", [])
    constraints = rmp.get("constraints", [])
    cs = parsed.get("candidate_selection", {})
    candidate = cs.get("final_candidate", cs) if cs else None
    manufacturing = parsed.get("manufacturing_process", {
        "status": "unknown", "process_recipe": None,
        "blocking_constraints": [], "feedback_to_system2": "",
    })

    print(f"    Properties: {len(properties)}, Constraints: {len(constraints)}")
    if candidate:
        print(f"    Candidate: {candidate.get('material_name', 'N/A')}")
    print(f"    Status: {manufacturing.get('status', 'unknown')}")

    return build_ablation_evaluation(
        query=query, properties=properties, constraints=constraints,
        candidate=candidate, manufacturing=manufacturing,
        condition_name="1agent_rag",
        run_id=datetime.now().strftime("%Y%m%d%H"),
        duration_seconds=time.time() - start_time,
        raw_responses=raw_responses,
    )


# ---------------------------------------------------------------------------
# Condition 3: 1-agent, no RAG/KG
# ---------------------------------------------------------------------------
def run_1agent_no_rag_ablation(query, generate_fn, ablation_prompts, temperature=0):
    """Single LLM call, purely parametric knowledge."""
    raw_responses = {}
    start_time = time.time()

    print("  Running single-agent LLM call (no context)…")
    sys_prompt = ablation_prompts["single_agent_no_context"]
    usr_prompt = ablation_prompts["single_agent_no_context_user_prompt"].format(
        sentence=query["sentence"],
        material_X=query["material_X"],
        application_Y=query["application_Y"],
    )
    raw = generate_fn(system_prompt=sys_prompt, prompt=usr_prompt, temperature=temperature)
    raw_responses["single_agent"] = raw

    parsed = extract_json_from_response(raw) or {}
    rmp = parsed.get("required_material_properties", {})
    properties = rmp.get("properties", [])
    constraints = rmp.get("constraints", [])
    cs = parsed.get("candidate_selection", {})
    candidate = cs.get("final_candidate", cs) if cs else None
    manufacturing = parsed.get("manufacturing_process", {
        "status": "unknown", "process_recipe": None,
        "blocking_constraints": [], "feedback_to_system2": "",
    })

    print(f"    Properties: {len(properties)}, Constraints: {len(constraints)}")
    if candidate:
        print(f"    Candidate: {candidate.get('material_name', 'N/A')}")
    print(f"    Status: {manufacturing.get('status', 'unknown')}")

    return build_ablation_evaluation(
        query=query, properties=properties, constraints=constraints,
        candidate=candidate, manufacturing=manufacturing,
        condition_name="1agent_no_rag",
        run_id=datetime.now().strftime("%Y%m%d%H"),
        duration_seconds=time.time() - start_time,
        raw_responses=raw_responses,
    )


# ---------------------------------------------------------------------------
# Resource loading for 1agent_rag (needs KGs + ChromaDB)
# ---------------------------------------------------------------------------
def _init_rag_resources(config):
    """Load KGs, ChromaDB, and material DB needed by the 1agent_rag condition."""
    from chromadb import PersistentClient
    from sentence_transformers import SentenceTransformer
    try:
        from GraphReasoning import load_embeddings
    except ImportError:
        raise ImportError("GraphReasoning is required for the 1agent_rag condition.")

    os.environ["TQDM_DISABLE"] = "1"

    emb_tokenizer = ""
    emb_model = SentenceTransformer(config["embeddings"]["model_name"], trust_remote_code=True)
    emb_fn = TransformerEmbeddingFunction(embedding_tokenizer=emb_tokenizer, embedding_model=emb_model)

    graphs_cfg = config["data"]["graphs"]
    kg_dir = graphs_cfg["kg_dir"]

    def _load_kg(cfg, label):
        gp = os.path.join(kg_dir, cfg["graph_file"])
        G = __import__("networkx").read_graphml(gp)
        rel = __import__("networkx").get_edge_attributes(G, "title")
        __import__("networkx").set_edge_attributes(G, rel, "relation")
        ep = os.path.join(kg_dir, cfg["embedding_file"])
        embs = load_embeddings(ep)
        print(f"  {label} KG: {G.number_of_nodes()} nodes")
        return G, embs

    G_mp, emb_mp = _load_kg(graphs_cfg["material_properties"], "MaterialProperties")
    G_pfas, emb_pfas = _load_kg(graphs_cfg["pfas"], "PFAS")
    G_pat, emb_pat = _load_kg(graphs_cfg["patents"], "Patents")

    chroma_cfg = config["data"]["chromadb"]
    base_path = chroma_cfg.get("base_path", "")

    def _load_col(db_cfg, label):
        dp = os.path.join(base_path, db_cfg["database_path"]) if base_path else db_cfg["database_path"]
        cl = PersistentClient(path=dp)
        cn = db_cfg.get("collection_name") or cl.list_collections()[0].name
        col = cl.get_collection(cn, embedding_function=emb_fn)
        print(f"  {label} ChromaDB loaded")
        return col

    pfas_col = _load_col(chroma_cfg["pfas"], "PFAS")
    patents_col = _load_col(chroma_cfg["patents"], "Patents")
    matdb_col = _load_col(chroma_cfg["materialdb"], "MaterialDB")

    mfg_cfg = chroma_cfg.get("manufacturing_textbooks")
    if not mfg_cfg:
        raise ValueError("config data.chromadb.manufacturing_textbooks is required for 1agent_rag")
    mfg_col = _load_col(mfg_cfg, "MfgTextbooks")

    ra_cfg = config["agents"]["research_analyst"]
    n_results_mfg = (
        config.get("pipelines", {})
        .get("manufacturability_assessment", {})
        .get("n_results_per_source", ra_cfg["n_results"])
    )
    rag_analysts = {
        "pfas": ResearchAnalyst(collection=pfas_col, embedding_function=emb_fn, n_results=ra_cfg["n_results"]),
        "patents": ResearchAnalyst(collection=patents_col, embedding_function=emb_fn, n_results=ra_cfg["n_results"]),
        "materialdb": ResearchAnalyst(collection=matdb_col, embedding_function=emb_fn, n_results=ra_cfg["n_results"]),
        "manufacturing_textbooks": ResearchAnalyst(
            collection=mfg_col, embedding_function=emb_fn, n_results=n_results_mfg,
        ),
    }

    scientists = {
        "material_properties": ResearchScientist(
            knowledge_graph=G_mp, node_embeddings=emb_mp,
            embedding_tokenizer=emb_tokenizer, embedding_model=emb_model, algorithm="shortest",
        ),
        "pfas": ResearchScientist(
            knowledge_graph=G_pfas, node_embeddings=emb_pfas,
            embedding_tokenizer=emb_tokenizer, embedding_model=emb_model, algorithm="shortest",
        ),
        "patents": ResearchScientist(
            knowledge_graph=G_pat, node_embeddings=emb_pat,
            embedding_tokenizer=emb_tokenizer, embedding_model=emb_model, algorithm="shortest",
        ),
    }

    pm = PropertyMapper(embedding_model=emb_model, embedding_tokenizer=emb_tokenizer)
    db_path = config.get("data", {}).get("material_database", {}).get("path", "")
    material_db = MaterialDatabase.load_from_json(db_path, property_mapper=pm)
    print(f"  Material DB: {len(material_db)} materials")

    return rag_analysts, scientists, material_db


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MARS Ablation Study — Run Conditions")
    parser.add_argument("--queries", default=None, help="Comma-separated query names")
    parser.add_argument(
        "--condition", default=None, choices=ALL_CONDITIONS,
        help="Single condition to run (default: all)",
    )
    parser.add_argument("--output-dir", default="results", help="Root output directory")
    args = parser.parse_args()

    config = load_config()
    prompts = load_prompts()
    ablation_prompts = prompts["ablation"]
    queries = load_ablation_queries()

    if args.queries:
        selected = {q.strip() for q in args.queries.split(",")}
        queries = [q for q in queries if q["name"] in selected]
        if not queries:
            print(f"ERROR: No matching queries found: {args.queries}")
            sys.exit(1)

    conditions = [args.condition] if args.condition else ALL_CONDITIONS

    # LLM (needed by all conditions)
    llm_cfg = config["llm"]
    llm_instance = llm({
        "api_key": llm_cfg["api_key"], "base_url": llm_cfg["base_url"],
        "model": llm_cfg["model_name"], "max_tokens": llm_cfg["max_tokens"],
    })
    generate = llm_instance.generate_cli
    temperature = llm_cfg.get("temperature", 0)

    # Extra resources for 1agent_rag
    rag_analysts = scientists = material_db = None
    if "1agent_rag" in conditions:
        print("Loading RAG/KG resources for 1agent_rag condition…")
        rag_analysts, scientists, material_db = _init_rag_resources(config)
        print()

    print(f"MARS Ablation Study")
    print(f"Queries:    {', '.join(q['name'] for q in queries)}")
    print(f"Conditions: {', '.join(conditions)}")
    print(f"Output:     {args.output_dir}/")
    print()

    for cond in conditions:
        for i, query in enumerate(queries, 1):
            name = query["name"]
            out_dir = str(PROJECT_ROOT / args.output_dir / name)
            os.makedirs(out_dir, exist_ok=True)

            print(f"{'='*70}")
            print(f"[{cond}] {name} ({i}/{len(queries)})")
            print(f"{'='*70}")
            print(f"  Query: {query['sentence'][:100]}…")

            if cond == "3agent":
                result = run_3agent_ablation(query, generate, ablation_prompts, temperature)
            elif cond == "1agent_rag":
                result = run_1agent_rag_ablation(
                    query, generate, ablation_prompts,
                    rag_analysts, scientists, material_db, temperature,
                )
            elif cond == "1agent_no_rag":
                result = run_1agent_no_rag_ablation(query, generate, ablation_prompts, temperature)
            else:
                continue

            out_path = os.path.join(out_dir, f"ablation_{cond}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)
            print(f"  Saved → {out_path}")
            print(f"  Duration: {result['metadata']['duration_seconds']:.1f}s")
            cand = result["candidate_selection"]["final_candidate"]
            print(f"  Candidate: {cand.get('material_name', 'N/A') if cand else 'N/A'}")
            print(f"  Mfg status: {result['manufacturing_process']['status']}")
            print()

    print("All ablation runs complete.")


if __name__ == "__main__":
    main()

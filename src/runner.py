"""MARS pipeline orchestrator.

Provides ``initialize`` to load all resources (KGs, ChromaDB, agents) from
config, and ``run_query`` to execute the full System 1 -> System 2 <-> System 3
loop for a single query.  These two functions are the public API consumed by
``scripts/run_mars.py`` and ``notebooks/walkthrough.ipynb``.
"""

import glob
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import networkx as nx
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

from .agents import (
    MultiAnalyst,
    RejectedCandidateTracker,
    ResearchAnalyst,
    ResearchAssistant,
    ResearchManager,
    ResearchScientist,
)
from .config import load_config, load_prompts
from .pipelines import (
    run_fixed_pipeline,
    run_manufacturability_assessment_pipeline,
    run_material_discovery_pipeline,
)
from .utils import (
    ChatLogger,
    MaterialDatabase,
    MaterialGrounding,
    PropertyMapper,
    TransformerEmbeddingFunction,
    llm,
    save_evaluation_export,
)

try:
    from GraphReasoning import load_embeddings
except ImportError:
    raise ImportError(
        "GraphReasoning is required but not installed. "
        "Install it with: pip install GraphReasoning"
    )

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Disable tqdm progress bars globally (reduces clutter in batch runs)
# ---------------------------------------------------------------------------
os.environ["TQDM_DISABLE"] = "1"
try:
    import tqdm as _tqdm

    _orig_init = _tqdm.tqdm.__init__

    def _disabled_init(self, *a, **kw):
        kw["disable"] = True
        return _orig_init(self, *a, **kw)

    _tqdm.tqdm.__init__ = _disabled_init
except Exception:
    pass


# ---------------------------------------------------------------------------
# Data container returned by ``initialize``
# ---------------------------------------------------------------------------
@dataclass
class MARSComponents:
    """All initialized pipeline components, ready for ``run_query``."""

    config: Dict[str, Any]
    generate: Callable
    embedding_model: Any
    embedding_tokenizer: str
    embedding_function: Any

    # Knowledge graphs
    G_materialproperties: nx.DiGraph = field(repr=False)
    node_embeddings_materialproperties: Dict = field(repr=False)
    G_pfas: nx.DiGraph = field(repr=False)
    node_embeddings_pfas: Dict = field(repr=False)
    G_patents: nx.DiGraph = field(repr=False)
    node_embeddings_patents: Dict = field(repr=False)

    # ChromaDB collections
    pfas_collection: Any = field(repr=False)
    patents_collection: Any = field(repr=False)
    materialdb_collection: Any = field(repr=False)

    # Agents that persist across queries
    analyst_patents_s2: ResearchAnalyst = field(repr=False)
    analyst_materialdb_s2: ResearchAnalyst = field(repr=False)
    scientist_s2: ResearchScientist = field(repr=False)
    process_analyst: MultiAnalyst = field(repr=False)

    # Material discovery helpers
    property_mapper: PropertyMapper = field(repr=False)
    material_db: MaterialDatabase = field(repr=False)
    material_grounding_material: MaterialGrounding = field(repr=False)
    material_grounding_patents: MaterialGrounding = field(repr=False)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------
def _require_dir(path: str, description: str) -> None:
    """Raise FileNotFoundError if *path* is not an existing directory."""
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"Required directory missing or not a directory ({description}): {path!r}"
        )


def _require_file(path: str, description: str) -> None:
    """Raise FileNotFoundError if *path* is not an existing file."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Required file missing or not a file ({description}): {path!r}"
        )


def _load_kg(kg_dir: str, cfg: dict, label: str, config_section: str) -> tuple:
    """Load a knowledge graph + its node embeddings from disk."""
    graph_path = os.path.join(kg_dir, cfg["graph_file"])
    emb_path = os.path.join(kg_dir, cfg["embedding_file"])
    _require_file(graph_path, f"{config_section} graph_file ({label})")
    _require_file(emb_path, f"{config_section} embedding_file ({label})")

    G = nx.read_graphml(graph_path)
    relation = nx.get_edge_attributes(G, "title")
    nx.set_edge_attributes(G, relation, "relation")
    print(f"  {label} KG loaded: {G}")

    embeddings = load_embeddings(emb_path)
    print(f"  {label} embeddings loaded: {len(embeddings)} nodes")
    return G, embeddings


def _load_chroma_collection(
    base_path: str,
    db_cfg: dict,
    ef,
    label: str,
    config_key: str,
) -> tuple:
    """Open a ChromaDB PersistentClient and return (client, collection).

    Fails fast if the DB directory is missing, empty of collections when
    ``collection_name`` is unset, or the collection cannot be opened.
    """
    db_path = os.path.join(base_path, db_cfg["database_path"]) if base_path else db_cfg["database_path"]
    _require_dir(db_path, f"data.chromadb.{config_key} (Chroma persist directory)")

    client = PersistentClient(path=db_path)
    col_name = db_cfg.get("collection_name")
    if not col_name:
        collections = client.list_collections()
        if not collections:
            raise RuntimeError(
                f"No collections in Chroma DB ({label}, config data.chromadb.{config_key}): {db_path!r}. "
                "Set collection_name explicitly or populate the database."
            )
        col_name = collections[0].name
    collection = client.get_collection(col_name, embedding_function=ef)
    print(f"  {label} ChromaDB loaded (collection={col_name!r})")
    return client, collection


def initialize(config: Optional[Dict[str, Any]] = None) -> MARSComponents:
    """Load every resource the MARS pipeline needs and return a
    ``MARSComponents`` instance.

    Args:
        config: Pre-loaded config dict.  If *None*, ``load_config()`` is called.
    """
    if config is None:
        config = load_config()

    # -- LLM -----------------------------------------------------------------
    llm_cfg = config["llm"]
    llm_instance = llm({
        "api_key": llm_cfg["api_key"],
        "base_url": llm_cfg["base_url"],
        "model": llm_cfg["model_name"],
        "max_tokens": llm_cfg["max_tokens"],
    })
    generate = llm_instance.generate_cli
    print("LLM wrapper initialized")

    # -- Embeddings -----------------------------------------------------------
    embedding_tokenizer = ""
    embedding_model = SentenceTransformer(
        config["embeddings"]["model_name"], trust_remote_code=True,
    )
    embedding_function = TransformerEmbeddingFunction(
        embedding_tokenizer=embedding_tokenizer,
        embedding_model=embedding_model,
    )
    print("Embedding model initialized")

    # -- Knowledge graphs -----------------------------------------------------
    graphs_cfg = config["data"]["graphs"]
    kg_dir = graphs_cfg["kg_dir"]
    _require_dir(kg_dir, "data.graphs.kg_dir")
    print("Loading knowledge graphs …")
    G_mp, emb_mp = _load_kg(
        kg_dir, graphs_cfg["material_properties"], "MaterialProperties", "graphs.material_properties",
    )
    G_pfas, emb_pfas = _load_kg(kg_dir, graphs_cfg["pfas"], "PFAS", "graphs.pfas")
    G_pat, emb_pat = _load_kg(kg_dir, graphs_cfg["patents"], "Patents", "graphs.patents")

    # -- ChromaDB -------------------------------------------------------------
    chroma_cfg = config["data"]["chromadb"]
    base_path = chroma_cfg.get("base_path", "")
    print("Loading ChromaDB collections …")

    _, pfas_col = _load_chroma_collection(
        base_path, chroma_cfg["pfas"], embedding_function, "PFAS", "pfas",
    )
    _, patents_col = _load_chroma_collection(
        base_path, chroma_cfg["patents"], embedding_function, "Patents", "patents",
    )
    _, matdb_col = _load_chroma_collection(
        base_path, chroma_cfg["materialdb"], embedding_function, "MaterialDB", "materialdb",
    )

    mfg_cfg = chroma_cfg.get("manufacturing_textbooks")
    if not mfg_cfg:
        raise ValueError("config data.chromadb.manufacturing_textbooks is required for MARS")
    _, mfg_col = _load_chroma_collection(
        base_path, mfg_cfg, embedding_function, "MfgTextbooks", "manufacturing_textbooks",
    )

    # Process analysts for System 3
    n_results_s3 = (
        config.get("pipelines", {})
        .get("manufacturability_assessment", {})
        .get("n_results_per_source", 5)
    )
    process_analysts: Dict[str, ResearchAnalyst] = {
        "manufacturing_textbooks": ResearchAnalyst(
            collection=mfg_col, embedding_function=embedding_function, n_results=n_results_s3,
        ),
        "patents": ResearchAnalyst(
            collection=patents_col, embedding_function=embedding_function, n_results=n_results_s3,
        ),
        "materialdb": ResearchAnalyst(
            collection=matdb_col, embedding_function=embedding_function, n_results=n_results_s3,
        ),
    }

    spec_cfg = chroma_cfg.get("spec_sheets") or {}
    if spec_cfg.get("enabled"):
        if not spec_cfg.get("database_path"):
            raise ValueError(
                "data.chromadb.spec_sheets.database_path is required when spec_sheets.enabled is true",
            )
        _, spec_col = _load_chroma_collection(
            base_path, spec_cfg, embedding_function, "SpecSheets", "spec_sheets",
        )
        process_analysts["spec_sheets"] = ResearchAnalyst(
            collection=spec_col, embedding_function=embedding_function, n_results=n_results_s3,
        )

    process_analyst = MultiAnalyst(process_analysts)
    print(f"  Process analyst initialized with {len(process_analysts)} sources")

    # -- System 2 persistent agents -------------------------------------------
    s2_cfg = config["pipelines"]["material_discovery"]
    ra_cfg = config["agents"]["research_analyst"]

    analyst_patents_s2 = ResearchAnalyst(
        collection=patents_col, embedding_function=embedding_function,
        n_results=ra_cfg["n_results"],
    )
    analyst_materialdb_s2 = ResearchAnalyst(
        collection=matdb_col, embedding_function=embedding_function,
        n_results=ra_cfg["n_results"],
    )
    scientist_s2 = ResearchScientist(
        knowledge_graph=G_mp,
        node_embeddings=emb_mp,
        embedding_tokenizer=embedding_tokenizer,
        embedding_model=embedding_model,
        algorithm="shortest",
        generate_fn=generate,
        knowledge_graph_2=G_pat,
        node_embeddings_2=emb_pat,
        kg_names=["material_properties", "patents"],
        kg_descriptions=[
            "Material properties and characteristics knowledge graph",
            "Patent knowledge graph with materials and related technologies",
        ],
        multi_kg_strategy="separate",
    )

    # -- Material DB / grounding / mapper -------------------------------------
    property_mapper = PropertyMapper(
        embedding_model=embedding_model, embedding_tokenizer=embedding_tokenizer,
    )
    mat_db_path = (
        config.get("data", {}).get("material_database", {}).get("path", "./data/internal_material_database.json")
    )
    _require_file(mat_db_path, "data.material_database.path")
    material_db = MaterialDatabase.load_from_json(mat_db_path, property_mapper=property_mapper)
    print(f"Material database loaded: {len(material_db)} materials")

    mg_material = MaterialGrounding(
        knowledge_graph=G_mp, node_embeddings=emb_mp,
        embedding_model=embedding_model, embedding_tokenizer=embedding_tokenizer,
    )
    mg_patents = MaterialGrounding(
        knowledge_graph=G_pat, node_embeddings=emb_pat,
        embedding_model=embedding_model, embedding_tokenizer=embedding_tokenizer,
    )

    print("All components initialized")

    return MARSComponents(
        config=config,
        generate=generate,
        embedding_model=embedding_model,
        embedding_tokenizer=embedding_tokenizer,
        embedding_function=embedding_function,
        G_materialproperties=G_mp,
        node_embeddings_materialproperties=emb_mp,
        G_pfas=G_pfas,
        node_embeddings_pfas=emb_pfas,
        G_patents=G_pat,
        node_embeddings_patents=emb_pat,
        pfas_collection=pfas_col,
        patents_collection=patents_col,
        materialdb_collection=matdb_col,
        analyst_patents_s2=analyst_patents_s2,
        analyst_materialdb_s2=analyst_materialdb_s2,
        scientist_s2=scientist_s2,
        process_analyst=process_analyst,
        property_mapper=property_mapper,
        material_db=material_db,
        material_grounding_material=mg_material,
        material_grounding_patents=mg_patents,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _generate_run_id(output_dir: str, prefix: str = "system1") -> str:
    """YYYYMMDDHH_N run-id format."""
    base = datetime.now().strftime("%Y%m%d%H")
    pattern = os.path.join(output_dir, f"{prefix}_{base}_*.json")
    existing = glob.glob(pattern)
    max_ctr = -1
    for fp in existing:
        parts = os.path.basename(fp).replace(".json", "").split("_")
        if len(parts) >= 3:
            try:
                max_ctr = max(max_ctr, int(parts[-1]))
            except ValueError:
                pass
    return f"{base}_{max_ctr + 1}"


def _next_counter(output_dir: str, prefix: str, base_id: str) -> str:
    pattern = os.path.join(output_dir, f"{prefix}_{base_id}_*.json")
    existing = glob.glob(pattern)
    mx = -1
    for fp in existing:
        try:
            mx = max(mx, int(os.path.basename(fp).replace(".json", "").split("_")[-1]))
        except (ValueError, IndexError):
            pass
    return f"{base_id}_{mx + 1}"


# ---------------------------------------------------------------------------
# Full query execution
# ---------------------------------------------------------------------------
def run_query(
    components: MARSComponents,
    query: Dict[str, str],
    output_dir: str,
) -> Dict[str, Any]:
    """Execute the full MARS pipeline (System 1 -> System 2 <-> System 3) for
    one query and write all artefacts to *output_dir*.

    Args:
        components: Initialised ``MARSComponents`` from ``initialize()``.
        query: Dict with keys ``sentence``, ``material_X``, ``application_Y``.
        output_dir: Directory for this query's outputs (e.g. ``results/Query1``).

    Returns:
        The completed ``pipeline_run`` metadata dict.
    """
    c = components
    config = c.config
    os.makedirs(output_dir, exist_ok=True)
    artifacts_dir = os.path.join(output_dir, "artifacts")
    chats_dir = os.path.join(artifacts_dir, "chats")
    subgraphs_dir = os.path.join(artifacts_dir, "subgraphs")
    os.makedirs(chats_dir, exist_ok=True)
    os.makedirs(subgraphs_dir, exist_ok=True)

    sentence = query["sentence"]
    material_X = query["material_X"]
    application_Y = query["application_Y"]
    keywords = [material_X, application_Y]

    base_run_id = _generate_run_id(artifacts_dir, "system1").split("_")[0]

    # -- Pipeline run tracking ------------------------------------------------
    pipeline_start = datetime.utcnow()
    pipeline_run: Dict[str, Any] = {
        "pipeline_run_id": base_run_id,
        "start_time": pipeline_start.isoformat() + "Z",
        "end_time": None,
        "total_duration_seconds": None,
        "system1": {
            "run_id": None, "start_time": None, "end_time": None,
            "duration_seconds": None, "result_path": None,
            "chat_log_path": None, "properties_extracted": None,
        },
        "system2_system3_loop": {
            "max_iterations": None, "total_iterations": 0, "iterations": [],
        },
        "final_outcome": {
            "status": None, "final_candidate": None,
            "total_rejected_candidates": None,
        },
    }

    print("=" * 70)
    print(f"Running MARS pipeline — {query.get('name', 'query')}")
    print("=" * 70)
    print(f"Query: {sentence[:120]}…")
    print(f"Material X: {material_X}")
    print(f"Application Y: {application_Y}")
    print()

    # =========================================================================
    # System 1 — Property Extraction
    # =========================================================================
    print("=" * 70)
    print("System 1: Property Extraction")
    print("=" * 70)

    s1_cfg = config["pipelines"]["material_requirements"]
    s1_run_id = _generate_run_id(artifacts_dir, "system1")
    chat_logger_s1 = ChatLogger(
        run_id=s1_run_id, pipeline="material_requirements", log_dir=chats_dir
    )

    analyst_s1 = ResearchAnalyst(
        collection=c.pfas_collection, embedding_function=c.embedding_function,
        n_results=s1_cfg["n_results"], chat_logger=chat_logger_s1,
    )
    manager_s1 = ResearchManager(
        name="research_manager", system_message=None,
        generate_fn=c.generate, chat_logger=chat_logger_s1,
    )
    assistant_s1 = ResearchAssistant(
        name="research_assistant", system_message=None,
        generate_fn=c.generate, chat_logger=chat_logger_s1,
    )
    pfas_scientist_s1 = ResearchScientist(
        knowledge_graph=c.G_pfas, node_embeddings=c.node_embeddings_pfas,
        embedding_tokenizer=c.embedding_tokenizer,
        embedding_model=c.embedding_model,
        algorithm="shortest", chat_logger=chat_logger_s1,
    )

    s1_start = datetime.utcnow()
    pipeline_run["system1"]["start_time"] = s1_start.isoformat() + "Z"

    system1_result = run_fixed_pipeline(
        sentence=sentence, keywords=keywords,
        analyst=analyst_s1, manager=manager_s1,
        research_assistant=assistant_s1,
        scientist=pfas_scientist_s1,
        pfas_scientist=pfas_scientist_s1,
        include_rag_context=s1_cfg["include_rag_context"],
        max_items=s1_cfg["max_items"],
        temperature=config["llm"]["temperature"],
        n_results=s1_cfg["n_results"],
        chat_logger=chat_logger_s1,
    )

    s1_end = datetime.utcnow()
    s1_dur = (s1_end - s1_start).total_seconds()
    pipeline_run["system1"].update({
        "run_id": s1_run_id,
        "end_time": s1_end.isoformat() + "Z",
        "duration_seconds": s1_dur,
    })

    extracted_keywords = system1_result.get("extracted_keywords", []) or []
    extracted_constraints = system1_result.get("extracted_constraints", []) or []
    properties_W = {"required": extracted_keywords, "target_values": {}}

    # Save System 1 output
    s1_payload = {
        "run_id": s1_run_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "sentence": system1_result.get("sentence"),
        "keywords": system1_result.get("keywords", []) or [],
        "material_X": material_X,
        "application_Y": application_Y,
        "properties_W": properties_W,
        "extracted_keywords": extracted_keywords,
        "extracted_constraints": extracted_constraints,
        "num_keywords": len(extracted_keywords),
        "num_constraints": len(extracted_constraints),
        "chat_log_path": system1_result.get("chat_log_path"),
    }
    s1_path = os.path.join(artifacts_dir, f"system1_{s1_run_id}.json")
    with open(s1_path, "w", encoding="utf-8") as f:
        json.dump(s1_payload, f, indent=2, ensure_ascii=False, default=str)

    pipeline_run["system1"]["result_path"] = s1_path
    pipeline_run["system1"]["chat_log_path"] = system1_result.get("chat_log_path")
    pipeline_run["system1"]["properties_extracted"] = len(extracted_keywords)

    print(f"System 1 complete — {len(extracted_keywords)} properties, "
          f"{len(extracted_constraints)} constraints ({s1_dur:.0f}s)")

    # =========================================================================
    # System 2 <-> System 3 loop
    # =========================================================================
    max_iterations = config["pipelines"]["material_discovery"].get("max_iterations", 5)
    pipeline_run["system2_system3_loop"]["max_iterations"] = max_iterations

    tracker = RejectedCandidateTracker(
        log_file=os.path.join(artifacts_dir, "rejected_candidates.json")
    )
    constraints_U = list(extracted_constraints)
    cached_substitution_result = None
    s1_base = s1_run_id.split("_")[0] if "_" in s1_run_id else s1_run_id
    all_iterations: List[Dict[str, Any]] = []

    print()
    print("=" * 70)
    print(f"Starting System 2 <-> System 3 loop (max {max_iterations} iterations)")
    print("=" * 70)

    for iteration in range(1, max_iterations + 1):
        print(f"\n--- Iteration {iteration}/{max_iterations} ---")

        # -- System 2 --------------------------------------------------------
        s2_run_id = _next_counter(artifacts_dir, "system2", s1_base)
        chat_logger_s2 = ChatLogger(
            run_id=s2_run_id, pipeline="material_discovery", log_dir=chats_dir
        )

        c.analyst_patents_s2.chat_logger = chat_logger_s2
        c.analyst_materialdb_s2.chat_logger = chat_logger_s2
        c.scientist_s2.chat_logger = chat_logger_s2

        analyst_s2 = MultiAnalyst({"patents": c.analyst_patents_s2, "materialdb": c.analyst_materialdb_s2})
        manager_s2 = ResearchManager(
            name="research_manager", system_message=None,
            generate_fn=c.generate, chat_logger=chat_logger_s2,
        )

        s2_start = datetime.utcnow()

        system2_result = run_material_discovery_pipeline(
            material_X=material_X, application_Y=application_Y,
            properties_W=properties_W, constraints_U=constraints_U,
            analyst=analyst_s2, manager=manager_s2, scientist=c.scientist_s2,
            tracker=tracker,
            max_iterations=config["pipelines"]["material_discovery"]["max_iterations"],
            temperature=config["llm"]["temperature"],
            chat_logger=chat_logger_s2,
            material_db=c.material_db,
            material_grounding_material=c.material_grounding_material,
            material_grounding_patents=c.material_grounding_patents,
            knowledge_graph_material=c.G_materialproperties,
            knowledge_graph_patents=c.G_patents,
            substitution_result=cached_substitution_result,
            subgraphs_dir=subgraphs_dir,
        )

        s2_end = datetime.utcnow()
        s2_dur = (s2_end - s2_start).total_seconds()

        # Save System 2
        s2_out = os.path.join(artifacts_dir, f"system2_{s2_run_id}.json")
        with open(s2_out, "w", encoding="utf-8") as f:
            json.dump({
                "run_id": s2_run_id,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "application": application_Y,
                "properties": properties_W,
                "constraints": constraints_U,
                "success": system2_result.get("success", False),
                "candidate": system2_result.get("candidate"),
                "iterations": system2_result.get("iterations", 0),
                "rejected_candidates": system2_result.get("rejected_candidates", []),
                "final_constraints": system2_result.get("final_constraints", []),
                "evidence_summary": system2_result.get("evidence_summary", {}),
                "iteration_history": system2_result.get("iteration_history", []),
                "property_mapping": system2_result.get("property_mapping", {}),
                "chat_log_path": system2_result.get("chat_log_path"),
            }, f, indent=2, ensure_ascii=False, default=str)

        print(f"  System 2 done ({s2_dur:.0f}s)")

        if iteration == 1 and cached_substitution_result is None:
            cached_substitution_result = system2_result.get("substitution_result")

        if not system2_result.get("success"):
            print("  System 2 failed to find a candidate")
            all_iterations.append({
                "iteration": iteration, "system2_run_id": s2_run_id,
                "system2_success": False, "system3_run_id": None, "system3_status": None,
            })
            break

        candidate_name = system2_result.get("candidate", {}).get("material_name", "Unknown")
        print(f"  Candidate: {candidate_name}")

        # -- System 3 --------------------------------------------------------
        s3_run_id = _next_counter(artifacts_dir, "system3", s1_base)
        chat_logger_s3 = ChatLogger(
            run_id=s3_run_id,
            pipeline="manufacturability_assessment",
            log_dir=chats_dir,
        )
        manager_s3 = ResearchManager(
            name="research_manager", system_message=None,
            generate_fn=c.generate, chat_logger=chat_logger_s3,
        )

        s3_start = datetime.utcnow()

        system3_result = run_manufacturability_assessment_pipeline(
            system2_result=system2_result,
            initial_query=sentence,
            material_X=material_X,
            application_Y=application_Y,
            constraints_U=constraints_U,
            tracker=tracker,
            process_analyst=c.process_analyst,
            manager=manager_s3,
            properties_W=properties_W,
            config=config,
            temperature=0,
            chat_logger=chat_logger_s3,
        )

        s3_end = datetime.utcnow()
        s3_dur = (s3_end - s3_start).total_seconds()

        # Save System 3
        s3_out = os.path.join(artifacts_dir, f"system3_{s3_run_id}.json")
        system3_result["run_id"] = s3_run_id
        with open(s3_out, "w", encoding="utf-8") as f:
            json.dump(system3_result, f, indent=2, ensure_ascii=False, default=str)

        status = system3_result.get("status", "")
        print(f"  System 3 done ({s3_dur:.0f}s) — status: {status}")

        # -- Record iteration -------------------------------------------------
        all_iterations.append({
            "iteration": iteration, "system2_run_id": s2_run_id,
            "system2_success": True, "candidate": candidate_name,
            "system3_run_id": s3_run_id, "system3_status": status,
            "manufacturable": system3_result.get("manufacturable", False),
        })

        # Build internal iteration timing (best-effort from chat log)
        internal_iterations: List[Dict] = []
        try:
            it_hist = system2_result.get("iteration_history", [])
            for idx, h in enumerate(it_hist):
                frac = s2_dur / max(len(it_hist), 1)
                internal_iterations.append({
                    "iteration": h.get("iteration", idx + 1),
                    "start_time": (s2_start + timedelta(seconds=frac * idx)).isoformat() + "Z",
                    "end_time": (s2_start + timedelta(seconds=frac * (idx + 1))).isoformat() + "Z",
                    "duration_seconds": frac,
                    "candidate": (h.get("candidate") or {}).get("material_name", "Unknown"),
                    "status": "accepted" if h.get("feasible") else "rejected",
                })
        except Exception:
            pass

        iteration_data = {
            "iteration": iteration,
            "system2": {
                "run_id": s2_run_id,
                "start_time": s2_start.isoformat() + "Z",
                "end_time": s2_end.isoformat() + "Z",
                "duration_seconds": s2_dur,
                "result_path": s2_out,
                "chat_log_path": system2_result.get("chat_log_path"),
                "candidate": candidate_name,
                "internal_iterations": internal_iterations,
            },
            "system3": {
                "run_id": s3_run_id,
                "start_time": s3_start.isoformat() + "Z",
                "end_time": s3_end.isoformat() + "Z",
                "duration_seconds": s3_dur,
                "result_path": s3_out,
                "chat_log_path": system3_result.get("chat_log_path"),
                "status": status,
                "blocking_constraints": system3_result.get("blocking_constraints", []),
                "feedback_to_system2": system3_result.get("feedback_to_system2", ""),
            },
        }
        pipeline_run["system2_system3_loop"]["iterations"].append(iteration_data)
        pipeline_run["system2_system3_loop"]["total_iterations"] = len(
            pipeline_run["system2_system3_loop"]["iterations"]
        )

        if status == "manufacturable":
            print("\n  SUCCESS — manufacturable candidate found")
            break

        if status == "blocked":
            feedback = system3_result.get("feedback_to_system2", "")
            blocking = system3_result.get("blocking_constraints", [])
            print(f"  Blocked — feeding constraints back to System 2")

            fh = pipeline_run["system2_system3_loop"].setdefault("system3_feedback_history", [])
            fh.append({
                "iteration": iteration,
                "system3_run_id": s3_run_id,
                "blocking_constraints": blocking,
                "feedback_to_system2": feedback,
            })

            existing_norm = {str(x).strip().lower() for x in constraints_U if str(x).strip()}
            new_constraints = []
            for bc in blocking:
                if isinstance(bc, dict):
                    ctype = str(bc.get("type", "missing_critical_info")).strip()
                    desc = str(bc.get("description", "")).strip()
                else:
                    ctype = "missing_critical_info"
                    desc = str(bc).strip()
                if not desc:
                    continue
                compact = f"S3[{ctype}]: {desc[:220]}"
                if compact.lower() not in existing_norm:
                    new_constraints.append(compact)
                    existing_norm.add(compact.lower())
            if feedback:
                fc = f"S3 feedback: {feedback[:220]}"
                if fc.lower() not in existing_norm:
                    new_constraints.append(fc)
            constraints_U.extend(new_constraints)

    # =========================================================================
    # Finalize pipeline_run and save
    # =========================================================================
    pipeline_end = datetime.utcnow()
    total_dur = (pipeline_end - pipeline_start).total_seconds()
    pipeline_run["end_time"] = pipeline_end.isoformat() + "Z"
    pipeline_run["total_duration_seconds"] = total_dur

    if all_iterations:
        last = all_iterations[-1]
        st = last.get("system3_status", "unknown")
        if st == "manufacturable":
            pipeline_run["final_outcome"]["status"] = "manufacturable"
        elif st == "blocked" and len(all_iterations) >= max_iterations:
            pipeline_run["final_outcome"]["status"] = "max_iterations_reached"
        else:
            pipeline_run["final_outcome"]["status"] = st
        pipeline_run["final_outcome"]["final_candidate"] = last.get("candidate")
    else:
        pipeline_run["final_outcome"]["status"] = "no_iterations_completed"
    pipeline_run["final_outcome"]["total_rejected_candidates"] = len(tracker.get_all_rejected())

    # Save tracker
    tracker_path = os.path.join(artifacts_dir, "rejected_candidates.json")
    with open(tracker_path, "w", encoding="utf-8") as f:
        json.dump({"rejected_candidates": tracker.get_all_rejected()}, f, indent=2, default=str)

    # Save pipeline_run
    pr_path = os.path.join(artifacts_dir, f"pipeline_run_{base_run_id}.json")
    with open(pr_path, "w", encoding="utf-8") as f:
        json.dump(pipeline_run, f, indent=2, ensure_ascii=False, default=str)

    # Save evaluation export (mars.json in parent dir)
    eval_path = save_evaluation_export(pipeline_run, artifacts_dir)
    if eval_path:
        mars_json = os.path.join(output_dir, "mars.json")
        with open(eval_path, "r", encoding="utf-8") as src:
            data = json.load(src)
        with open(mars_json, "w", encoding="utf-8") as dst:
            json.dump(data, dst, indent=2, ensure_ascii=False, default=str)
        print(f"Evaluation export → {mars_json}")

    print(f"\nPipeline complete — {total_dur:.0f}s "
          f"({total_dur/60:.1f} min), status: {pipeline_run['final_outcome']['status']}")

    return pipeline_run

"""Material Discovery Pipeline Function"""

import logging
from typing import List, Dict, Any, Optional, Callable
import re
import json
import networkx as nx

logger = logging.getLogger(__name__)
from ..agents import ResearchAnalyst, ResearchManager, ResearchScientist, RejectedCandidateTracker
from ..utils.material_database import MaterialDatabase
from ..utils.material_grounding import MaterialGrounding
from ..utils.dual_kg_subgraph import (
    KgMappingCaps,
    build_connection_subgraph_shortest_paths,
    map_terms_to_nodes_best_match,
    merge_subgraphs_unify_by_embedding,
)
from ..utils.subgraph_storage import SubgraphStorage
from ..config import load_prompts, load_config
from .material_requirements import _clean_extracted_keywords
from ..utils.parsing import clean_material_name


def _req(d: Dict[str, Any], key: str, section: str = "") -> Any:
    """Retrieve a required config key, raising ``KeyError`` if missing."""
    if key not in d:
        prefix = f"{section}." if section else ""
        raise KeyError(f"Missing required config key: {prefix}{key}")
    return d[key]


def extract_subgraph_insights(
    subgraph: nx.DiGraph,
    material_X: str,
    properties_W: Dict[str, Any],
    generate_fn: Callable,
    batch_size: Optional[int] = None,
    temperature: Optional[float] = None,
    chat_logger: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Extract key insights from material-informed subgraph using LLM-based classification.
    
    Args:
        subgraph: Material-informed subgraph
        material_X: Material to be replaced
        properties_W: Property requirements dictionary
        generate_fn: LLM generate function (required)
        batch_size: Number of nodes to process per LLM call (default: None, uses config value)
        temperature: Temperature for LLM generation (default: None, uses config value)
        chat_logger: Optional chat logger for tracking LLM calls
        
    Returns:
        Dictionary containing subgraph insights
    """
    # Load config
    config = load_config()
    pipeline_config = config.get("pipelines", {}).get("material_discovery", {})
    
    # Use config defaults if not provided
    if batch_size is None:
        batch_size = _req(pipeline_config, "batch_size", "pipelines.material_discovery")
    if temperature is None:
        temperature = _req(pipeline_config, "temperature", "pipelines.material_discovery")
    
    if subgraph is None or subgraph.number_of_nodes() == 0:
        return {
            'material_nodes': [],
            'property_nodes': [],
            'total_nodes': 0,
            'total_edges': 0
        }
    
    # Load prompt from config
    prompts = load_prompts()
    system_prompt = prompts.get("pipelines", {}).get("material_discovery", {}).get("extract_subgraph_insights")
    if system_prompt is None:
        raise ValueError(
            "Missing required prompt in config/prompts.yaml: pipelines.material_discovery.extract_subgraph_insights"
        )
    
    # Get required properties
    required_properties = properties_W.get('required', [])
    required_properties_str = ', '.join(required_properties) if required_properties else 'None'
    
    # Format system prompt with context
    formatted_system_prompt = system_prompt.format(
        material_X=material_X,
        required_properties=required_properties_str
    )
    
    # Collect all nodes with their attributes
    all_nodes = list(subgraph.nodes())
    material_nodes = []
    property_nodes = []
    
    # Process nodes in batches
    for batch_start in range(0, len(all_nodes), batch_size):
        batch_nodes = all_nodes[batch_start:batch_start + batch_size]
        
        # Format node data for the prompt
        node_data_list = []
        for node in batch_nodes:
            node_attrs = subgraph.nodes[node] if node in subgraph.nodes() else {}
            
            # Build node description from attributes
            node_info = {
                'node_id': str(node),
                'attributes': {}
            }
            
            # Extract relevant attributes
            for attr_key in ["title", "label", "name", "description"]:
                if attr_key in node_attrs:
                    node_info['attributes'][attr_key] = str(node_attrs[attr_key])
            
            node_data_list.append(node_info)
        
        # Load user prompt template from YAML
        extract_subgraph_user_prompt_template = prompts.get("pipelines", {}).get("material_discovery", {}).get("extract_subgraph_insights_user_prompt")
        if extract_subgraph_user_prompt_template is None:
            raise ValueError(
                "Missing required prompt in config/prompts.yaml: pipelines.material_discovery.extract_subgraph_insights_user_prompt. "
                "All system prompts must be defined in the config file."
            )
        
        # Build node list
        node_list_parts = []
        for i, node_info in enumerate(node_data_list, 1):
            node_id = node_info['node_id']
            attrs = node_info['attributes']
            attr_str = ', '.join([f"{k}: {v}" for k, v in attrs.items()])
            if attr_str:
                node_list_parts.append(f"{i}. Node ID: {node_id}, Attributes: {attr_str}")
            else:
                node_list_parts.append(f"{i}. Node ID: {node_id}")
        node_list = "\n".join(node_list_parts)
        
        # Format user prompt with dynamic content
        user_prompt = extract_subgraph_user_prompt_template.format(
            num_nodes=len(batch_nodes),
            node_list=node_list
        )
        
        # Call LLM
        try:
            # Note: chat_logger is already handled by the wrapped generate_fn, so don't pass it explicitly
            response = generate_fn(
                system_prompt=formatted_system_prompt,
                prompt=user_prompt,
                temperature=temperature,
                agent_name="extract_subgraph_insights",
                method_name="classify_nodes_batch"
            )
            
            # Parse JSON response
            # Try to extract JSON from response (in case there's extra text)
            response = response.strip()
            
            # Try to find JSON object in response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                result = json.loads(json_str)
                
                # Extract classified nodes
                batch_material_nodes = result.get('material_nodes', [])
                batch_property_nodes = result.get('property_nodes', [])
                
                # Convert node IDs back to original node objects
                node_id_to_node = {str(node): node for node in batch_nodes}
                
                for node_id_str in batch_material_nodes:
                    if node_id_str in node_id_to_node:
                        material_nodes.append(node_id_to_node[node_id_str])
                
                for node_id_str in batch_property_nodes:
                    if node_id_str in node_id_to_node:
                        property_nodes.append(node_id_to_node[node_id_str])
            else:
                print(f"Warning: Could not parse JSON from LLM response in batch {batch_start // batch_size + 1}")
        
        except json.JSONDecodeError as e:
            print(f"Warning: JSON parsing error in batch {batch_start // batch_size + 1}: {e}")
            if 'response' in locals():
                print(f"Response was: {response[:200]}...")
        except Exception as e:
            print(f"Warning: Error processing batch {batch_start // batch_size + 1}: {e}")
            # Continue with next batch
    
    max_nodes_for_context = _req(pipeline_config, "batch_nodes_for_llm_context", "pipelines.material_discovery")
    return {
        'material_nodes': material_nodes[:max_nodes_for_context],
        'property_nodes': property_nodes[:max_nodes_for_context],
        'total_nodes': subgraph.number_of_nodes(),
        'total_edges': subgraph.number_of_edges()
    }


def run_material_substitution_step(
    material_X: str,
    application_Y: str,
    properties_W: Dict[str, Any],
    constraints_U: List[str],
    material_db: MaterialDatabase,
    knowledge_graph_material: nx.DiGraph,
    knowledge_graph_patents: nx.DiGraph,
    material_grounding_material: MaterialGrounding,
    material_grounding_patents: MaterialGrounding,
    scientist: ResearchScientist,
    temperature: Optional[float] = None,
    run_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run the material substitution step (dual-KG canonical path).

    Builds a material-informed subgraph from both the Material Properties KG
    and the Patents KG, grounds lab materials, and extracts material classes.

    Args:
        material_X: Material to be replaced.
        application_Y: Target application description.
        properties_W: Required material properties (dict with ``required`` key).
        constraints_U: List of constraint strings.
        material_db: MaterialDatabase instance.
        knowledge_graph_material: Material Properties knowledge graph (DiGraph).
        knowledge_graph_patents: Patents knowledge graph (DiGraph).
        material_grounding_material: MaterialGrounding bound to the Material Properties KG.
        material_grounding_patents: MaterialGrounding bound to the Patents KG.
        scientist: ResearchScientist used for material-class extraction from the subgraph.
        temperature: LLM temperature (default: ``None``, uses config value).
        run_id: Pipeline run identifier used for subgraph persistence.

    Returns:
        Dict with ``ranked_candidates`` (empty), ``material_informed_subgraph``,
        ``kg_material_mapping``, ``property_mapping``, ``material_grounding``,
        and ``subgraph_storage_path``.
    """
    # Load config
    config = load_config()
    pipeline_config = config.get("pipelines", {}).get("material_discovery", {})
    
    if temperature is None:
        temperature = _req(pipeline_config, "temperature", "pipelines.material_discovery")
    
    print("\n" + "=" * 70)
    print("Step 1: Material Substitution (Re-engineered)")
    print("=" * 70)
    print(f"\nInput Parameters:")
    print(f"  Material to replace (X): {material_X}")
    print(f"  Application (Y): {application_Y}")
    print(f"  Required Properties: {properties_W.get('required', [])}")
    if properties_W.get('target_values'):
        print(f"  Target Values:")
        for prop, value in properties_W.get('target_values', {}).items():
            print(f"    - {prop}: {value}")
    print(f"  Constraints: {len(constraints_U)} constraint(s)")
    if constraints_U:
        for constraint in constraints_U:
            print(f"    - {constraint}")
    print(f"  Temperature: {temperature}")
    print(f"  Material Properties KG: {knowledge_graph_material.number_of_nodes()} nodes, {knowledge_graph_material.number_of_edges()} edges")
    print(f"  Patents KG: {knowledge_graph_patents.number_of_nodes()} nodes, {knowledge_graph_patents.number_of_edges()} edges")
    print(f"  Material Database: {len(material_db)} materials")
    
    # Canonical dual-KG material-informed subgraph construction (single allowed path).
    # This step does not use System 1's serialized subgraph_data.
    print("\n" + "-" * 70)
    print("Phase 1.1: Dual-KG material-informed subgraph (canonical)")
    print("-" * 70)

    dual_cfg = pipeline_config.get("dual_kg_material_informed_subgraph", {})
    _section = "pipelines.material_discovery.dual_kg_material_informed_subgraph"
    caps = KgMappingCaps(
        max_property_terms=int(_req(dual_cfg, "max_property_terms", _section)),
        max_materials=int(_req(dual_cfg, "max_materials", _section)),
        max_nodes_total=int(_req(dual_cfg, "max_nodes_total", _section)),
        max_pairs_evaluated=int(_req(dual_cfg, "max_pairs_evaluated", _section)),
        max_shortest_path_len=int(_req(dual_cfg, "max_shortest_path_len", _section)),
    )
    merge_threshold = float(_req(dual_cfg, "merge_similarity_threshold", _section))
    mapping_n_samples = int(_req(dual_cfg, "n_samples", _section))
    mapping_similarity_threshold = float(_req(dual_cfg, "similarity_threshold", _section))

    # Ground lab materials in MatKG first (reused for seed nodes and downstream prompts)
    print("\n  Phase 1.1a: Ground lab materials in MatKG")
    material_grounding_map = material_grounding_material.ground_material_database(material_db)
    print(f"  [OK] Grounded {len(material_grounding_map)} materials in MatKG")

    min_matches = _req(pipeline_config, "min_matches_for_multiple_grounding", "pipelines.material_discovery")
    if material_grounding_map:
        sample_materials = list(material_grounding_map.keys())[:3]
        for mat_id in sample_materials:
            matches = material_grounding_map[mat_id]
            print(f"    - {mat_id}: {len(matches)} match(es)")
            for i, match in enumerate(matches[:min_matches], 1):
                print(f"        {i}. {match.get('node_id', 'N/A')} (sim={match.get('similarity', 0.0):.3f})")
    else:
        print("  [WARNING] No materials were successfully grounded in MatKG")

    required_props = properties_W.get("required", []) or []

    prop_nodes_matkg = map_terms_to_nodes_best_match(
        required_props,
        material_grounding_material.node_embeddings,
        material_grounding_material.embedding_tokenizer,
        material_grounding_material.embedding_model,
        n_samples=mapping_n_samples,
        similarity_threshold=mapping_similarity_threshold,
        max_terms=caps.max_property_terms,
    )
    prop_nodes_patkg = map_terms_to_nodes_best_match(
        required_props,
        material_grounding_patents.node_embeddings,
        material_grounding_patents.embedding_tokenizer,
        material_grounding_patents.embedding_model,
        n_samples=mapping_n_samples,
        similarity_threshold=mapping_similarity_threshold,
        max_terms=caps.max_property_terms,
    )

    # Extract MatKG material seed nodes from the grounding map (no redundant calls)
    material_nodes_matkg: List[str] = []
    for matches in material_grounding_map.values():
        if matches:
            material_nodes_matkg.append(matches[0]["node_id"])

    # PatKG material seed nodes still need individual grounding
    all_materials = material_db.get_all_materials()
    materials_capped = all_materials[: caps.max_materials]
    material_nodes_patkg: List[str] = []
    for m in materials_capped:
        name = m.get("material_name")
        if not name:
            continue
        pat_matches = material_grounding_patents.ground_material(name)
        if pat_matches:
            material_nodes_patkg.append(pat_matches[0]["node_id"])

    seed_nodes_matkg = list(dict.fromkeys(prop_nodes_matkg + material_nodes_matkg))
    seed_nodes_patkg = list(dict.fromkeys(prop_nodes_patkg + material_nodes_patkg))

    print(f"  → MatKG seeds: {len(seed_nodes_matkg)} (props={len(prop_nodes_matkg)}, materials={len(material_nodes_matkg)})")
    print(f"  → PatKG seeds: {len(seed_nodes_patkg)} (props={len(prop_nodes_patkg)}, materials={len(material_nodes_patkg)})")

    subgraph_matkg = build_connection_subgraph_shortest_paths(
        knowledge_graph_material,
        seed_nodes_matkg,
        max_pairs_evaluated=caps.max_pairs_evaluated,
        max_shortest_path_len=caps.max_shortest_path_len,
        max_nodes_total=caps.max_nodes_total,
    )
    subgraph_patkg = build_connection_subgraph_shortest_paths(
        knowledge_graph_patents,
        seed_nodes_patkg,
        max_pairs_evaluated=caps.max_pairs_evaluated,
        max_shortest_path_len=caps.max_shortest_path_len,
        max_nodes_total=caps.max_nodes_total,
    )

    print(f"  [OK] MatKG subgraph: {subgraph_matkg.number_of_nodes()} nodes, {subgraph_matkg.number_of_edges()} edges")
    print(f"  [OK] PatKG subgraph: {subgraph_patkg.number_of_nodes()} nodes, {subgraph_patkg.number_of_edges()} edges")

    material_informed_subgraph, cross_kg_mapping = merge_subgraphs_unify_by_embedding(
        subgraph_matkg,
        subgraph_patkg,
        material_grounding_material.node_embeddings,
        material_grounding_patents.node_embeddings,
        similarity_threshold=merge_threshold,
    )
    print(f"  [OK] Merged material-informed subgraph: {material_informed_subgraph.number_of_nodes()} nodes, {material_informed_subgraph.number_of_edges()} edges")
    print(f"  → Unified {len(cross_kg_mapping)} PatKG nodes onto MatKG nodes (threshold={merge_threshold})")
    
    final_nodes = material_informed_subgraph.number_of_nodes()
    final_edges = material_informed_subgraph.number_of_edges()
    required_properties = properties_W.get("required", [])

    if final_nodes > 0:
        print(f"\n  Subgraph Statistics:")
        print(f"    - Total nodes: {final_nodes}")
        print(f"    - Total edges: {final_edges}")
        print(f"    - Average degree: {2*final_edges/max(final_nodes,1):.2f}")
        
        # Check for material and property nodes
        material_node_count = 0
        property_node_count = 0
        for node in material_informed_subgraph.nodes():
            node_str = str(node).lower()
            if any(mat.lower() in node_str for mat in [material_X] + list(material_grounding_map.keys())[:10]):
                material_node_count += 1
            if any(prop.lower() in node_str for prop in required_properties):
                property_node_count += 1
        print(f"    - Estimated material nodes: {material_node_count}")
        print(f"    - Estimated property nodes: {property_node_count}")
    
    # Save merged material-informed subgraph to persistent storage
    subgraph_storage_path = None
    if run_id:
        subgraph_storage = SubgraphStorage()
        subgraph_storage_path = subgraph_storage.save_subgraph(
            subgraph=material_informed_subgraph,
            run_id=run_id,
            subgraph_type="material_informed"
        )
        if subgraph_storage_path:
            print(f"  ✓ Subgraph saved to persistent storage: {subgraph_storage_path}")
    
    # Extract material classes from material-informed subgraph using ResearchScientist
    print("\n" + "-" * 70)
    print("Phase 1.3: Extract Material Classes from Material-Informed Subgraph")
    print("-" * 70)
    print("→ Extracting material classes from material-informed subgraph...")
    try:
        kg_material_mapping = scientist.map_properties_to_materials(
            properties_W=properties_W,
            application_Y=application_Y,
            subgraph=material_informed_subgraph  # Query subgraph instead of full KGs
        )
        num_material_classes = len(kg_material_mapping.get('material_classes', []))
        num_paths = kg_material_mapping.get('kg_insights', {}).get('num_paths_found', 0)
        print(f"  ✓ Found {num_material_classes} material classes from KG")
        print(f"  ✓ Found {num_paths} paths connecting properties to materials")
        if num_material_classes > 0:
            print(f"  → Sample material classes:")
            for i, mc in enumerate(kg_material_mapping.get('material_classes', [])[:5], 1):
                node_id = mc.get('node_id', 'Unknown')
                node_data = mc.get('node_data', {})
                material_name = node_data.get('material_name') or node_data.get('title') or node_id
                print(f"    {i}. {material_name} (KG node: {node_id[:50]})")
    except Exception as e:
        raise RuntimeError(f"Failed to extract material classes from KG: {e}") from e
    
    # Candidates are generated iteratively in Step 2, not here
    ranked_candidates = []
    property_mapping_result = {}
    material_grounding_result = {}
    
    print("→ Material-informed subgraph built successfully")
    print("→ Candidates will be generated iteratively in Step 2")
    print("→ Returning empty ranked_candidates list")
    
    print("\n" + "=" * 70)
    print("Step 1 Complete")
    print("=" * 70)
    
    # Format result for compatibility with existing Step 2
    result = {
        "ranked_candidates": ranked_candidates,
        "material_informed_subgraph": material_informed_subgraph,
        "property_mapping": property_mapping_result,
        "material_grounding": material_grounding_result,
        "kg_material_mapping": kg_material_mapping,  # KG-derived material classes and insights
        "subgraph_storage_path": subgraph_storage_path  # Path to persisted subgraph (if saved)
    }
    
    return result


def run_material_discovery_pipeline(
    material_X: str,
    application_Y: str,
    properties_W: Dict[str, Any],
    constraints_U: List[str],
    analyst: ResearchAnalyst,
    manager: ResearchManager,
    scientist: ResearchScientist,
    tracker: RejectedCandidateTracker,
    material_db: MaterialDatabase,
    material_grounding_material: MaterialGrounding,
    material_grounding_patents: MaterialGrounding,
    knowledge_graph_material: nx.DiGraph,
    knowledge_graph_patents: nx.DiGraph,
    max_iterations: int = None,
    temperature: float = None,
    chat_logger=None,
    run_id: Optional[str] = None,
    substitution_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run the iterative, closed-loop material discovery pipeline.

    Workflow:
        1. Material Substitution (Step 1): build dual-KG material-informed
           subgraph, ground materials, extract material classes.
        2. Iterative candidate loop (Step 2, up to *max_iterations*):
           a. Propose candidate Z_i (avoiding previously rejected).
           b. Generate validation queries.
           c. Retrieve evidence via RAG.
           d. Validate feasibility.
           e. Accept or reject and continue.
        3. Return final result (success or exhaustion).

    Args:
        material_X: Material to be replaced.
        application_Y: Target application description.
        properties_W: Dict with ``required`` (list) and optional ``target_values``.
        constraints_U: List of constraint strings.
        analyst: ResearchAnalyst instance.
        manager: ResearchManager instance.
        scientist: ResearchScientist instance.
        tracker: RejectedCandidateTracker instance.
        material_db: MaterialDatabase instance.
        material_grounding_material: MaterialGrounding bound to Material Properties KG.
        material_grounding_patents: MaterialGrounding bound to Patents KG.
        knowledge_graph_material: Material Properties knowledge graph (DiGraph).
        knowledge_graph_patents: Patents knowledge graph (DiGraph).
        max_iterations: Max candidate iterations (default: config value).
        temperature: LLM temperature (default: config value).
        chat_logger: Optional ChatLogger for interaction logging.
        run_id: Pipeline run identifier for subgraph persistence.
        substitution_result: Pre-computed Step 1 result; skips Step 1 if provided.

    Returns:
        Dict with ``success``, ``candidate``, ``iterations``,
        ``rejected_candidates``, ``final_constraints``, ``evidence_summary``,
        ``property_mapping``, ``iteration_history``, and ``substitution_result``.
    """
    # Load config
    config = load_config()
    pipeline_config = config.get("pipelines", {}).get("material_discovery", {})
    
    if max_iterations is None:
        max_iterations = _req(pipeline_config, "max_iterations", "pipelines.material_discovery")
    if temperature is None:
        temperature = _req(pipeline_config, "temperature", "pipelines.material_discovery")
    
    # Extract run_id from chat_logger if not provided
    if run_id is None and chat_logger is not None:
        run_id = getattr(chat_logger, 'run_id', None)
    
    # --- Clean incoming properties to remove LLM formatting artifacts ---
    raw_required = properties_W.get("required", [])
    cleaned_required = _clean_extracted_keywords(raw_required)
    if len(cleaned_required) != len(raw_required):
        print(f"  → Cleaned {len(raw_required) - len(cleaned_required)} formatting artifacts from properties_W")
        properties_W = dict(properties_W)  # shallow copy
        properties_W["required"] = cleaned_required

    print("=" * 70)
    print("Material Discovery Pipeline: Starting Process")
    print("=" * 70)
    
    # Add hard constraint: No PFAS materials allowed
    # Per- and polyfluoroalkyl substances (PFAS) are a large class of thousands of synthetic chemicals 
    # that all contain carbon-fluorine bonds.
    PFAS_CONSTRAINT = "Cannot propose PFAS (per- and polyfluoroalkyl substances) materials - materials containing carbon-fluorine bonds are not allowed"
    
    # Create a copy of constraints_U and add the PFAS constraint if not already present
    constraints_U_final = list(constraints_U) if constraints_U else []
    if PFAS_CONSTRAINT not in constraints_U_final:
        constraints_U_final.insert(0, PFAS_CONSTRAINT)  # Add at the beginning for visibility
    
    print(f"\nPipeline Configuration:")
    print(f"  [OK] Agents verified")
    print(f"  [OK] Material to replace (X): {material_X}")
    print(f"  [OK] Application (Y): {application_Y}")
    print(f"  [OK] Required Properties: {properties_W.get('required', [])}")
    if properties_W.get('target_values'):
        print(f"  [OK] Target Values:")
        for prop, value in properties_W.get('target_values', {}).items():
            print(f"      - {prop}: {value}")
    print(f"  [OK] Constraints: {len(constraints_U_final)} constraint(s)")
    if constraints_U_final:
        for constraint in constraints_U_final:
            print(f"      - {constraint}")
    print(f"  [OK] Max iterations: {max_iterations}")
    print(f"  [OK] Temperature: {temperature}")

    # No Phase 0: keyword-connection graphs are not part of the canonical pipeline.
    
    # Initialize variables for Step 2
    ranked_candidates = []
    property_mapping_final = {}  # Will be populated with Step 1 results
    
    # Step 1: Material Substitution (pre-computed or fresh run)
    try:
        if substitution_result is not None:
            print("\n" + "=" * 70)
            print("Step 1: Material Substitution (PRE-COMPUTED)")
            print("=" * 70)
            print("✓ Using pre-computed Step 1 results from previous iteration")
            print("✓ Skipping expensive operations (material grounding, KG extraction, etc.)...")
        else:
            print("\n" + "=" * 70)
            print("Step 1: Material Substitution")
            print("=" * 70)
            print("→ Cache miss - running Step 1 operations...")
            
            substitution_result = run_material_substitution_step(
                material_X=material_X,
                application_Y=application_Y,
                properties_W=properties_W,
                constraints_U=constraints_U_final,
                material_db=material_db,
                knowledge_graph_material=knowledge_graph_material,
                knowledge_graph_patents=knowledge_graph_patents,
                material_grounding_material=material_grounding_material,
                material_grounding_patents=material_grounding_patents,
                scientist=scientist,
                temperature=temperature,
                run_id=run_id,
            )
        
        # Extract ranked candidates for Step 2 (will be empty in new flow)
        ranked_candidates = substitution_result.get("ranked_candidates", [])
        
        # Extract KG material mapping and subgraph insights
        kg_material_mapping = substitution_result.get("kg_material_mapping")
        material_informed_subgraph = substitution_result.get("material_informed_subgraph")
        
        # Extract subgraph insights for LLM context
        subgraph_insights = extract_subgraph_insights(
            material_informed_subgraph,
            material_X,
            properties_W,
            generate_fn=manager._generate_fn,
            temperature=temperature,
            chat_logger=chat_logger
        )
        
        # Convert to format compatible with Step 2 (propose_candidate expects specific format)
        # Keep original structure for final result, create converted version for Step 2
        property_mapping_for_step2 = {}
        
        # Convert property_mappings from dict format to list format expected by propose_candidate
        # Old format expected: [{"property": "name", "target_value": "value"}, ...]
        # New format: {"target_prop": {"db_prop": value}, ...}
        properties_W_required = properties_W.get("required", [])
        properties_W_target_values = properties_W.get("target_values", {})
        
        # Convert to list format - build directly from properties_W input
        property_mappings_list = []
        for target_prop in properties_W_required:
            target_val = properties_W_target_values.get(target_prop)
            property_mappings_list.append({
                "property": target_prop,
                "target_value": target_val
            })
        property_mapping_for_step2["property_mappings"] = property_mappings_list
        
        # Populate material_classes from KG mapping (instead of empty list)
        if kg_material_mapping:
            material_classes_for_step2 = kg_material_mapping.get("material_classes", [])
            property_mapping_for_step2["material_classes"] = material_classes_for_step2
            kg_insights = kg_material_mapping.get("kg_insights", {})

            # Replace pre-baked all-pairs paths with scored, diverse, property→material paths
            # computed fresh from the material_informed_subgraph.
            if material_informed_subgraph is None:
                raise RuntimeError(
                    "material_informed_subgraph is None — cannot select KG paths for proposal. "
                    "Ensure Step 1 (run_material_substitution_step) completed successfully and "
                    "returned a valid subgraph."
                )
            print("\n→ Selecting KG paths for proposal (scored + diversity-filtered)...")
            smart_paths = scientist.select_paths_for_proposal(
                subgraph=material_informed_subgraph,
                properties_W=properties_W,
                application_Y=application_Y,
                material_classes=material_classes_for_step2,
            )
            kg_insights["found_paths"] = smart_paths
            print(f"  ✓ {len(smart_paths)} paths selected for propose_candidate prompt")

            property_mapping_for_step2["kg_insights"] = kg_insights
            property_mapping_for_step2["subgraph_insights"] = subgraph_insights
            print(f"\n✓ Populated property_mapping with {len(material_classes_for_step2)} material classes from KG")
        else:
            property_mapping_for_step2["material_classes"] = []
            property_mapping_for_step2["kg_insights"] = {}
            property_mapping_for_step2["subgraph_insights"] = subgraph_insights
            print(f"\n⚠ No KG material mapping available, material_classes will be empty")
        
        # Use converted version for Step 2 and also store as final result
        property_mapping = property_mapping_for_step2
        property_mapping_final = property_mapping_for_step2
            
    except Exception as e:
        print(f"[ERROR] Error in Step 1: {str(e)}")
        result = {
            "success": False,
            "candidate": None,
            "iterations": 0,
            "rejected_candidates": [],
            "final_constraints": [f"Step 1 failed: {str(e)}"],
            "evidence_summary": {},
            "property_mapping": None,
            "iteration_history": [],
            "substitution_result": None
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
    
    # Step 2: Iterative candidate proposal and validation
    print("\n" + "-" * 70)
    print("Step 2: Iterative Candidate Proposal and Validation")
    print("-" * 70)
    
    iteration_history = []
    rejected_candidates_list = tracker.get_all_rejected()
    
    for iteration in range(1, max_iterations + 1):
        print(f"\n--- Iteration {iteration}/{max_iterations} ---")
        
        # 2a. Generate rejection lessons summary (if there are rejected candidates)
        rejection_lessons = None
        if rejected_candidates_list:
            print(f"\n  [2a.1] ResearchManager - Summarizing lessons from {len(rejected_candidates_list)} rejected candidates...")
            try:
                # Get full rejection details from tracker
                rejection_details = tracker.get_all_rejection_details()
                if rejection_details:
                    rejection_lessons = manager.summarize_rejection_lessons(
                        rejection_details=rejection_details,
                        temperature=temperature,
                    )
                    if rejection_lessons:
                        print(f"      [OK] Generated rejection lessons summary ({len(rejection_lessons)} chars)")
                        # Show preview of summary
                        preview = rejection_lessons[:200] + "..." if len(rejection_lessons) > 200 else rejection_lessons
                        print(f"      Preview: {preview}")
                    else:
                        print(f"      [WARNING] Rejection summary is empty")
                else:
                    print(f"      [WARNING] No rejection details available from tracker")
            except Exception as e:
                print(f"      [WARNING] Failed to generate rejection summary: {str(e)}")
                print(f"      → Continuing without rejection lessons")
                rejection_lessons = None
        
        # 2a. Propose candidate
        print(f"\n  [2a] ResearchManager - Proposing candidate (avoiding {len(rejected_candidates_list)} rejected)...")
        if rejected_candidates_list:
            print(f"      Previously rejected: {', '.join(rejected_candidates_list)}")
        if rejection_lessons:
            print(f"      → Using lessons learned from previous rejections to guide proposal")
        try:
            candidate_Z = manager.propose_candidate(
                property_mapping=property_mapping,
                application_Y=application_Y,
                rejected_candidates=rejected_candidates_list,
                rejection_lessons=rejection_lessons,
                material_db=material_db,
                temperature=temperature,
            )
            
            material_name = candidate_Z.get("material_name", "Unknown")
            material_class = candidate_Z.get("material_class", "Unknown")
            material_id = candidate_Z.get("material_id", "N/A")
            
            # Validate material name using shared cleaner
            material_name_clean = clean_material_name(material_name)
            is_valid_name = bool(material_name_clean)
            
            if not is_valid_name:
                print(f"      [WARNING] Material name '{material_name}' may be invalid or improperly parsed")
                print(f"         Cleaned name: '{material_name_clean}'")
            
            # Check if proposed material is PFAS using LLM evaluation (must be rejected immediately)
            # --- PFAS heuristic keywords for a quick pre-check ---
            _PFAS_POSITIVE_KEYWORDS = [
                "ptfe", "teflon", "polytetrafluoroethylene", "pfa ", "pfas",
                "fep", "etfe", "ectfe", "pctfe", "pvdf", "polyvinylidene fluoride",
                "fluoropolymer", "fluoroelastomer", "fluorocarbon", "perfluoro",
                "polyfluoro", "vinylidene fluoride", "fluorinated ethylene",
                "fluorosilicone", "fkm", "ffkm", "aflas",
            ]
            _PFAS_NEGATIVE_KEYWORDS = [
                "pfas-free", "pfas free", "non-pfas", "non pfas",
                "without pfas", "alternative to pfas", "pfas replacement",
                "replace ptfe", "replace teflon", "substitute for ptfe",
            ]

            def _heuristic_pfas_check(name: str, justification: str = "") -> bool:
                """Quick keyword-based PFAS pre-check (second opinion)."""
                combined = f"{name} {justification}".lower()
                # Negative keywords take priority — they indicate an *alternative* to PFAS
                if any(neg in combined for neg in _PFAS_NEGATIVE_KEYWORDS):
                    return False
                return any(pos in combined for pos in _PFAS_POSITIVE_KEYWORDS)

            def _llm_pfas_check(name: str, justification: str = "") -> Optional[bool]:
                """LLM-based PFAS classification. Returns True/False or None on unclear."""
                if not name or not name.strip():
                    return False
                prompts_local = load_prompts()
                system_prompt = prompts_local.get("pipelines", {}).get("material_discovery", {}).get("check_pfas")
                if system_prompt is None:
                    raise ValueError(
                        "Missing required prompt in config/prompts.yaml: pipelines.material_discovery.check_pfas. "
                        "All system prompts must be defined in the config file."
                    )
                user_tmpl = prompts_local.get("pipelines", {}).get("material_discovery", {}).get("check_pfas_user_prompt")
                if user_tmpl is None:
                    raise ValueError(
                        "Missing required prompt in config/prompts.yaml: pipelines.material_discovery.check_pfas_user_prompt. "
                        "All system prompts must be defined in the config file."
                    )
                prompt = user_tmpl.format(
                    name=name,
                    justification=justification if justification else "No description provided."
                )
                try:
                    response = manager._generate_fn(
                        system_prompt=system_prompt,
                        prompt=prompt,
                        temperature=temperature,
                        method_name="check_pfas"
                    )
                    resp_upper = response.strip().upper()
                    if resp_upper.startswith("YES"):
                        return True
                    elif resp_upper.startswith("NO"):
                        return False
                    else:
                        print(f"      [WARNING] LLM PFAS check returned unclear response: '{response[:50]}...'")
                        return None
                except Exception as e:
                    print(f"      [WARNING] LLM PFAS check failed: {e}")
                    return None

            def is_pfas_material(name: str, justification: str = "") -> bool:
                """Determine if a material is PFAS using heuristic + LLM with tie-breaking.

                When the heuristic and LLM agree, their answer is used directly.
                When they disagree, the LLM is called a second time to resolve the
                conflict, reducing both false positives and false negatives.
                """
                heuristic = _heuristic_pfas_check(name, justification)
                llm_result = _llm_pfas_check(name, justification)

                # If LLM was unclear, fall back to heuristic
                if llm_result is None:
                    print(f"      [PFAS] Heuristic={heuristic}, LLM=unclear → using heuristic")
                    return heuristic

                # Agreement → high confidence
                if heuristic == llm_result:
                    print(f"      [PFAS] Heuristic={heuristic}, LLM={llm_result} → agree")
                    return llm_result

                # Disagreement → re-query LLM with extra emphasis
                print(f"      [PFAS] Heuristic={heuristic}, LLM={llm_result} → DISAGREEMENT, re-querying LLM")
                llm_result_2 = _llm_pfas_check(
                    name,
                    justification=(
                        f"{justification}\n\n"
                        f"NOTE: A keyword-based check flagged this material as "
                        f"{'likely PFAS' if heuristic else 'likely NOT PFAS'}. "
                        f"Please re-evaluate carefully whether '{name}' contains carbon-fluorine bonds."
                    ),
                )
                if llm_result_2 is not None:
                    print(f"      [PFAS] Second LLM opinion: {llm_result_2}")
                    return llm_result_2

                # If second call also unclear, be conservative: flag as PFAS if heuristic said yes
                print(f"      [PFAS] Second LLM unclear, defaulting to heuristic={heuristic}")
                return heuristic
            
            justification = candidate_Z.get("justification", "")
            if is_pfas_material(material_name_clean, justification):
                print(f"\n      [REJECTED] Proposed material '{material_name_clean}' is a PFAS material (contains carbon-fluorine bonds)")
                print(f"      → PFAS materials are not allowed per hard constraint")
                print(f"      → Adding to rejected list and continuing to next iteration")
                
                # Record rejection
                tracker.add_rejected(
                    candidate=material_name_clean,
                    constraints=[PFAS_CONSTRAINT],
                    reason="PFAS material detected - contains carbon-fluorine bonds"
                )
                rejected_candidates_list.append(material_name_clean)
                
                # Record iteration
                iteration_history.append({
                    "iteration": iteration,
                    "candidate": material_name_clean,
                    "feasible": False,
                    "constraints_violated": [PFAS_CONSTRAINT],
                    "reasoning": "PFAS material detected - contains carbon-fluorine bonds",
                    "num_queries": 0,
                    "num_evidence_docs": 0
                })
                continue  # Skip to next iteration
            
            print(f"      [OK] Proposed candidate:")
            print(f"         Name: {material_name_clean if is_valid_name else material_name}")
            print(f"         Class: {material_class}")
            print(f"         ID: {material_id}")
            if candidate_Z.get("properties"):
                print(f"         Properties: {candidate_Z.get('properties')}")
            
            # Update candidate_Z with cleaned name if we fixed it
            if material_name_clean != material_name:
                candidate_Z["material_name"] = material_name_clean
                material_name = material_name_clean
        except Exception as e:
            print(f"      [ERROR] Error proposing candidate: {str(e)}")
            iteration_history.append({
                "iteration": iteration,
                "candidate": None,
                "error": str(e),
                "feasible": False
            })
            continue
        
        # Check if this candidate was already rejected (shouldn't happen, but safety check)
        if tracker.is_rejected(material_name):
            print(f"      [WARNING] Proposed candidate {material_name} was already rejected, skipping")
            iteration_history.append({
                "iteration": iteration,
                "candidate": material_name,
                "feasible": False,
                "reason": "Already rejected"
            })
            continue
        
        # 2b. Generate validation queries
        print(f"\n  [2b] ResearchManager - Generating validation queries...")
        try:
            # Extract KG context for candidate
            kg_context = None
            subgraph_for_paths = None
            if substitution_result and substitution_result.get('material_informed_subgraph'):
                # Find candidate in subgraph
                candidate_name = candidate_Z.get('material_name', '')
                subgraph_for_paths = substitution_result['material_informed_subgraph']
                material_nodes = [n for n in subgraph_for_paths.nodes() if candidate_name.lower() in str(n).lower()]
                if material_nodes:
                    kg_context = {'material_nodes': material_nodes[:5]}
                    print(f"      → Found {len(material_nodes)} KG nodes for candidate")
            
            validation_queries = manager.generate_validation_queries(
                candidate_Z=candidate_Z,
                properties_W=properties_W,
                constraints_U=constraints_U_final,  # Use constraints with PFAS constraint included
                temperature=temperature,
                kg_context=kg_context,  # Pass KG context
                subgraph=subgraph_for_paths,  # Pass subgraph for path finding
                node_embeddings=scientist.node_embeddings if scientist else None,
                embedding_model=scientist.embedding_model if scientist else None,
                embedding_tokenizer=scientist.embedding_tokenizer if scientist else None,
            )
            print(f"      [OK] Generated {len(validation_queries)} validation queries:")
            for i, query in enumerate(validation_queries, 1):
                print(f"         {i}. {query}")
        except Exception as e:
            print(f"      [ERROR] Error generating queries: {str(e)}")
            iteration_history.append({
                "iteration": iteration,
                "candidate": material_name,
                "error": f"Query generation failed: {str(e)}",
                "feasible": False
            })
            continue
        
        # 2c. Retrieve evidence via RAG
        print(f"\n  [2c] ResearchAnalyst - Retrieving evidence for {len(validation_queries)} queries...")
        evidence_I = []
        
        for i, query in enumerate(validation_queries, 1):
            try:
                print(f"      [{i}/{len(validation_queries)}] Processing query: \"{query}\"")
                rag_result = analyst.analyze_question(question=query)
                rag_results = rag_result.get("rag_results", [])
                
                print(f"         → Retrieved {len(rag_results)} documents from RAG")
                if rag_results:
                    # Show snippet of first result
                    first_result = rag_results[0] if rag_results else {}
                    content_preview = str(first_result.get("content", ""))[:150] if first_result.get("content") else "N/A"
                    print(f"         → Top result preview: {content_preview}...")
                
                # Answer the query using ResearchManager
                print(f"         → Generating answer using ResearchManager...")
                answer = manager.answer_question(
                    question=query,
                    rag_results=rag_results,
                    temperature=temperature,
                )
                
                answer_preview = str(answer)[:200] if answer else "N/A"
                print(f"         → Answer: {answer_preview}...")
                
                evidence_I.append({
                    "query": query,
                    "rag_results": rag_results,
                    "answer": answer,
                    "num_documents": len(rag_results)
                })
                
                print(f"         [OK] Query {i} complete: {len(rag_results)} docs, answer generated")
            except Exception as e:
                print(f"      [{i}/{len(validation_queries)}] [ERROR] Error: {str(e)}")
                evidence_I.append({
                    "query": query,
                    "rag_results": [],
                    "answer": f"Error: {str(e)}",
                    "num_documents": 0
                })
        
        # 2d. Validate feasibility
        print(f"\n  [2d] ResearchManager - Validating feasibility...")
        print(f"      → Analyzing candidate '{material_name}' against requirements...")
        print(f"      → Reviewing {len(evidence_I)} evidence items...")
        try:
            # Extract KG evidence for candidate
            kg_evidence = None
            subgraph_for_feasibility = None
            if substitution_result and substitution_result.get('material_informed_subgraph'):
                subgraph_for_feasibility = substitution_result['material_informed_subgraph']
                # Find material nodes for candidate
                material_nodes = [n for n in subgraph_for_feasibility.nodes() if material_name.lower() in str(n).lower()]
                if material_nodes:
                    kg_evidence = {
                        'nodes_connected': material_nodes[:5]
                    }
            
            feasibility_result = manager.validate_feasibility(
                candidate_Z=candidate_Z,
                evidence_I=evidence_I,
                properties_W=properties_W,
                constraints_U=constraints_U_final,  # Use constraints with PFAS constraint included
                temperature=temperature,
                kg_evidence=kg_evidence,  # Pass KG evidence
                subgraph=subgraph_for_feasibility,  # Pass subgraph for path finding
                node_embeddings=scientist.node_embeddings if scientist else None,
                embedding_model=scientist.embedding_model if scientist else None,
                embedding_tokenizer=scientist.embedding_tokenizer if scientist else None,
            )

            is_feasible = feasibility_result.get("is_feasible", False)
            constraints_violated = feasibility_result.get("constraints_violated", [])
            reasoning = feasibility_result.get("reasoning", "")
            
            print(f"\n      Feasibility Assessment:")
            print(f"         Feasible: {'[YES]' if is_feasible else '[NO]'}")
            if reasoning:
                reasoning_preview = reasoning[:300] if len(reasoning) > 300 else reasoning
                print(f"         Reasoning: {reasoning_preview}")
                if len(reasoning) > 300:
                    print(f"         ... (truncated, full reasoning saved in results)")
            if constraints_violated:
                print(f"         Constraints violated: {len(constraints_violated)}")
                for constraint in constraints_violated:
                    print(f"           [VIOLATED] {constraint}")
            else:
                print(f"         [OK] No constraints violated")
            
            # Record iteration
            iteration_record = {
                "iteration": iteration,
                "candidate": material_name,
                "feasible": is_feasible,
                "constraints_violated": constraints_violated,
                "reasoning": reasoning,  # Full reasoning preserved for downstream analysis
                "num_queries": len(validation_queries),
                "num_evidence_docs": sum(ev.get("num_documents", 0) for ev in evidence_I)
            }
            iteration_history.append(iteration_record)
            
            # If feasible, return success
            if is_feasible:
                print("\n" + "=" * 70)
                print("SUCCESS: Viable candidate material found!")
                print("=" * 70)
                print(f"\nFinal Candidate:")
                print(f"   Name: {material_name}")
                print(f"   Class: {candidate_Z.get('material_class', 'Unknown')}")
                print(f"   ID: {candidate_Z.get('material_id', 'N/A')}")
                print(f"   Iterations required: {iteration}/{max_iterations}")
                print(f"\nEvidence Summary:")
                print(f"   Total queries: {len(validation_queries)}")
                print(f"   Total documents retrieved: {sum(ev.get('num_documents', 0) for ev in evidence_I)}")
                print(f"   Queries with evidence: {sum(1 for ev in evidence_I if ev.get('num_documents', 0) > 0)}")
                if reasoning:
                    print(f"\nReasoning:")
                    print(f"   {reasoning[:500]}...")
                
                # Build evidence summary
                evidence_summary = {
                    "total_queries": len(validation_queries),
                    "total_documents": sum(ev.get("num_documents", 0) for ev in evidence_I),
                    "queries_with_evidence": sum(1 for ev in evidence_I if ev.get("num_documents", 0) > 0)
                }
                
                result = {
                    "success": True,
                    "candidate": candidate_Z,
                    "iterations": iteration,
                    "rejected_candidates": rejected_candidates_list,
                    "final_constraints": [],
                    "evidence_summary": evidence_summary,
                    "property_mapping": property_mapping_final,
                    "iteration_history": iteration_history,
                    "substitution_result": substitution_result
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
            
            # If not feasible, record rejection and continue
            print(f"\n      [WARNING] Candidate rejected, recording constraints and continuing...")
            print(f"      → Adding '{material_name}' to rejected list")
            print(f"      → Recording {len(constraints_violated)} violated constraints")
            tracker.add_rejected(
                candidate=material_name,
                constraints=constraints_violated,
                reason=reasoning[:200]
            )
            rejected_candidates_list.append(material_name)
            print(f"      → Total rejected so far: {len(rejected_candidates_list)}")
            
        except Exception as e:
            print(f"      [ERROR] Error validating feasibility: {str(e)}")
            iteration_history.append({
                "iteration": iteration,
                "candidate": material_name,
                "error": f"Feasibility validation failed: {str(e)}",
                "feasible": False
            })
            # Still record as rejected to avoid retrying
            tracker.add_rejected(
                candidate=material_name,
                constraints=[],
                reason=f"Validation error: {str(e)}"
            )
            rejected_candidates_list.append(material_name)
            continue
    
    # If we reach here, all iterations exhausted without finding viable candidate
    print("\n" + "=" * 70)
    print(f"[EXHAUSTED] No viable candidate found after {max_iterations} iterations")
    print("=" * 70)
    print(f"\nFinal Summary:")
    print(f"   Total iterations: {max_iterations}")
    print(f"   Rejected candidates: {len(rejected_candidates_list)}")
    if rejected_candidates_list:
        print(f"   Rejected list: {', '.join(rejected_candidates_list)}")
    if iteration_history:
        print(f"\nIteration Details:")
        for hist in iteration_history:
            iter_num = hist.get('iteration', 0)
            candidate = hist.get('candidate', 'Unknown')
            feasible = hist.get('feasible', False)
            status = "[FEASIBLE]" if feasible else "[REJECTED]"
            print(f"   Iteration {iter_num}: {candidate} - {status}")
            if not feasible and hist.get('constraints_violated'):
                print(f"      Constraints violated: {', '.join(hist.get('constraints_violated', [])[:3])}")
    
    # Collect final constraints from last iteration
    final_constraints = []
    if iteration_history:
        last_iteration = iteration_history[-1]
        final_constraints = last_iteration.get("constraints_violated", [])
    
    # Build evidence summary from all iterations
    evidence_summary = {
        "total_iterations": len(iteration_history),
        "total_candidates_tested": len([h for h in iteration_history if h.get("candidate")])
    }
    
    return {
        "success": False,
        "candidate": None,
        "iterations": max_iterations,
        "rejected_candidates": rejected_candidates_list,
        "final_constraints": final_constraints,
        "evidence_summary": evidence_summary,
        "property_mapping": property_mapping_final,
        "iteration_history": iteration_history,
        "substitution_result": substitution_result
    }


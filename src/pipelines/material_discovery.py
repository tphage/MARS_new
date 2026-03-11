"""Material Discovery Pipeline Function"""

import logging
from typing import List, Dict, Any, Optional, Callable
import re
import json
import networkx as nx

logger = logging.getLogger(__name__)
from ..agents import ResearchAnalyst, ResearchManager, ResearchScientist, RejectedCandidateTracker
from ..agents.material_scientist import MaterialScientist
from ..utils.material_database import MaterialDatabase
from ..utils.subgraph_processor import SubgraphProcessor
from ..utils.material_grounding import MaterialGrounding
from ..utils.step1_cache import Step1Cache
from ..utils.subgraph_storage import SubgraphStorage
from ..config import load_prompts, load_config
from .material_requirements import _clean_extracted_keywords
from ..utils.parsing import clean_material_name


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
        batch_size = pipeline_config.get("batch_size", 30)
    if temperature is None:
        temperature = pipeline_config.get("temperature", 0)
    
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
    
    max_nodes_for_context = pipeline_config.get("batch_nodes_for_llm_context", 10)
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
    subgraph_data: Optional[Dict[str, Any]],
    material_db: MaterialDatabase,
    subgraph_processor: SubgraphProcessor,
    material_grounding: MaterialGrounding,
    material_scientist: MaterialScientist,
    knowledge_graph: nx.DiGraph,
    scientist: ResearchScientist,
    temperature: Optional[float] = None,
    run_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run the material substitution step.
    
    Args:
        material_X: Material to be replaced
        application_Y: Target application
        properties_W: Required material properties
        constraints_U: List of constraints
        subgraph_data: Subgraph data from System 1 (optional)
        material_db: MaterialDatabase instance
        subgraph_processor: SubgraphProcessor instance
        material_grounding: MaterialGrounding instance
        material_scientist: MaterialScientist instance
        knowledge_graph: Global knowledge graph
        scientist: ResearchScientist instance for extracting material classes from KG
        temperature: Temperature for LLM generation (default: None, uses config value)
        
    Returns:
        Dictionary with substitution results (ranked_candidates will be empty as candidates
        are generated iteratively in Step 2), and kg_material_mapping (KG-derived material classes and insights)
    """
    # Load config
    config = load_config()
    pipeline_config = config.get("pipelines", {}).get("material_discovery", {})
    
    # Use config default if not provided
    if temperature is None:
        temperature = pipeline_config.get("temperature", 0)
    
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
    print(f"  Knowledge Graph: {knowledge_graph.number_of_nodes()} nodes, {knowledge_graph.number_of_edges()} edges")
    print(f"  Material Database: {len(material_db)} materials")
    
    # Load and filter subgraph
    print("\n" + "-" * 70)
    print("Phase 1.1: Load and Filter Subgraph from System 1")
    print("-" * 70)
    filtered_subgraph = None
    if subgraph_data:
        print("→ Loading subgraph from System 1...")
        print(f"  → Subgraph data keys: {list(subgraph_data.keys())}")
        if subgraph_data.get('matched_node_ids'):
            print(f"  → Matched node IDs in data: {len(subgraph_data.get('matched_node_ids', []))}")
        if subgraph_data.get('found_paths'):
            print(f"  → Found paths in data: {len(subgraph_data.get('found_paths', []))}")
        
        filtered_subgraph = subgraph_processor.load_subgraph(subgraph_data)
        if filtered_subgraph:
            initial_nodes = filtered_subgraph.number_of_nodes()
            initial_edges = filtered_subgraph.number_of_edges()
            print(f"  [OK] Loaded subgraph: {initial_nodes} nodes, {initial_edges} edges")
            
            # Filter by relevance
            print("\n→ Filtering subgraph by relevance...")
            print(f"  → Application context: {application_Y}")
            print(f"  → Required properties: {properties_W.get('required', [])}")
            print(f"  → Constraints: {constraints_U}")
            
            filtered_subgraph = subgraph_processor.filter_by_relevance(
                filtered_subgraph,
                application_Y,
                properties_W,
                constraints_U
            )
            final_nodes = filtered_subgraph.number_of_nodes()
            final_edges = filtered_subgraph.number_of_edges()
            print(f"  [OK] Filtered subgraph: {final_nodes} nodes, {final_edges} edges")
            print(f"  → Reduction: {initial_nodes - final_nodes} nodes removed ({100*(initial_nodes-final_nodes)/max(initial_nodes,1):.1f}%)")
            print(f"  → Reduction: {initial_edges - final_edges} edges removed ({100*(initial_edges-final_edges)/max(initial_edges,1):.1f}%)")
        else:
            print("  [WARNING] Could not load subgraph from data, will use empty subgraph")
            filtered_subgraph = nx.DiGraph()
    else:
        print("  [WARNING] No subgraph data provided, will use empty subgraph")
        filtered_subgraph = nx.DiGraph()
    
    # Ground lab materials in KG
    print("\n" + "-" * 70)
    print("Phase 1.2: Ground Lab Materials in Knowledge Graph")
    print("-" * 70)
    print("→ Grounding lab materials in knowledge graph...")
    print(f"  → Material database contains {len(material_db)} materials")
    
    # Show sample materials from database
    if len(material_db) > 0:
        sample_db_materials = []
        # MaterialDatabase stores materials in _material_index dict (material_id -> material_dict)
        material_index = material_db._material_index
        for i, (mat_id, mat_data) in enumerate(list(material_index.items())[:5]):
            mat_name = mat_data.get('material_name', mat_id)
            sample_db_materials.append(f"{mat_name} ({mat_id})")
        print(f"  → Sample materials in database: {', '.join(sample_db_materials)}")
    
    material_grounding_map = material_grounding.ground_material_database(material_db)
    print(f"  [OK] Grounded {len(material_grounding_map)} materials")
    
    if material_grounding_map:
        sample_materials = list(material_grounding_map.keys())[:5]
        print(f"  → Sample grounded materials: {', '.join(sample_materials)}")
        
        # Show grounding details for sample materials
        min_matches = load_config().get("pipelines", {}).get("material_discovery", {}).get("min_matches_for_multiple_grounding", 2)
        print(f"\n  Grounding Details (sample):")
        for mat_id in sample_materials[:3]:
            matches = material_grounding_map[mat_id]
            print(f"    - {mat_id}:")
            print(f"      → Found {len(matches)} matching nodes in KG")
            for i, match in enumerate(matches[:min_matches], 1):
                node_id = match.get('node_id', 'N/A')
                similarity = match.get('similarity', 0.0)
                print(f"        {i}. Node: {node_id} (similarity: {similarity:.3f})")
            if len(matches) > min_matches:
                print(f"        ... and {len(matches) - min_matches} more matches")
    else:
        print("  [WARNING] No materials were successfully grounded in the knowledge graph")
    
    # Retrieve material relationships
    print("\n" + "-" * 70)
    print("Phase 1.3: Retrieve Material-Property Relationships")
    print("-" * 70)
    print("→ Retrieving material-property relationships...")
    all_material_node_ids = []
    for material_id, matches in material_grounding_map.items():
        all_material_node_ids.extend([m["node_id"] for m in matches])
    print(f"  → Collected {len(all_material_node_ids)} material node IDs from grounding")
    print(f"  → Unique material node IDs: {len(set(all_material_node_ids))}")
    
    required_properties = properties_W.get("required", [])
    print(f"\n  → Searching for relationships with {len(required_properties)} required properties:")
    for i, prop in enumerate(required_properties, 1):
        print(f"    {i}. {prop}")
    
    material_relationships = material_grounding.retrieve_material_relationships(
        all_material_node_ids,
        None  # Will search for property nodes in the subgraph
    )
    rel_nodes = material_relationships.number_of_nodes()
    rel_edges = material_relationships.number_of_edges()
    print(f"  [OK] Found {rel_nodes} nodes, {rel_edges} edges in material relationships")
    
    if rel_nodes > 0:
        # Show sample nodes and edges
        sample_nodes = list(material_relationships.nodes())[:5]
        print(f"  → Sample relationship nodes: {sample_nodes}")
        if rel_edges > 0:
            sample_edges = list(material_relationships.edges())[:3]
            print(f"  → Sample relationship edges: {sample_edges}")
    
    # Merge into material-informed subgraph
    print("\n" + "-" * 70)
    print("Phase 1.4: Merge into Material-Informed Subgraph")
    print("-" * 70)
    print("→ Merging filtered subgraph with material relationships...")
    print(f"  → Filtered subgraph: {filtered_subgraph.number_of_nodes()} nodes, {filtered_subgraph.number_of_edges()} edges")
    print(f"  → Material relationships: {rel_nodes} nodes, {rel_edges} edges")
    
    material_informed_subgraph = material_grounding.merge_into_subgraph(
        filtered_subgraph,
        material_relationships
    )
    final_nodes = material_informed_subgraph.number_of_nodes()
    final_edges = material_informed_subgraph.number_of_edges()
    print(f"  [OK] Material-informed subgraph: {final_nodes} nodes, {final_edges} edges")
    
    # Show subgraph statistics
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
    
    # Save material-informed subgraph to persistent storage
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
    print("Phase 1.6: Extract Material Classes from Material-Informed Subgraph")
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
    subgraph_processor: SubgraphProcessor,
    material_grounding: MaterialGrounding,
    material_scientist: MaterialScientist,
    knowledge_graph: nx.DiGraph,
    subgraph_data: Optional[Dict[str, Any]] = None,
    max_iterations: int = None,
    temperature: float = None,
    chat_logger=None,
    step1_cache: Optional[Step1Cache] = None,
    material_db_path: Optional[str] = None,
    run_id: Optional[str] = None,
    substitution_result: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run the iterative, closed-loop material discovery pipeline.
    
    Workflow:
    1. Material Substitution: Find ranked candidates using lab DB + subgraph + KG
    2. Loop (max_iterations):
       a. ResearchManager: Propose candidate Z_i (avoiding rejected)
       b. ResearchManager: Generate validation queries SQ_i
       c. ResearchAnalyst: Retrieve evidence I_i for each query
       d. ResearchManager: Validate feasibility
       e. If feasible → return Z_j
       f. If not feasible → record constraints U_i, add to rejected list, continue
    3. Return final result (success or exhaustion)
    
    Args:
        material_X: Material to be replaced (required)
        application_Y: Target application description
        properties_W: Dictionary with property requirements:
            - "required": List of property names
            - "target_values": Optional dict mapping properties to target values
        constraints_U: List of constraint strings
        analyst: ResearchAnalyst instance (required)
        manager: ResearchManager instance (required)
        scientist: ResearchScientist instance (required)
        tracker: RejectedCandidateTracker instance (required)
        material_db: MaterialDatabase instance (required)
        subgraph_processor: SubgraphProcessor instance (required)
        material_grounding: MaterialGrounding instance (required)
        material_scientist: MaterialScientist instance (required)
        knowledge_graph: Global knowledge graph (required)
        subgraph_data: Optional subgraph data from System 1
        max_iterations: Maximum number of iteration attempts (default: None, uses config value)
        temperature: Temperature for LLM generation (default: None, uses config value)
        step1_cache: Optional Step1Cache instance for caching Step 1 results
        material_db_path: Optional path to material database file (for cache key generation)
        substitution_result: Optional pre-computed Step 1 result. If provided, Step 1 will be skipped.
        **kwargs: Additional arguments passed to agent methods
    
    Returns:
        Dict containing:
            - "success": bool indicating if viable candidate was found
            - "candidate": Candidate material dict if success, else None
            - "iterations": Number of iterations performed
            - "rejected_candidates": List of rejected candidate names
            - "final_constraints": List of final constraints if not success
            - "evidence_summary": Summary of evidence gathered
            - "property_mapping": Result from substitution step
            - "iteration_history": List of iteration results
            - "substitution_result": Result from Step 1
    """
    # Load config
    config = load_config()
    pipeline_config = config.get("pipelines", {}).get("material_discovery", {})
    
    # Use config defaults if not provided
    if max_iterations is None:
        max_iterations = pipeline_config.get("max_iterations", 5)
    if temperature is None:
        temperature = pipeline_config.get("temperature", 0)
    
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
    
    # Initialize variables for Step 2
    ranked_candidates = []
    property_mapping_final = {}  # Will be populated with Step 1 results
    
    # Step 1: Material Substitution (with caching or pre-computed result)
    try:
        # If substitution_result is provided, skip Step 1 entirely
        if substitution_result is not None:
            print("\n" + "=" * 70)
            print("Step 1: Material Substitution (PRE-COMPUTED)")
            print("=" * 70)
            print("✓ Using pre-computed Step 1 results from previous iteration")
            print("✓ Skipping expensive operations (material grounding, KG extraction, etc.)...")
            cache_hit = True  # Treat as cache hit to skip Step 1 execution
        else:
            substitution_result = None
            cache_hit = False
        
        # Check cache first if available and no pre-computed result provided
        if not cache_hit and step1_cache is not None:
            cached_result = step1_cache.get(
                material_X=material_X,
                application_Y=application_Y,
                properties_W=properties_W,
                constraints_U=constraints_U_final,
                subgraph_data=subgraph_data,
                material_db_path=material_db_path
            )
            
            if cached_result is not None:
                substitution_result = cached_result
                cache_hit = True
                print("\n" + "=" * 70)
                print("Step 1: Material Substitution (CACHED)")
                print("=" * 70)
                print("✓ Using cached Step 1 results (material grounding, KG extraction, etc.)")
                print("✓ Skipping expensive operations...")
        
        # If cache miss and no pre-computed result, run Step 1 normally
        if not cache_hit:
            print("\n" + "=" * 70)
            print("Step 1: Material Substitution")
            print("=" * 70)
            print("→ Cache miss - running Step 1 operations...")
            
            substitution_result = run_material_substitution_step(
                material_X=material_X,
                application_Y=application_Y,
                properties_W=properties_W,
                constraints_U=constraints_U_final,  # Use constraints with PFAS constraint included
                subgraph_data=subgraph_data,
                material_db=material_db,
                subgraph_processor=subgraph_processor,
                material_grounding=material_grounding,
                material_scientist=material_scientist,
                knowledge_graph=knowledge_graph,
                scientist=scientist,  # Pass scientist for KG material class extraction
                temperature=temperature,
                run_id=run_id  # Pass run_id for subgraph persistence
            )
            
            # Store in cache for future iterations
            if step1_cache is not None:
                step1_cache.set(
                    material_X=material_X,
                    application_Y=application_Y,
                    properties_W=properties_W,
                    constraints_U=constraints_U_final,
                    substitution_result=substitution_result,
                    subgraph_data=subgraph_data,
                    material_db_path=material_db_path
                )
                print("✓ Step 1 results cached for future iterations")
        
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
                        **kwargs
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
                **kwargs
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
                **kwargs
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
                    **kwargs
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
                **kwargs
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


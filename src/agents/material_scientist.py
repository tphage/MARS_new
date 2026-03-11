"""Material Scientist - Finds substitute materials using graph reasoning and RAG"""

import logging
import networkx as nx
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)
from ..utils.material_database import MaterialDatabase
from ..agents.research_analyst import ResearchAnalyst
from ..agents.research_manager import ResearchManager
from ..config import load_prompts, load_config


class MaterialScientist:
    """
    Reasons about material substitution using:
    - Material-informed subgraph
    - Lab material database
    - Required properties
    - RAG evidence
    """
    
    def __init__(
        self,
        material_db: MaterialDatabase,
        knowledge_graph: nx.DiGraph,
        analyst: ResearchAnalyst,
        manager: ResearchManager,
        embedding_model,
        embedding_tokenizer
    ):
        """
        Initialize the material scientist.
        
        Args:
            material_db: MaterialDatabase instance with lab materials
            knowledge_graph: Global knowledge graph
            analyst: ResearchAnalyst instance for RAG queries
            manager: ResearchManager instance for LLM reasoning
            embedding_model: Embedding model for semantic matching
            embedding_tokenizer: Embedding tokenizer
        """
        self.material_db = material_db
        self.knowledge_graph = knowledge_graph
        self.analyst = analyst
        self.manager = manager
        self.embedding_model = embedding_model
        self.embedding_tokenizer = embedding_tokenizer
        
        # Load config
        config = load_config()
        agent_config = config.get("agents", {}).get("material_scientist", {})
        self.temperature = agent_config.get("temperature", 0)
        evidence_limits = agent_config.get("evidence_limits", {})
        self.matching_nodes_preview = evidence_limits.get("matching_nodes_preview", 3)
        self.property_nodes_preview = evidence_limits.get("property_nodes_preview", 3)
        self.matching_nodes_in_dict = evidence_limits.get("matching_nodes_in_dict", 5)
        self.property_nodes_in_dict = evidence_limits.get("property_nodes_in_dict", 5)
        self.property_matches_for_rag = evidence_limits.get("property_matches_for_rag", 5)
        scoring = agent_config.get("scoring", {})
        self.graph_normalization_divisor = scoring.get("graph_normalization_divisor", 5.0)
        self.rag_normalization_divisor = scoring.get("rag_normalization_divisor", 10.0)
        self.property_weight = scoring.get("property_weight", 0.5)
        self.graph_weight = scoring.get("graph_weight", 0.3)
        self.rag_weight = scoring.get("rag_weight", 0.2)
        self.min_property_matches_for_many = agent_config.get("min_property_matches_for_many", 3)
    
    def find_substitute(
        self,
        material_X: str,
        application_Y: str,
        properties_W: Dict[str, Any],
        constraints_U: List[str],
        material_informed_subgraph: nx.DiGraph,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Find a substitute material Z that can replace material X in application Y.
        
        Args:
            material_X: Material to be replaced
            application_Y: Target application
            properties_W: Required material properties
            constraints_U: List of constraints
            material_informed_subgraph: Filtered subgraph with material relationships
            temperature: Temperature for LLM generation (default: None, uses config value)
            
        Returns:
            Dictionary containing:
                - ranked_candidates: List of candidate materials with scores
                - material_informed_subgraph: The subgraph used
                - property_mapping: Property matching results
                - material_grounding: Material grounding information
        """
        # Use config default if not provided
        if temperature is None:
            temperature = self.temperature
        
        print(f"\n  [Step 1.5.1] Property Matching")
        print(f"  → Matching {len(properties_W.get('required', []))} required properties to lab materials...")
        
        # Step 1: Match properties to lab materials
        property_matches, property_mappings = self.match_properties(properties_W)
        
        print(f"  ✓ Found {len(property_matches)} material matches")
        if property_matches:
            print(f"  → Top {self.property_matches_for_rag} property matches:")
            for i, match in enumerate(property_matches[:self.property_matches_for_rag], 1):
                mat_name = match.get('material_name', 'Unknown')
                score = match.get('property_match_score', 0.0)
                matched_props = match.get('matched_properties', [])
                print(f"    {i}. {mat_name}: score={score:.3f}, matched {len(matched_props)}/{len(properties_W.get('required', []))} properties")
        
        # Display property mappings
        if property_mappings:
            print(f"\n  Property Mappings (Database Property → Target Property):")
            for target_prop, db_props in property_mappings.items():
                db_prop_names = list(db_props.keys())
                print(f"    - '{target_prop}' ← {', '.join(db_prop_names[:self.min_property_matches_for_many])}")
                if len(db_prop_names) > self.min_property_matches_for_many:
                    print(f"      ... and {len(db_prop_names) - self.min_property_matches_for_many} more database properties")
        
        print(f"\n  [Step 1.5.2] Graph Evidence Evaluation")
        print(f"  → Evaluating graph evidence for {len(property_matches)} candidates...")
        print(f"  → Subgraph has {material_informed_subgraph.number_of_nodes()} nodes, {material_informed_subgraph.number_of_edges()} edges")
        
        # Step 2: Evaluate graph evidence for candidates
        graph_evidence = self.evaluate_graph_evidence(
            property_matches,
            material_informed_subgraph,
            properties_W
        )
        
        print(f"  ✓ Evaluated graph evidence for {len(graph_evidence)} candidates")
        if graph_evidence:
            total_paths = sum(ev.get('paths_found', 0) for ev in graph_evidence.values())
            candidates_with_paths = sum(1 for ev in graph_evidence.values() if ev.get('paths_found', 0) > 0)
            print(f"  → Total paths found: {total_paths}")
            print(f"  → Candidates with graph paths: {candidates_with_paths}/{len(graph_evidence)}")
            if candidates_with_paths > 0:
                print(f"  → Sample candidates with graph evidence:")
                for mat_id, ev in list(graph_evidence.items())[:3]:
                    if ev.get('paths_found', 0) > 0:
                        mat_name = next((m.get('material_name', mat_id) for m in property_matches if m.get('material_id') == mat_id), mat_id)
                        print(f"    - {mat_name}: {ev.get('paths_found', 0)} paths")
        
        print(f"\n  [Step 1.5.3] RAG Evidence Retrieval")
        required_properties = properties_W.get("required", [])
        top_candidates = property_matches[:self.property_matches_for_rag]
        print(f"  → Querying RAG for top {len(top_candidates)} candidates...")
        
        rag_evidence = {}
        for idx, candidate in enumerate(top_candidates, 1):
            material_id = candidate["material_id"]
            material_name = candidate["material_name"]
            
            # Create RAG query
            query = f"Can {material_name} replace {material_X} in {application_Y}? Properties: {', '.join(required_properties)}"
            print(f"  → [{idx}/{len(top_candidates)}] Querying RAG for {material_name}...")
            print(f"     Query: \"{query}\"")
            
            rag_result = self.analyst.analyze_question(query)
            rag_results = rag_result.get("rag_results", [])
            num_docs = len(rag_results)
            
            rag_evidence[material_id] = {
                "rag_results": rag_results,
                "num_documents": num_docs
            }
            
            print(f"     ✓ Retrieved {num_docs} documents")
            if rag_results:
                # Show snippet of top result
                top_result = rag_results[0] if rag_results else {}
                content_preview = str(top_result.get("content", ""))[:150] if top_result.get("content") else "N/A"
                print(f"     → Top result preview: {content_preview}...")
        
        total_rag_docs = sum(ev.get("num_documents", 0) for ev in rag_evidence.values())
        print(f"  ✓ RAG retrieval complete: {total_rag_docs} total documents retrieved")
        
        print(f"\n  [Step 1.5.4] Candidate Ranking")
        print(f"  → Ranking {len(property_matches)} candidates using combined evidence...")
        print(f"  → Evidence sources:")
        print(f"    - Property matching: {len(property_matches)} candidates")
        print(f"    - Graph evidence: {len(graph_evidence)} candidates")
        print(f"    - RAG evidence: {len(rag_evidence)} candidates")
        
        # Step 4: Rank candidates combining all evidence
        ranked_candidates = self.rank_candidates(
            property_matches,
            graph_evidence,
            rag_evidence,
            material_X,
            application_Y,
            properties_W,
            constraints_U,
            temperature
        )
        
        print(f"  ✓ Ranked {len(ranked_candidates)} candidates")
        if ranked_candidates:
            print(f"  → Property match score distribution:")
            scores = [c.get('property_matches', {}).get('score', 0.0) for c in ranked_candidates]
            if scores:
                print(f"    - Min: {min(scores):.3f}")
                print(f"    - Max: {max(scores):.3f}")
                print(f"    - Mean: {sum(scores)/len(scores):.3f}")
                print(f"    - Median: {sorted(scores)[len(scores)//2]:.3f}")
        
        print(f"\n  [Step 1.5.5] Explanation Generation")
        # Step 5: Generate explanation for top candidate
        if ranked_candidates:
            top_candidate = ranked_candidates[0]
            top_name = top_candidate.get('material_name', 'Unknown')
            print(f"  → Generating explanation for top candidate: {top_name}...")
            
            explanation = self.generate_explanation(
                top_candidate,
                material_X,
                application_Y,
                properties_W
            )
            ranked_candidates[0]["explanation"] = explanation
            print(f"  ✓ Explanation generated ({len(explanation)} characters)")
        else:
            print(f"  ⚠ No candidates available for explanation generation")
        
        return {
            "ranked_candidates": ranked_candidates,
            "material_informed_subgraph": material_informed_subgraph,
            "property_mapping": {
                "property_matches": property_matches,
                "num_matches": len(property_matches),
                "property_mappings": property_mappings  # Maps target properties to DB properties
            },
            "material_grounding": {
                "graph_evidence": graph_evidence,
                "rag_evidence": rag_evidence
            }
        }
    
    def match_properties(self, properties_W: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, str]]]:
        """
        Match required properties to lab materials.
        
        Args:
            properties_W: Dictionary with property requirements
            
        Returns:
            Tuple of:
            - List of candidate materials with property match scores
            - Dictionary mapping target properties to database property mappings
                Format: {target_prop: {db_prop_name: db_value, ...}, ...}
        """
        required_properties = properties_W.get("required", [])
        target_values = properties_W.get("target_values", {})
        
        candidates = []
        property_mappings = {}  # Track which DB properties map to which target properties
        
        # Search materials matching properties
        properties_dict = {}
        for prop in required_properties:
            if prop in target_values:
                properties_dict[prop] = target_values[prop]
            else:
                properties_dict[prop] = None  # Any value acceptable
        
        # Try matching all properties first
        matches = self.material_db.search_by_properties(properties_dict, match_all=True)
        
        # Score each match - if material was returned, it matched all required properties
        for material in matches:
            material_props = material.get("properties", {})
            # Since search_by_properties returned this material with match_all=True,
            # it matched all required properties
            score = 1.0  # Perfect match
            
            # Track property mappings for this material
            material_mappings = {}
            property_mapper = self.material_db.property_mapper
            if property_mapper:
                for target_prop in required_properties:
                    # Find which DB property maps to this target property
                    for db_prop_name, db_value in material_props.items():
                        mapped_target = property_mapper.map_property_name(
                            db_prop_name,
                            [target_prop]
                        )
                        if mapped_target == target_prop:
                            if target_prop not in material_mappings:
                                material_mappings[target_prop] = {}
                            material_mappings[target_prop][db_prop_name] = db_value
                            # Store in global mappings
                            if target_prop not in property_mappings:
                                property_mappings[target_prop] = {}
                            if db_prop_name not in property_mappings[target_prop]:
                                property_mappings[target_prop][db_prop_name] = db_value
            
            candidates.append({
                "material_id": material["material_id"],
                "material_name": material["material_name"],
                "properties": material_props,
                "property_match_score": score,
                "matched_properties": required_properties.copy(),  # All properties matched
                "property_mappings": material_mappings  # DB property -> target property mappings
            })
        
        # If no perfect matches, try partial matches
        if not candidates:
            matches = self.material_db.search_by_properties(properties_dict, match_all=False)
            for material in matches:
                material_props = material.get("properties", {})
                # Count how many properties were matched by checking which target properties
                # have corresponding DB properties that matched
                # Since search_by_properties handles mapping internally, we need to
                # re-check which properties actually matched
                matched_props = []
                material_mappings = {}
                property_mapper = self.material_db.property_mapper
                if property_mapper:
                    for target_prop in required_properties:
                        # Check if any DB property maps to this target property
                        for db_prop_name, db_value in material_props.items():
                            mapped_target = property_mapper.map_property_name(
                                db_prop_name,
                                [target_prop]
                            )
                            if mapped_target == target_prop:
                                target_val = properties_dict.get(target_prop)
                                if property_mapper.compare_property_values(db_value, target_val):
                                    matched_props.append(target_prop)
                                    if target_prop not in material_mappings:
                                        material_mappings[target_prop] = {}
                                    material_mappings[target_prop][db_prop_name] = db_value
                                    # Store in global mappings
                                    if target_prop not in property_mappings:
                                        property_mappings[target_prop] = {}
                                    if db_prop_name not in property_mappings[target_prop]:
                                        property_mappings[target_prop][db_prop_name] = db_value
                                    break
                
                matched_count = len(matched_props)
                total_count = len(required_properties)
                score = matched_count / total_count if total_count > 0 else 0.0
                
                candidates.append({
                    "material_id": material["material_id"],
                    "material_name": material["material_name"],
                    "properties": material_props,
                    "property_match_score": score,
                    "matched_properties": matched_props,
                    "property_mappings": material_mappings  # DB property -> target property mappings
                })
        
        # Sort by property match score
        candidates.sort(key=lambda x: x["property_match_score"], reverse=True)
        
        return candidates, property_mappings
    
    def evaluate_graph_evidence(
        self,
        candidates: List[Dict[str, Any]],
        material_informed_subgraph: nx.DiGraph,
        properties_W: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Use material-informed subgraph to find supporting evidence for candidates.
        
        Args:
            candidates: List of candidate materials
            material_informed_subgraph: Filtered subgraph
            properties_W: Property requirements
            
        Returns:
            Dictionary mapping material_id -> graph evidence dict
        """
        graph_evidence = {}
        required_properties = properties_W.get("required", [])
        
        if material_informed_subgraph is None or material_informed_subgraph.number_of_nodes() == 0:
            return {c["material_id"]: {"paths_found": 0, "nodes_connected": []} for c in candidates}
        
        # Try to find material nodes in subgraph
        for candidate in candidates:
            material_name = candidate["material_name"]
            material_id = candidate["material_id"]
            
            # Search for material name in subgraph nodes
            matching_nodes = []
            for node in material_informed_subgraph.nodes():
                node_attrs = material_informed_subgraph.nodes[node]
                node_text = str(node).lower()
                
                # Check if material name appears in node
                if material_name.lower() in node_text:
                    matching_nodes.append(node)
                
                # Check node attributes
                for attr_key in ["title", "label", "name", "description"]:
                    if attr_key in node_attrs:
                        attr_text = str(node_attrs[attr_key]).lower()
                        if material_name.lower() in attr_text:
                            matching_nodes.append(node)
                            break
            
            # Find paths to property nodes
            property_nodes = []
            for prop in required_properties:
                for node in material_informed_subgraph.nodes():
                    node_attrs = material_informed_subgraph.nodes[node]
                    node_text = str(node).lower()
                    
                    if prop.lower() in node_text:
                        property_nodes.append(node)
                        break
                    
                    for attr_key in ["title", "label", "name", "description"]:
                        if attr_key in node_attrs:
                            attr_text = str(node_attrs[attr_key]).lower()
                            if prop.lower() in attr_text:
                                property_nodes.append(node)
                                break
            
            # Count paths between material and property nodes
            paths_found = 0
            if matching_nodes and property_nodes:
                undirected = material_informed_subgraph.to_undirected()
                for mat_node in matching_nodes[:self.matching_nodes_preview]:
                    for prop_node in property_nodes[:self.property_nodes_preview]:
                        try:
                            if nx.has_path(undirected, mat_node, prop_node):
                                paths_found += 1
                        except (nx.NetworkXError, nx.NodeNotFound):
                            pass
            
            graph_evidence[material_id] = {
                "paths_found": paths_found,
                "nodes_connected": matching_nodes[:self.matching_nodes_in_dict],
                "property_nodes": property_nodes[:self.property_nodes_in_dict]
            }
        
        return graph_evidence
    
    def rank_candidates(
        self,
        property_matches: List[Dict[str, Any]],
        graph_evidence: Dict[str, Dict[str, Any]],
        rag_evidence: Dict[str, Dict[str, Any]],
        material_X: str,
        application_Y: str,
        properties_W: Dict[str, Any],
        constraints_U: List[str],
        temperature: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Rank candidates by combining property matching, graph evidence, and RAG evidence.
        
        Args:
            property_matches: Candidates from property matching
            graph_evidence: Graph-based evidence for each candidate
            rag_evidence: RAG evidence for each candidate
            material_X: Material to replace
            application_Y: Application context
            properties_W: Property requirements
            constraints_U: Constraints
            temperature: Temperature for LLM ranking (default: None, uses config value)
            
        Returns:
            Ranked list of candidates
        """
        # Use config default if not provided
        if temperature is None:
            temperature = self.temperature
        
        ranked = []
        
        for candidate in property_matches:
            material_id = candidate["material_id"]
            material_name = candidate["material_name"]
            
            # Get evidence
            graph_ev = graph_evidence.get(material_id, {})
            rag_ev = rag_evidence.get(material_id, {})
            
            # Calculate composite score
            property_score = candidate.get("property_match_score", 0.0)
            graph_score = min(graph_ev.get("paths_found", 0) / self.graph_normalization_divisor, 1.0)
            rag_score = min(rag_ev.get("num_documents", 0) / self.rag_normalization_divisor, 1.0)

            # Weighted combination
            composite_score = (
                self.property_weight * property_score +
                self.graph_weight * graph_score +
                self.rag_weight * rag_score
            )
            
            ranked.append({
                "material_id": material_id,
                "material_name": material_name,
                "property_matches": {
                    "score": property_score,
                    "matched_properties": candidate.get("matched_properties", [])
                },
                "graph_evidence": {
                    "paths_found": graph_ev.get("paths_found", 0),
                    "nodes_connected": graph_ev.get("nodes_connected", [])
                },
                "rag_evidence": {
                    "num_documents": rag_ev.get("num_documents", 0)
                },
                "properties": candidate.get("properties", {})
            })
        
        # Sort by property match score
        ranked.sort(key=lambda x: x["property_matches"]["score"], reverse=True)
        
        # Use LLM to refine ranking if we have RAG evidence
        if ranked and any(r["rag_evidence"]["num_documents"] > 0 for r in ranked[:3]):
            try:
                # Load prompts from YAML
                prompts = load_prompts()
                
                # Load user prompt template from YAML
                rank_candidates_user_prompt_template = prompts.get("agents", {}).get("material_scientist", {}).get("rank_candidates_user_prompt")
                if rank_candidates_user_prompt_template is None:
                    raise ValueError(
                        "Missing required prompt in config/prompts.yaml: agents.material_scientist.rank_candidates_user_prompt. "
                        "All system prompts must be defined in the config file."
                    )
                
                # Build candidate list
                top_3 = ranked[:3]
                candidate_list_parts = []
                for i, cand in enumerate(top_3, 1):
                    candidate_list_parts.append(
                        f"{i}. {cand['material_name']} - "
                        f"Property match: {cand['property_matches']['score']:.2f}, "
                        f"Graph paths: {cand['graph_evidence']['paths_found']}, "
                        f"RAG docs: {cand['rag_evidence']['num_documents']}"
                    )
                candidate_list = "\n".join(candidate_list_parts)
                
                # Format user prompt with dynamic content
                prompt = rank_candidates_user_prompt_template.format(
                    material_X=material_X,
                    application_Y=application_Y,
                    required_properties=', '.join(properties_W.get('required', [])),
                    candidate_list=candidate_list
                )
                
                ranking_system_prompt = prompts.get("agents", {}).get("material_scientist", {}).get("rank_candidates")
                if ranking_system_prompt is None:
                    raise ValueError(
                        "Missing required prompt in config/prompts.yaml: agents.material_scientist.rank_candidates. "
                        "All system prompts must be defined in the config file."
                    )
                
                # Use manager to get ranking
                ranking_response = self.manager._generate_fn(
                    system_prompt=ranking_system_prompt,
                    prompt=prompt,
                    temperature=temperature,
                    method_name="rank_candidates"
                )
                
                # Parse ranking (simple heuristic - can be enhanced)
                # For now, keep original ranking but could reorder based on LLM response
                
            except Exception as e:
                logger.warning("LLM ranking failed, using property-score ordering: %s", e)
        
        return ranked
    
    def generate_explanation(
        self,
        candidate: Dict[str, Any],
        material_X: str,
        application_Y: str,
        properties_W: Dict[str, Any]
    ) -> str:
        """
        Generate a structured explanation for why a candidate is a good substitute.
        
        Args:
            candidate: Top-ranked candidate material
            material_X: Material to replace
            application_Y: Application context
            properties_W: Property requirements
            
        Returns:
            Explanation string
        """
        material_name = candidate["material_name"]
        matched_props = candidate.get("property_matches", {}).get("matched_properties", [])
        graph_paths = candidate.get("graph_evidence", {}).get("paths_found", 0)
        
        explanation_parts = [
            f"{material_name} is proposed as a substitute for {material_X} in {application_Y}.",
            f"\nProperty Matches:",
            f"- Matched {len(matched_props)} out of {len(properties_W.get('required', []))} required properties",
            f"- Matched properties: {', '.join(matched_props) if matched_props else 'None'}",
            f"\nGraph Evidence:",
            f"- Found {graph_paths} paths connecting {material_name} to required properties in knowledge graph"
        ]
        
        return "\n".join(explanation_parts)

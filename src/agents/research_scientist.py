"""ResearchScientist Agent - Finds connections between keywords in knowledge graphs"""

import itertools
import json
from typing import List, Dict, Any, Optional
import numpy as np
import networkx as nx
from GraphReasoning import find_best_fitting_node_list
from ..config import load_prompts, load_config


class ResearchScientist:
    """
    A research scientist agent that takes keywords and finds connections between them
    in a knowledge graph. Uses semantic matching to link keywords to nodes, then
    finds paths/connections between those nodes.
    
    Supports single or multiple knowledge graphs with different query strategies.
    """
    
    def __init__(
        self,
        knowledge_graph: nx.DiGraph,
        node_embeddings: Dict[str, np.ndarray],
        embedding_tokenizer,
        embedding_model,
        algorithm: str = "shortest",
        n_samples: int = None,
        similarity_threshold: float = None,
        chat_logger=None,
        generate_fn=None,
        # Multi-KG support
        knowledge_graph_2: Optional[nx.DiGraph] = None,
        node_embeddings_2: Optional[Dict[str, np.ndarray]] = None,
        kg_names: Optional[List[str]] = None,
            kg_descriptions: Optional[List[str]] = None,
        multi_kg_strategy: str = "separate"
    ):
        """
        Initialize the ResearchScientist agent.
        
        Args:
            knowledge_graph: NetworkX knowledge graph (primary)
            node_embeddings: Dictionary mapping node IDs to embedding vectors (primary)
            embedding_tokenizer: Tokenizer for generating embeddings
            embedding_model: Model for generating embeddings
            algorithm: Path-finding algorithm ("shortest", "top_n", or "dfs") (default: "shortest")
            n_samples: Number of top matching nodes per keyword (default: None, uses config value)
            similarity_threshold: Minimum similarity threshold for keyword-to-node matching (default: None, uses config value)
            chat_logger: Optional ChatLogger instance for logging KG queries
            generate_fn: Optional LLM generate function for LLM-based node classification. If None, falls back to keyword-based filtering.
            knowledge_graph_2: Optional second NetworkX knowledge graph
            node_embeddings_2: Optional second dictionary mapping node IDs to embedding vectors
            kg_names: Optional list of names for the knowledge graphs (e.g., ["material_properties", "pfas"])
            kg_descriptions: Optional list of descriptions for the knowledge graphs
                (e.g., ["Material properties and characteristics", "PFAS compounds and their properties"])
            multi_kg_strategy: Strategy for using multiple KGs:
                - "separate": Query both KGs and keep results separate with descriptions (default)
                - "merged": Query both KGs and merge all results into a single combined result
        """
        # Load config
        config = load_config()
        agent_config = config.get("agents", {}).get("research_scientist", {})
        
        # Use config defaults if not provided
        if n_samples is None:
            n_samples = agent_config.get("n_samples", 5)
        if similarity_threshold is None:
            similarity_threshold = agent_config.get("similarity_threshold", 0.8)
        
        self.knowledge_graph = knowledge_graph
        self.node_embeddings = node_embeddings
        self.embedding_tokenizer = embedding_tokenizer
        self.embedding_model = embedding_model
        self.algorithm = algorithm
        self.n_samples = n_samples
        self.similarity_threshold = similarity_threshold
        
        # Load path finding config
        path_finding_config = agent_config.get("path_finding", {})
        self.top_paths = path_finding_config.get("top_paths", 10)
        self.max_depth = path_finding_config.get("max_depth", 10)
        
        # Load batch size and temperature config
        self.batch_size = agent_config.get("batch_size", 50)
        self.temperature = agent_config.get("temperature", 0)
        
        # Load material class extraction limits
        self.max_material_classes = agent_config.get("max_material_classes", 200)
        self.max_candidate_nodes = agent_config.get("max_candidate_nodes", 200)
        self.path_scoring_weight = agent_config.get("path_scoring_weight", 2.0)
        self.max_keywords_for_material_match = agent_config.get("max_keywords_for_material_match", 10)

        self.chat_logger = chat_logger
        self.generate_fn = generate_fn
        
        # Multi-KG support
        self.knowledge_graph_2 = knowledge_graph_2
        self.node_embeddings_2 = node_embeddings_2
        self.has_multiple_kgs = knowledge_graph_2 is not None and node_embeddings_2 is not None
        self.kg_names = kg_names if kg_names else (["kg1", "kg2"] if self.has_multiple_kgs else ["kg1"])
        self.kg_descriptions = kg_descriptions if kg_descriptions else (
            ["Knowledge Graph 1", "Knowledge Graph 2"] if self.has_multiple_kgs else ["Knowledge Graph 1"]
        )
        self.multi_kg_strategy = multi_kg_strategy if self.has_multiple_kgs else "single"
        
        if self.has_multiple_kgs:
            if len(self.kg_names) < 2:
                self.kg_names = ["kg1", "kg2"]
            if len(self.kg_descriptions) < 2:
                self.kg_descriptions = ["Knowledge Graph 1", "Knowledge Graph 2"]
            print(f"✓ ResearchScientist initialized with 2 knowledge graphs: {self.kg_names}")
            print(f"  Strategy: {self.multi_kg_strategy}")
            if self.multi_kg_strategy == "separate":
                print(f"  KG Descriptions: {self.kg_descriptions}")
    
    def _extract_path_metadata(self, path: List[str], graph: nx.Graph, fallback_graph: Optional[nx.Graph] = None) -> Dict[str, Any]:
        """Extract node and edge metadata for a given path.
        
        Args:
            path: List of node IDs in the path
            graph: Primary graph to extract metadata from (usually a subgraph)
            fallback_graph: Optional fallback graph (usually the full knowledge graph) to use
                          if edge attributes are missing in the primary graph
        """
        nodes_metadata = []
        edges_metadata = []
        
        for node_id in path:
            node_info = {"node_id": node_id}
            if node_id in graph.nodes():
                node_attrs = graph.nodes[node_id]
                node_info["node_data"] = dict(node_attrs)
            else:
                node_info["node_data"] = {}
            nodes_metadata.append(node_info)
        
        for i in range(len(path) - 1):
            source = path[i]
            target = path[i + 1]
            
            edge_info = {"source": source, "target": target}
            edge_data = {}
            
            # Try to get edge data from primary graph
            if graph.has_edge(source, target):
                edge_attrs = graph[source][target]
                edge_data = dict(edge_attrs)
            elif graph.has_edge(target, source):
                edge_attrs = graph[target][source]
                edge_data = dict(edge_attrs)
            
            # If no edge data found in primary graph, try fallback graph
            if not edge_data and fallback_graph is not None:
                if fallback_graph.has_edge(source, target):
                    edge_attrs = fallback_graph[source][target]
                    edge_data = dict(edge_attrs)
                elif fallback_graph.has_edge(target, source):
                    edge_attrs = fallback_graph[target][source]
                    edge_data = dict(edge_attrs)
            
            edge_info["edge_data"] = edge_data
            edges_metadata.append(edge_info)
        
        return {
            "nodes": nodes_metadata,
            "edges": edges_metadata
        }
    
    def format_separate_results(self, result: Dict[str, Any]) -> str:
        """
        Format results from separate KG strategy for human-readable display.
        
        Args:
            result: Result dictionary from find_connections with strategy="separate"
        
        Returns:
            Formatted string with results from both KGs clearly separated
        """
        if result.get("summary", {}).get("strategy") != "separate":
            return "This method only works with 'separate' strategy results."
        
        kg_results = result.get("kg_results", {})
        summary = result.get("summary", {})
        
        output_parts = []
        
        # First KG
        kg1_info = summary.get("kg1", {})
        kg1_name = kg1_info.get("name", "KG1")
        kg1_desc = kg1_info.get("description", "")
        kg1_result = kg_results.get(kg1_name, {}).get("result", {})
        kg1_summary = kg1_info.get("summary", {})
        
        output_parts.append("=" * 70)
        output_parts.append(f"KNOWLEDGE GRAPH 1: {kg1_name.upper()}")
        output_parts.append("=" * 70)
        output_parts.append(f"Description: {kg1_desc}")
        output_parts.append("")
        
        if kg1_summary.get("connections_found"):
            output_parts.append(f"✓ Connections found: {kg1_summary.get('num_paths_found', 0)} paths")
            output_parts.append(f"  Matched nodes: {kg1_summary.get('num_matched_nodes', 0)}")
            output_parts.append(f"  Subgraph: {kg1_summary.get('subgraph_nodes', 0)} nodes, {kg1_summary.get('subgraph_edges', 0)} edges")
            
            # Show sample paths
            kg1_paths = kg1_result.get("found_paths", [])
            if kg1_paths:
                output_parts.append(f"\n  Sample paths (showing first 3):")
                for i, path_info in enumerate(kg1_paths[:3], 1):
                    path = path_info.get("path", [])
                    if path:
                        output_parts.append(f"    {i}. {' → '.join(path[:5])}{'...' if len(path) > 5 else ''}")
        else:
            output_parts.append("✗ No connections found in this knowledge graph")
        
        output_parts.append("")
        output_parts.append("")
        
        # Second KG
        kg2_info = summary.get("kg2", {})
        kg2_name = kg2_info.get("name", "KG2")
        kg2_desc = kg2_info.get("description", "")
        kg2_result = kg_results.get(kg2_name, {}).get("result", {})
        kg2_summary = kg2_info.get("summary", {})
        
        output_parts.append("=" * 70)
        output_parts.append(f"KNOWLEDGE GRAPH 2: {kg2_name.upper()}")
        output_parts.append("=" * 70)
        output_parts.append(f"Description: {kg2_desc}")
        output_parts.append("")
        
        if kg2_summary.get("connections_found"):
            output_parts.append(f"✓ Connections found: {kg2_summary.get('num_paths_found', 0)} paths")
            output_parts.append(f"  Matched nodes: {kg2_summary.get('num_matched_nodes', 0)}")
            output_parts.append(f"  Subgraph: {kg2_summary.get('subgraph_nodes', 0)} nodes, {kg2_summary.get('subgraph_edges', 0)} edges")
            
            # Show sample paths
            kg2_paths = kg2_result.get("found_paths", [])
            if kg2_paths:
                output_parts.append(f"\n  Sample paths (showing first 3):")
                for i, path_info in enumerate(kg2_paths[:3], 1):
                    path = path_info.get("path", [])
                    if path:
                        output_parts.append(f"    {i}. {' → '.join(path[:5])}{'...' if len(path) > 5 else ''}")
        else:
            output_parts.append("✗ No connections found in this knowledge graph")
        
        return "\n".join(output_parts)
    
    def _find_connections_single_kg(
        self,
        keywords: List[str],
        knowledge_graph: nx.DiGraph,
        node_embeddings: Dict[str, np.ndarray],
        kg_name: str = "kg1",
        use_best_match_only: bool = True
    ) -> Dict[str, Any]:
        """
        Internal method to find connections in a single knowledge graph.
        This is the core logic extracted from the original find_connections method.
        """
        if len(keywords) < 2:
            return {
                "keyword_to_nodes": {},
                "matched_node_ids": [],
                "connection_subgraph": None,
                "connections_text": "Not enough keywords to find connections (need at least 2).",
                "found_paths": [],
                "summary": {
                    "num_keywords": len(keywords),
                    "num_matched_nodes": 0,
                    "connections_found": False,
                    "kg_name": kg_name
                }
            }
        
        # Step 1: Link keywords to nodes
        keyword_mappings = []
        all_matched_node_ids = []
        
        for keyword in keywords:
            if not isinstance(keyword, str) or not keyword.strip():
                continue
            
            # Find best matching nodes for this keyword
            matched_nodes = find_best_fitting_node_list(
                keyword,
                node_embeddings,
                self.embedding_tokenizer,
                self.embedding_model,
                N_samples=self.n_samples,
                similarity_threshold=self.similarity_threshold
            )
            
            formatted_matches = []
            for node_id, similarity in matched_nodes:
                node_info = {
                    "node_id": node_id,
                    "similarity": float(similarity),
                    "kg_name": kg_name
                }
                
                if knowledge_graph is not None and node_id in knowledge_graph.nodes():
                    node_attrs = knowledge_graph.nodes[node_id]
                    node_info["node_data"] = dict(node_attrs)
                else:
                    node_info["node_data"] = {}
                
                formatted_matches.append(node_info)
                all_matched_node_ids.append(node_id)
            
            keyword_mappings.append({
                "keyword": keyword,
                "matched_nodes": formatted_matches,
                "num_matches": len(formatted_matches)
            })
        
        # Step 2: Extract node IDs for path finding
        if use_best_match_only:
            matched_node_ids = []
            for km in keyword_mappings:
                if km['matched_nodes']:
                    best_node = km['matched_nodes'][0]['node_id']
                    matched_node_ids.append(best_node)
        else:
            matched_node_ids = all_matched_node_ids
        
        # Remove duplicates while preserving order
        seen = set()
        unique_node_ids = []
        for node_id in matched_node_ids:
            if node_id not in seen:
                seen.add(node_id)
                unique_node_ids.append(node_id)
        
        if len(unique_node_ids) < 2:
            return {
                "keyword_to_nodes": {"keyword_mappings": keyword_mappings},
                "matched_node_ids": unique_node_ids,
                "connection_subgraph": None,
                "connections_text": "Not enough matched nodes to find connections (need at least 2).",
                "found_paths": [],
                "summary": {
                    "num_keywords": len(keywords),
                    "num_matched_nodes": len(unique_node_ids),
                    "connections_found": False,
                    "kg_name": kg_name
                }
            }
        
        # Step 3: Find connections/paths between nodes
        found_paths = []
        try:
            undirected_graph = knowledge_graph.to_undirected()
            
            if self.algorithm == "shortest":
                subgraph_nodes_set = set()
                found = 0
                all_path = 0
                
                for i in range(len(unique_node_ids)):
                    for j in range(i+1, len(unique_node_ids)):
                        all_path += 1
                        try:
                            path = nx.shortest_path(undirected_graph, unique_node_ids[i], unique_node_ids[j])
                            
                            # Use original knowledge graph as fallback if subgraph doesn't have edge attributes
                            fallback_graph = self.knowledge_graph if knowledge_graph is not self.knowledge_graph else None
                            path_metadata = self._extract_path_metadata(path, knowledge_graph, fallback_graph=fallback_graph)
                            
                            found_paths.append({
                                "source": unique_node_ids[i],
                                "target": unique_node_ids[j],
                                "path": path,
                                "length": len(path) - 1,
                                "nodes": path_metadata["nodes"],
                                "edges": path_metadata["edges"],
                                "kg_name": kg_name
                            })
                            subgraph_nodes_set.update(path)
                            found += 1
                        except nx.NetworkXNoPath:
                            pass
                
                connection_subgraph = undirected_graph.subgraph(list(subgraph_nodes_set)) if subgraph_nodes_set else None
                
            elif self.algorithm == "top_n":
                subgraph_nodes_set = set()
                subgraph_edges_set = set()
                found = 0
                
                for i in range(len(unique_node_ids)):
                    for j in range(i + 1, len(unique_node_ids)):
                        source = unique_node_ids[i]
                        target = unique_node_ids[j]
                        
                        try:
                            from networkx.algorithms.simple_paths import shortest_simple_paths
                            
                            path_generator = shortest_simple_paths(undirected_graph, source, target)
                            top_paths = list(itertools.islice(path_generator, self.top_paths))
                            
                            for path in top_paths:
                                # Use original knowledge graph as fallback if subgraph doesn't have edge attributes
                                fallback_graph = self.knowledge_graph if knowledge_graph is not self.knowledge_graph else None
                                path_metadata = self._extract_path_metadata(path, knowledge_graph, fallback_graph=fallback_graph)
                                
                                found_paths.append({
                                    "source": source,
                                    "target": target,
                                    "path": path,
                                    "length": len(path) - 1,
                                    "nodes": path_metadata["nodes"],
                                    "edges": path_metadata["edges"],
                                    "kg_name": kg_name
                                })
                                subgraph_nodes_set.update(path)
                                subgraph_edges_set.update(zip(path[:-1], path[1:]))
                            
                            found += 1
                        except nx.NetworkXNoPath:
                            pass
                
                connection_subgraph = undirected_graph.edge_subgraph(subgraph_edges_set).copy()
                
            elif self.algorithm == "dfs":
                subgraph_nodes_set = set()
                found = 0
                
                def dfs(current, target, visited, depth, max_depth=None):
                    if max_depth is None:
                        max_depth = self.max_depth
                    if current == target:
                        return [current]
                    if depth >= max_depth:
                        return None
                    visited.add(current)
                    for neighbor in undirected_graph.neighbors(current):
                        if neighbor not in visited:
                            result = dfs(neighbor, target, visited, depth + 1, max_depth)
                            if result is not None:
                                return [current] + result
                    visited.remove(current)
                    return None
                
                for i in range(len(unique_node_ids)):
                    for j in range(i + 1, len(unique_node_ids)):
                        visited = set()
                        path = dfs(unique_node_ids[i], unique_node_ids[j], visited, 0, self.max_depth)
                        if path:
                            # Use original knowledge graph as fallback if subgraph doesn't have edge attributes
                            fallback_graph = self.knowledge_graph if knowledge_graph is not self.knowledge_graph else None
                            path_metadata = self._extract_path_metadata(path, knowledge_graph, fallback_graph=fallback_graph)
                            
                            found_paths.append({
                                "source": unique_node_ids[i],
                                "target": unique_node_ids[j],
                                "path": path,
                                "length": len(path) - 1,
                                "nodes": path_metadata["nodes"],
                                "edges": path_metadata["edges"],
                                "kg_name": kg_name
                            })
                            subgraph_nodes_set.update(path)
                            found += 1
                
                connection_subgraph = undirected_graph.subgraph(list(subgraph_nodes_set))
            else:
                raise ValueError(f"Unknown algorithm: {self.algorithm}. Use 'shortest', 'top_n', or 'dfs'")
            
            connections_text = f"Found {len(found_paths)} paths between {len(unique_node_ids)} nodes in {kg_name}."
            
            connections_found = len(found_paths) > 0
            
            all_path_nodes = set()
            for path_info in found_paths:
                all_path_nodes.update(path_info.get('path', []))
            subgraph_nodes = len(all_path_nodes)
            subgraph_edges = sum(path_info.get('length', 0) for path_info in found_paths)
            
            try:
                if connection_subgraph:
                    subgraph_nodes_from_graph = connection_subgraph.number_of_nodes()
                    subgraph_edges_from_graph = connection_subgraph.number_of_edges()
                    if subgraph_nodes_from_graph > 0:
                        subgraph_nodes = subgraph_nodes_from_graph
                    if subgraph_edges_from_graph > 0:
                        subgraph_edges = subgraph_edges_from_graph
            except Exception:
                pass
            
            summary = {
                "num_keywords": len(keywords),
                "num_matched_nodes": len(unique_node_ids),
                "subgraph_nodes": subgraph_nodes,
                "subgraph_edges": subgraph_edges,
                "num_paths_found": len(found_paths),
                "connections_found": connections_found,
                "algorithm_used": self.algorithm,
                "kg_name": kg_name
            }
            
        except Exception as e:
            if 'found_paths' not in locals():
                found_paths = []
            
            connection_subgraph = None
            connections_found = len(found_paths) > 0
            
            if connections_found:
                connections_text = f"Found {len(found_paths)} paths between {len(unique_node_ids)} nodes in {kg_name} (processing note: {str(e)})"
            else:
                connections_text = f"Error finding connections in {kg_name}: {str(e)}"
            
            all_path_nodes = set()
            for path_info in found_paths:
                all_path_nodes.update(path_info.get('path', []))
            subgraph_nodes = len(all_path_nodes)
            subgraph_edges = sum(path_info.get('length', 0) for path_info in found_paths)
            
            summary = {
                "num_keywords": len(keywords),
                "num_matched_nodes": len(unique_node_ids),
                "subgraph_nodes": subgraph_nodes,
                "subgraph_edges": subgraph_edges,
                "num_paths_found": len(found_paths),
                "connections_found": connections_found,
                "algorithm_used": self.algorithm,
                "kg_name": kg_name,
                "error": str(e) if not connections_found else None
            }
        
        return {
            "keyword_to_nodes": {"keyword_mappings": keyword_mappings},
            "matched_node_ids": unique_node_ids,
            "connection_subgraph": connection_subgraph,
            "connections_text": connections_text,
            "found_paths": found_paths,
            "summary": summary
        }
    
    def find_connections(
        self,
        keywords: List[str],
        use_best_match_only: bool = True
    ) -> Dict[str, Any]:
        """
        Find connections between keywords in the knowledge graph(s).
        
        Supports single or multiple knowledge graphs based on initialization.
        If multiple KGs are configured, uses the strategy specified in multi_kg_strategy.
        
        Args:
            keywords: List of keyword strings to find connections between
            use_best_match_only: If True, use only the best matching node per keyword;
                                If False, use all matched nodes (default: False)
        
        Returns:
            Dict containing connection results with summary, found_paths, etc.
            For multiple KGs, results are merged according to the strategy.
        """
        if not isinstance(keywords, list):
            raise ValueError(f"keywords must be a list, got {type(keywords)}")
        
        # Single KG mode (backward compatible)
        if not self.has_multiple_kgs:
            result = self._find_connections_single_kg(
                keywords=keywords,
                knowledge_graph=self.knowledge_graph,
                node_embeddings=self.node_embeddings,
                kg_name=self.kg_names[0],
                use_best_match_only=use_best_match_only
            )
            
            # Log KG query if chat_logger is provided
            if self.chat_logger is not None:
                try:
                    self.chat_logger.log_kg_query(
                        agent_name="research_scientist",
                        method_name="find_connections",
                        keywords=keywords,
                        result=result
                    )
                except Exception as e:
                    print(f"Warning: Failed to log KG query: {e}")
            
            return result
        
        # Multi-KG mode
        if self.multi_kg_strategy == "merged":
            # Query both KGs and merge results
            result1 = self._find_connections_single_kg(
                keywords=keywords,
                knowledge_graph=self.knowledge_graph,
                node_embeddings=self.node_embeddings,
                kg_name=self.kg_names[0],
                use_best_match_only=use_best_match_only
            )
            
            result2 = self._find_connections_single_kg(
                keywords=keywords,
                knowledge_graph=self.knowledge_graph_2,
                node_embeddings=self.node_embeddings_2,
                kg_name=self.kg_names[1],
                use_best_match_only=use_best_match_only
            )
            
            # Merge results
            merged_keyword_mappings = {}
            for km in result1.get("keyword_to_nodes", {}).get("keyword_mappings", []):
                keyword = km["keyword"]
                merged_keyword_mappings[keyword] = {
                    "keyword": keyword,
                    "matched_nodes": km["matched_nodes"],
                    "num_matches": km["num_matches"]
                }
            
            # Add matches from second KG
            for km in result2.get("keyword_to_nodes", {}).get("keyword_mappings", []):
                keyword = km["keyword"]
                if keyword in merged_keyword_mappings:
                    # Merge matched nodes, avoiding duplicates
                    existing_node_ids = {n["node_id"] for n in merged_keyword_mappings[keyword]["matched_nodes"]}
                    for node_info in km["matched_nodes"]:
                        if node_info["node_id"] not in existing_node_ids:
                            merged_keyword_mappings[keyword]["matched_nodes"].append(node_info)
                    merged_keyword_mappings[keyword]["num_matches"] = len(merged_keyword_mappings[keyword]["matched_nodes"])
                else:
                    merged_keyword_mappings[keyword] = km
            
            # Combine paths from both KGs
            all_paths = result1.get("found_paths", []) + result2.get("found_paths", [])
            all_matched_nodes = list(set(result1.get("matched_node_ids", []) + result2.get("matched_node_ids", [])))
            
            # Create combined summary
            summary1 = result1.get("summary", {})
            summary2 = result2.get("summary", {})
            
            combined_summary = {
                "num_keywords": len(keywords),
                "num_matched_nodes": len(all_matched_nodes),
                "subgraph_nodes": summary1.get("subgraph_nodes", 0) + summary2.get("subgraph_nodes", 0),
                "subgraph_edges": summary1.get("subgraph_edges", 0) + summary2.get("subgraph_edges", 0),
                "num_paths_found": len(all_paths),
                "connections_found": len(all_paths) > 0,
                "algorithm_used": self.algorithm,
                "kg1_summary": summary1,
                "kg2_summary": summary2,
                "strategy": "merged"
            }
            
            connections_text = f"Found {len(all_paths)} paths across both knowledge graphs ({self.kg_names[0]}: {summary1.get('num_paths_found', 0)}, {self.kg_names[1]}: {summary2.get('num_paths_found', 0)})."
            
            result = {
                "keyword_to_nodes": {"keyword_mappings": list(merged_keyword_mappings.values())},
                "matched_node_ids": all_matched_nodes,
                "connection_subgraph": None,  # Can't easily merge subgraphs, set to None
                "connections_text": connections_text,
                "found_paths": all_paths,
                "summary": combined_summary,
                "kg_results": {
                    self.kg_names[0]: result1,
                    self.kg_names[1]: result2
                }
            }
            
        elif self.multi_kg_strategy == "separate":
            # Query both KGs and keep results separate with descriptions
            result1 = self._find_connections_single_kg(
                keywords=keywords,
                knowledge_graph=self.knowledge_graph,
                node_embeddings=self.node_embeddings,
                kg_name=self.kg_names[0],
                use_best_match_only=use_best_match_only
            )
            
            result2 = self._find_connections_single_kg(
                keywords=keywords,
                knowledge_graph=self.knowledge_graph_2,
                node_embeddings=self.node_embeddings_2,
                kg_name=self.kg_names[1],
                use_best_match_only=use_best_match_only
            )
            
            summary1 = result1.get("summary", {})
            summary2 = result2.get("summary", {})
            
            # Create formatted connections text with descriptions
            connections_text_parts = []
            
            # First KG results
            connections_text_parts.append(
                f"=== {self.kg_names[0].upper()} ===\n"
                f"Description: {self.kg_descriptions[0]}\n"
                f"Connections found: {summary1.get('num_paths_found', 0)} paths between "
                f"{summary1.get('num_matched_nodes', 0)} matched nodes.\n"
                f"{result1.get('connections_text', '')}"
            )
            
            # Second KG results
            connections_text_parts.append(
                f"\n=== {self.kg_names[1].upper()} ===\n"
                f"Description: {self.kg_descriptions[1]}\n"
                f"Connections found: {summary2.get('num_paths_found', 0)} paths between "
                f"{summary2.get('num_matched_nodes', 0)} matched nodes.\n"
                f"{result2.get('connections_text', '')}"
            )
            
            combined_connections_text = "\n".join(connections_text_parts)
            
            # Create summary that indicates separate results
            # Include top-level subgraph_nodes and subgraph_edges for backward compatibility with pipeline
            combined_summary = {
                "num_keywords": len(keywords),
                "connections_found": summary1.get("connections_found", False) or summary2.get("connections_found", False),
                "algorithm_used": self.algorithm,
                "strategy": "separate",
                # Top-level fields for backward compatibility with pipeline code
                "subgraph_nodes": summary1.get("subgraph_nodes", 0) + summary2.get("subgraph_nodes", 0),
                "subgraph_edges": summary1.get("subgraph_edges", 0) + summary2.get("subgraph_edges", 0),
                "num_paths_found": summary1.get("num_paths_found", 0) + summary2.get("num_paths_found", 0),
                "num_matched_nodes": len(set(result1.get("matched_node_ids", []) + result2.get("matched_node_ids", []))),
                # Nested KG-specific summaries
                "kg1": {
                    "name": self.kg_names[0],
                    "description": self.kg_descriptions[0],
                    "summary": summary1
                },
                "kg2": {
                    "name": self.kg_names[1],
                    "description": self.kg_descriptions[1],
                    "summary": summary2
                }
            }
            
            result = {
                "connections_text": combined_connections_text,
                "summary": combined_summary,
                "kg_results": {
                    self.kg_names[0]: {
                        "description": self.kg_descriptions[0],
                        "result": result1
                    },
                    self.kg_names[1]: {
                        "description": self.kg_descriptions[1],
                        "result": result2
                    }
                },
                # Keep individual results accessible
                "kg1_result": result1,
                "kg2_result": result2,
                # For backward compatibility, provide combined keyword mappings
                "keyword_to_nodes": {
                    "keyword_mappings": [
                        {
                            "keyword": km["keyword"],
                            "matched_nodes": km["matched_nodes"],
                            "num_matches": km["num_matches"],
                            "kg_source": self.kg_names[0]
                        }
                        for km in result1.get("keyword_to_nodes", {}).get("keyword_mappings", [])
                    ] + [
                        {
                            "keyword": km["keyword"],
                            "matched_nodes": km["matched_nodes"],
                            "num_matches": km["num_matches"],
                            "kg_source": self.kg_names[1]
                        }
                        for km in result2.get("keyword_to_nodes", {}).get("keyword_mappings", [])
                    ]
                },
                "matched_node_ids": list(set(result1.get("matched_node_ids", []) + result2.get("matched_node_ids", []))),
                "found_paths": result1.get("found_paths", []) + result2.get("found_paths", [])
            }
        
        else:
            raise ValueError(f"Unknown multi_kg_strategy: {self.multi_kg_strategy}. Use 'merged' or 'separate'")
        
        # Log KG query if chat_logger is provided
        if self.chat_logger is not None:
            try:
                self.chat_logger.log_kg_query(
                    agent_name="research_scientist",
                    method_name="find_connections",
                    keywords=keywords,
                    result=result
                )
            except Exception as e:
                print(f"Warning: Failed to log KG query: {e}")
        
        return result
    
    def _classify_nodes_batch(
        self,
        candidate_nodes: List[str],
        application_Y: str,
        required_properties: List[str],
        batch_size: int = None,
        temperature: float = None
    ) -> List[str]:
        """
        Classify a batch of nodes as materials using LLM.
        
        Args:
            candidate_nodes: List of node IDs to classify
            application_Y: Target application description
            required_properties: List of required property names
            batch_size: Number of nodes to process per LLM call (default: None, uses config value)
            temperature: Temperature for LLM generation (default: None, uses config value)
        
        Returns:
            List of node IDs classified as materials
        """
        # Use config defaults if not provided
        if batch_size is None:
            batch_size = self.batch_size
        if temperature is None:
            temperature = self.temperature
        
        if not candidate_nodes:
            return []
        
        if self.generate_fn is None:
            return []
        
        # Load prompt from config
        prompts = load_prompts()
        system_prompt_template = prompts.get("agents", {}).get("research_scientist", {}).get("classify_material_nodes")
        if system_prompt_template is None:
            print("Warning: Missing prompt config/prompts.yaml: agents.research_scientist.classify_material_nodes. Falling back to keyword-based filtering.")
            return []
        
        # Format system prompt with context
        required_properties_str = ', '.join(required_properties) if required_properties else 'None'
        formatted_system_prompt = system_prompt_template.format(
            application_Y=application_Y,
            required_properties=required_properties_str
        )
        
        material_nodes = []
        
        # Process nodes in batches
        for batch_start in range(0, len(candidate_nodes), batch_size):
            batch_nodes = candidate_nodes[batch_start:batch_start + batch_size]
            
            # Format node data for the prompt
            node_data_list = []
            for node_id in batch_nodes:
                node_attrs = {}
                if self.knowledge_graph is not None and node_id in self.knowledge_graph.nodes():
                    node_attrs = self.knowledge_graph.nodes[node_id]
                
                # Build node description from attributes
                node_info = {
                    'node_id': str(node_id),
                    'attributes': {}
                }
                
                # Extract relevant attributes
                for attr_key in ["title", "label", "name", "description", "material_name", "material_class"]:
                    if attr_key in node_attrs:
                        node_info['attributes'][attr_key] = str(node_attrs[attr_key])
                
                node_data_list.append(node_info)
            
            # Load user prompt template from YAML
            classify_nodes_user_prompt_template = prompts.get("agents", {}).get("research_scientist", {}).get("classify_nodes_batch_user_prompt")
            if classify_nodes_user_prompt_template is None:
                raise ValueError(
                    "Missing required prompt in config/prompts.yaml: agents.research_scientist.classify_nodes_batch_user_prompt. "
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
            user_prompt = classify_nodes_user_prompt_template.format(
                num_nodes=len(batch_nodes),
                node_list=node_list
            )
            
            # Call LLM
            try:
                response = self.generate_fn(
                    system_prompt=formatted_system_prompt,
                    prompt=user_prompt,
                    temperature=temperature,
                    chat_logger=self.chat_logger,
                    agent_name="research_scientist",
                    method_name="_classify_nodes_batch"
                )
                
                # Parse JSON response
                response = response.strip()
                
                # Try to find JSON object in response
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    result = json.loads(json_str)
                    
                    # Extract classified nodes
                    batch_material_nodes = result.get('material_nodes', [])
                    
                    # Convert node ID strings back to original node IDs
                    node_id_to_node = {str(node): node for node in batch_nodes}
                    
                    for node_id_str in batch_material_nodes:
                        if node_id_str in node_id_to_node:
                            material_nodes.append(node_id_to_node[node_id_str])
                else:
                    print(f"Warning: Could not parse JSON from LLM response in batch {batch_start // batch_size + 1}")
            
            except json.JSONDecodeError as e:
                print(f"Warning: JSON parsing error in batch {batch_start // batch_size + 1}: {e}")
                if 'response' in locals():
                    print(f"Response was: {response[:200]}...")
            except Exception as e:
                print(f"Warning: Error processing batch {batch_start // batch_size + 1}: {e}")
                # Continue with next batch
        
        return material_nodes
    
    def _is_likely_material_keyword_based(self, node_id: str, node_attrs: dict) -> bool:
        """
        Fallback keyword-based method to check if a node is likely a material.
        Used when generate_fn is not available or LLM classification fails.
        """
        if self.knowledge_graph is None or node_id not in self.knowledge_graph.nodes():
            return False

        # --- Primary check: trust the KG's own 'type' annotation -----------
        _MATERIAL_TYPES = {
            'material', 'chemical', 'compound', 'composite material', 'composite',
            'polymer', 'fluoropolymer resin', 'chemical compound', 'element',
            'substance', 'solvent', 'fluid', 'material category', 'material family',
            'additive', 'material sample', 'material composition', 'material layer',
            'material_property',  # keep for safety; rarely a false positive
        }
        _NON_MATERIAL_TYPES = {
            'property', 'condition', 'parameter', 'method', 'process', 'application',
            'phenomenon', 'characterization technique', 'deformation mechanism',
            'relaxation process', 'concept', 'technique', 'diagram',
            'data representation', 'experimental method', 'test', 'test system',
            'experimental setup', 'equipment', 'model', 'mechanism', 'reaction',
            'thermal treatment', 'surface property', 'thermal property',
        }
        node_type = str(node_attrs.get('type', '')).strip().lower()
        if node_type in _MATERIAL_TYPES:
            return True
        if node_type in _NON_MATERIAL_TYPES:
            return False

        # --- Secondary check: node attribute fields -------------------------
        if any(attr in node_attrs for attr in ["material_name", "material_class", "polymer", "compound"]):
            return True

        # --- Tertiary check: keyword scan of node ID and text attributes ----
        # Keywords that indicate a material (polymers, compounds, etc.)
        material_keywords = [
            'polymer', 'material', 'compound', 'resin', 'plastic', 'elastomer',
            'peek', 'pps', 'pvdf', 'ptfe', 'pfa', 'pei', 'pa', 'pp', 'pe',
            'poly', 'nylon', 'polyamide', 'polyester', 'polycarbonate', 'polyimide',
            'fluoropolymer', 'thermoplastic', 'thermoset', 'rubber', 'silicone',
            'ceramic', 'metal', 'alloy', 'composite', 'fiber', 'fabric'
        ]

        # Keywords that indicate a property/concept (should be excluded)
        property_keywords = [
            'property', 'characteristic', 'behavior', 'performance', 'measurement',
            'test', 'method', 'standard', 'specification', 'requirement', 'constraint',
            'vibration', 'stress', 'strain', 'deformation', 'failure', 'degradation',
            'stability', 'resistance', 'tolerance', 'compatibility', 'coefficient',
            'modulus', 'strength', 'hardness', 'temperature', 'pressure', 'density'
        ]

        # Get node text from various attributes
        node_text = str(node_id).lower()
        for attr_key in ["title", "label", "name", "description", "material_name", "material_class"]:
            if attr_key in node_attrs:
                node_text += " " + str(node_attrs[attr_key]).lower()

        has_material_keyword = any(keyword in node_text for keyword in material_keywords)
        has_property_keyword = any(keyword in node_text for keyword in property_keywords)

        # If it has material keywords and not property keywords, likely a material
        if has_material_keyword and not has_property_keyword:
            return True

        # If node_id itself looks like a material (contains common material patterns)
        if any(keyword in node_id.lower() for keyword in material_keywords[:self.max_keywords_for_material_match]):
            return True

        return False

    def map_properties_to_materials(
        self,
        properties_W: Dict[str, Any],
        application_Y: str,
        subgraph: Optional[nx.DiGraph] = None
    ) -> Dict[str, Any]:
        """
        Map property requirements to candidate material classes/families via knowledge graph traversal.
        
        This method translates abstract property requirements W into concrete material knowledge by:
        1. Mapping each property to KG nodes via semantic matching
        2. Finding connections between properties and material-related nodes
        3. Identifying candidate material classes/families
        
        Args:
            properties_W: Dictionary containing property requirements:
                - "required": List of property names (e.g., ["thermal_stability", "dielectric_constant"])
                - "target_values": Optional dict mapping property names to target values/ranges
            application_Y: Target application description (e.g., "industrial seals applications")
            subgraph: Optional NetworkX subgraph to query instead of full knowledge graph.
                     When provided, only nodes in this subgraph will be used for matching and path finding.
        
        Returns:
            Dict containing:
                - "property_mappings": List of dicts mapping each property to matched KG nodes
                - "material_classes": List of candidate material classes/families discovered
                - "kg_insights": Summary of KG traversal results
                - "matched_nodes": List of matched node IDs
                - "summary": High-level summary statistics
        """
        if not isinstance(properties_W, dict):
            raise ValueError(f"properties_W must be a dictionary, got {type(properties_W)}")
        
        if "required" not in properties_W:
            raise ValueError("properties_W must contain 'required' key with list of property names")
        
        required_properties = properties_W.get("required", [])
        target_values = properties_W.get("target_values", {})
        
        if not isinstance(required_properties, list) or len(required_properties) == 0:
            raise ValueError("properties_W['required'] must be a non-empty list")
        
        if not isinstance(application_Y, str) or not application_Y.strip():
            raise ValueError("application_Y must be a non-empty string")
        
        # Determine which graph and embeddings to use
        if subgraph is not None:
            # Use subgraph for querying
            query_graph = subgraph
            # Filter embeddings to only include nodes in subgraph
            subgraph_nodes = set(subgraph.nodes())
            query_embeddings = {node_id: emb for node_id, emb in self.node_embeddings.items() 
                              if node_id in subgraph_nodes}
            print(f"  Mapping {len(required_properties)} properties to materials via subgraph ({subgraph.number_of_nodes()} nodes)...")
        else:
            # Use full knowledge graph (backward compatibility)
            query_graph = self.knowledge_graph
            query_embeddings = self.node_embeddings
            print(f"  Mapping {len(required_properties)} properties to materials via KG...")
        
        # Check if subgraph is empty
        if subgraph is not None and subgraph.number_of_nodes() == 0:
            print("  Warning: Subgraph is empty, returning empty results")
            return {
                "property_mappings": [],
                "material_classes": [],
                "kg_insights": {
                    "num_properties_mapped": 0,
                    "num_property_nodes": 0,
                    "num_material_nodes": 0,
                    "num_paths_found": 0,
                    "connection_algorithm": self.algorithm
                },
                "matched_nodes": [],
                "summary": {
                    "num_properties": len(required_properties),
                    "num_property_mappings": 0,
                    "num_material_classes": 0,
                    "connections_found": False
                }
            }
        
        # Step 1: Map each property to KG nodes
        property_mappings = []
        all_property_nodes = []
        
        for prop in required_properties:
            if not isinstance(prop, str) or not prop.strip():
                continue
            
            # Find best matching nodes for this property (using filtered embeddings if subgraph provided)
            matched_nodes = find_best_fitting_node_list(
                prop,
                query_embeddings,
                self.embedding_tokenizer,
                self.embedding_model,
                N_samples=self.n_samples,
                similarity_threshold=self.similarity_threshold
            )
            
            formatted_matches = []
            for node_id, similarity in matched_nodes:
                node_info = {
                    "node_id": node_id,
                    "similarity": float(similarity),
                }
                
                if query_graph is not None and node_id in query_graph.nodes():
                    node_attrs = query_graph.nodes[node_id]
                    node_info["node_data"] = dict(node_attrs)
                else:
                    node_info["node_data"] = {}
                
                formatted_matches.append(node_info)
                all_property_nodes.append(node_id)
            
            property_mappings.append({
                "property": prop,
                "target_value": target_values.get(prop, None),
                "matched_nodes": formatted_matches,
                "num_matches": len(formatted_matches)
            })
        
        # Step 2: Find material nodes connected to property nodes
        # Use the application context to find material classes
        application_keywords = [application_Y] + required_properties
        
        # If subgraph is provided, use _find_connections_single_kg directly with subgraph
        # Otherwise, use find_connections which handles multi-KG mode
        if subgraph is not None:
            # Query subgraph directly (ignore multi-KG mode when subgraph is provided)
            material_connections = self._find_connections_single_kg(
                keywords=application_keywords,
                knowledge_graph=subgraph,
                node_embeddings=query_embeddings,
                kg_name="subgraph",
                use_best_match_only=True
            )
        else:
            # Use full find_connections (supports multi-KG mode)
            material_connections = self.find_connections(
                keywords=application_keywords,
                use_best_match_only=True
            )
        
        # Extract material-related nodes from the connection results
        matched_material_nodes = material_connections.get("matched_node_ids", [])
        found_paths = material_connections.get("found_paths", [])
        
        # Step 3: Identify material classes from connected nodes
        material_classes = []
        material_node_set = set()
        
        # Collect all candidate nodes (excluding known property nodes)
        candidate_nodes = []
        candidate_node_set = set()
        
        # Collect nodes from paths between properties and materials
        for path_info in found_paths:
            path_nodes = path_info.get("path", [])
            for node_id in path_nodes:
                if node_id not in all_property_nodes:  # Exclude property nodes
                    if node_id not in candidate_node_set:
                        if query_graph is not None and node_id in query_graph.nodes():
                            candidate_nodes.append(node_id)
                            candidate_node_set.add(node_id)
        
        # Also include directly matched nodes that aren't properties
        for node_id in matched_material_nodes:
            if node_id not in all_property_nodes:
                if node_id not in candidate_node_set:
                    if query_graph is not None and node_id in query_graph.nodes():
                        candidate_nodes.append(node_id)
                        candidate_node_set.add(node_id)
        
        # Classify nodes using LLM if available, otherwise fall back to keyword-based
        if self.generate_fn is not None and len(candidate_nodes) > 0:
            try:
                # Use LLM-based classification
                material_node_ids = self._classify_nodes_batch(
                    candidate_nodes=candidate_nodes,
                    application_Y=application_Y,
                    required_properties=required_properties,
                    batch_size=self.batch_size,
                    temperature=self.temperature
                )
                material_node_set = set(material_node_ids)
            except Exception as e:
                print(f"Warning: LLM classification failed ({e}), falling back to keyword-based filtering")
                # Fall back to keyword-based approach
                for node_id in candidate_nodes:
                    if query_graph is not None and node_id in query_graph.nodes():
                        node_attrs = query_graph.nodes[node_id]
                        if self._is_likely_material_keyword_based(node_id, node_attrs):
                            material_node_set.add(node_id)
        else:
            # Use keyword-based approach as fallback
            if self.generate_fn is None:
                print("Warning: generate_fn not provided, using keyword-based filtering (deprecated)")
            
            for node_id in candidate_nodes:
                if query_graph is not None and node_id in query_graph.nodes():
                    node_attrs = query_graph.nodes[node_id]
                    if self._is_likely_material_keyword_based(node_id, node_attrs):
                        material_node_set.add(node_id)
        
        # Format material class information for ALL classified material nodes first
        material_candidates = []
        for node_id in list(material_node_set):  # iterate full set before any truncation
            if query_graph is not None and node_id in query_graph.nodes():
                node_attrs = query_graph.nodes[node_id]
                # Get material name from node attributes
                material_name = None
                for attr_key in ["material_name", "title", "label", "name"]:
                    if attr_key in node_attrs:
                        material_name = str(node_attrs[attr_key])
                        break
                
                material_candidates.append({
                    "node_id": node_id,
                    "node_data": dict(node_attrs),
                    "material_name": material_name or node_id,
                    "has_material_name": material_name is not None
                })
        
        # Sort by whether they have material_name first, then alphabetically by node_id
        # Apply max_candidate_nodes AFTER sorting so the cap is deterministic, not set-order dependent
        material_candidates.sort(key=lambda x: (not x["has_material_name"], x["node_id"]))
        material_candidates = material_candidates[:self.max_candidate_nodes]  # processing budget cap
        material_classes = [{"node_id": mc["node_id"], "node_data": mc["node_data"]} 
                           for mc in material_candidates[:self.max_material_classes]]
        
        # Step 4: Generate KG insights summary
        kg_insights = {
            "num_properties_mapped": len(property_mappings),
            "num_property_nodes": len(set(all_property_nodes)),
            "num_material_nodes": len(material_node_set),
            "num_paths_found": len(found_paths),
            "connection_algorithm": self.algorithm
        }
        
        summary = {
            "num_properties": len(required_properties),
            "num_property_mappings": len(property_mappings),
            "num_material_classes": len(material_classes),
            "connections_found": len(found_paths) > 0
        }
        
        result = {
            "property_mappings": property_mappings,
            "material_classes": material_classes,
            "kg_insights": kg_insights,
            "matched_nodes": list(material_node_set),
            "summary": summary,
            "found_paths": found_paths  # Include paths for use in System 2
        }
        
        # Log KG query if chat_logger is provided
        if self.chat_logger is not None:
            try:
                # Extract keywords from properties for logging
                keywords = required_properties.copy()
                if application_Y:
                    keywords.append(application_Y)
                
                self.chat_logger.log_kg_query(
                    agent_name="research_scientist",
                    method_name="map_properties_to_materials",
                    keywords=keywords,
                    result=result
                )
            except Exception as e:
                print(f"Warning: Failed to log KG query: {e}")
        
        return result

    def select_paths_for_proposal(
        self,
        subgraph: nx.DiGraph,
        properties_W: Dict[str, Any],
        application_Y: str,
        material_classes: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Select scored, diverse KG paths from the material-informed subgraph for use in
        the propose_candidate prompt.

        Replaces the pre-baked, arbitrarily-ordered all-pairs paths that were computed
        in Step 1.  Instead, this method:

        1. Uses top-N (n_samples) anchor nodes per required property (not just best-match).
        2. Computes two families of shortest paths in the undirected subgraph:
           - Type A: property-anchor → material-class node  (every pair)
           - Type B: cross-property anchor pairs            (anchor_i × anchor_j, i≠j property)
        3. Scores each path:
               material_in_path  = 1.0 if any node in the path is a material-class node
               property_coverage = |distinct properties whose anchors appear in path| / total_properties
               score = 2 * material_in_path + property_coverage
           (path length is intentionally not penalised)
        4. Greedy diversity selection: paths are picked in score-descending order; a path
           is skipped when its Jaccard node-overlap with any already-selected path exceeds
           diversity_threshold (from config).  Continues until max_paths paths are collected.

        Args:
            subgraph:        Material-informed subgraph produced by Step 1.
            properties_W:    Property requirements dict (must contain "required" list).
            application_Y:   Target application string (used for logging/debug only).
            material_classes: List of material-class dicts from map_properties_to_materials(),
                              each with at least a "node_id" key.

        Returns:
            List of path dicts in the same format as found_paths elsewhere in the pipeline:
            {"source", "target", "path", "length", "nodes", "edges", "score", "path_type"}
        """
        # ------------------------------------------------------------------ #
        # Load config                                                          #
        # ------------------------------------------------------------------ #
        config = load_config()
        agent_config = config.get("agents", {}).get("research_scientist", {})
        path_finding_config = agent_config.get("path_finding", {})
        max_paths = config.get("agents", {}).get("research_manager", {}).get(
            "formatting", {}
        ).get("max_paths", 100)
        diversity_threshold = path_finding_config.get("diversity_threshold", 0.5)

        required_properties = properties_W.get("required", [])
        if not required_properties or subgraph is None or subgraph.number_of_nodes() == 0:
            return []

        # ------------------------------------------------------------------ #
        # Step 1: Build per-property anchor node sets                         #
        # ------------------------------------------------------------------ #
        # Filter embeddings to nodes present in the subgraph for speed.
        subgraph_nodes = set(subgraph.nodes())
        subgraph_embeddings = {
            nid: emb
            for nid, emb in self.node_embeddings.items()
            if nid in subgraph_nodes
        }

        # property_anchors: {property_name: [node_id, ...]}
        property_anchors: Dict[str, List[str]] = {}
        for prop in required_properties:
            if not isinstance(prop, str) or not prop.strip():
                continue
            matched = find_best_fitting_node_list(
                prop,
                subgraph_embeddings,
                self.embedding_tokenizer,
                self.embedding_model,
                N_samples=self.n_samples,
                similarity_threshold=self.similarity_threshold,
            )
            # Keep all matches (up to n_samples), not just the best one
            property_anchors[prop] = [nid for nid, _ in matched if nid in subgraph_nodes]

        all_anchor_nodes = {nid for nodes in property_anchors.values() for nid in nodes}

        # ------------------------------------------------------------------ #
        # Step 2: Build material node set from material_classes               #
        # ------------------------------------------------------------------ #
        material_node_ids: set = {
            mc["node_id"]
            for mc in material_classes
            if mc.get("node_id") in subgraph_nodes
        }

        total_properties = len(required_properties)
        undirected = subgraph.to_undirected()

        # Helper: extract path metadata (nodes + edges with attributes)
        def _path_meta(path: List[str]) -> Dict[str, Any]:
            fallback = self.knowledge_graph if subgraph is not self.knowledge_graph else None
            return self._extract_path_metadata(path, subgraph, fallback_graph=fallback)

        # Helper: compute property-coverage for a path
        def _property_coverage(path_set: set) -> float:
            if total_properties == 0:
                return 0.0
            covered = sum(
                1
                for prop, anchors in property_anchors.items()
                if any(a in path_set for a in anchors)
            )
            return covered / total_properties

        # ------------------------------------------------------------------ #
        # Step 3: Compute candidate paths                                     #
        # ------------------------------------------------------------------ #
        candidate_paths: List[Dict[str, Any]] = []

        # --- Type A: property anchor → material class node ---
        if material_node_ids:
            for prop, anchors in property_anchors.items():
                for anchor in anchors:
                    for mat_node in material_node_ids:
                        if anchor == mat_node:
                            continue
                        try:
                            path = nx.shortest_path(undirected, anchor, mat_node)
                            meta = _path_meta(path)
                            candidate_paths.append({
                                "source": anchor,
                                "target": mat_node,
                                "path": path,
                                "length": len(path) - 1,
                                "nodes": meta["nodes"],
                                "edges": meta["edges"],
                                "path_type": "A",
                                "_source_prop": prop,
                            })
                        except (nx.NetworkXNoPath, nx.NodeNotFound):
                            pass

        # --- Type B: cross-property anchor pairs ---
        prop_list = list(property_anchors.items())
        for i in range(len(prop_list)):
            for j in range(i + 1, len(prop_list)):
                prop_i, anchors_i = prop_list[i]
                prop_j, anchors_j = prop_list[j]
                for anchor_i in anchors_i:
                    for anchor_j in anchors_j:
                        if anchor_i == anchor_j:
                            continue
                        try:
                            path = nx.shortest_path(undirected, anchor_i, anchor_j)
                            meta = _path_meta(path)
                            candidate_paths.append({
                                "source": anchor_i,
                                "target": anchor_j,
                                "path": path,
                                "length": len(path) - 1,
                                "nodes": meta["nodes"],
                                "edges": meta["edges"],
                                "path_type": "B",
                                "_source_prop": prop_i,
                            })
                        except (nx.NetworkXNoPath, nx.NodeNotFound):
                            pass

        if not candidate_paths:
            return []

        # ------------------------------------------------------------------ #
        # Step 4: Score every path                                            #
        # ------------------------------------------------------------------ #
        for p in candidate_paths:
            path_node_set = set(p["path"])
            material_in_path = 1.0 if path_node_set & material_node_ids else 0.0
            prop_cov = _property_coverage(path_node_set)
            p["score"] = self.path_scoring_weight * material_in_path + prop_cov

        # Sort descending by score
        candidate_paths.sort(key=lambda x: x["score"], reverse=True)

        # ------------------------------------------------------------------ #
        # Step 5: Greedy diversity selection                                  #
        # ------------------------------------------------------------------ #
        selected: List[Dict[str, Any]] = []
        selected_node_sets: List[set] = []

        for p in candidate_paths:
            if len(selected) >= max_paths:
                break
            path_node_set = set(p["path"])
            # Check overlap against every already-selected path
            too_similar = False
            for sel_nodes in selected_node_sets:
                union = path_node_set | sel_nodes
                if not union:
                    continue
                overlap = len(path_node_set & sel_nodes) / len(union)
                if overlap > diversity_threshold:
                    too_similar = True
                    break
            if not too_similar:
                selected.append(p)
                selected_node_sets.append(path_node_set)

        print(
            f"  [select_paths_for_proposal] {len(candidate_paths)} candidate paths → "
            f"{len(selected)} selected (max_paths={max_paths}, "
            f"diversity_threshold={diversity_threshold})"
        )
        type_a = sum(1 for p in selected if p.get("path_type") == "A")
        type_b = len(selected) - type_a
        print(f"    Type A (property→material): {type_a},  Type B (property→property): {type_b}")

        # Remove internal-only keys before returning
        for p in selected:
            p.pop("_source_prop", None)

        return selected

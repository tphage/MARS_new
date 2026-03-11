"""Subgraph Processor - Filters and processes knowledge graph subgraphs"""

import networkx as nx
import numpy as np
from typing import List, Dict, Any, Optional, Set
from sentence_transformers import SentenceTransformer


class SubgraphProcessor:
    """
    Processes and filters knowledge graph subgraphs to retain only
    information relevant to application, properties, and constraints.
    """
    
    def __init__(
        self,
        embedding_model: SentenceTransformer,
        similarity_threshold: float = None,
        max_nodes: int = None,
        max_edges: int = None
    ):
        """
        Initialize the subgraph processor.

        Args:
            embedding_model: SentenceTransformer model for semantic similarity
            similarity_threshold: Minimum similarity threshold (default: None, uses config utils.subgraph_processor)
            max_nodes: Maximum number of nodes to retain (default: None, uses config)
            max_edges: Maximum number of edges to retain (default: None, uses config)
        """
        if similarity_threshold is None or max_nodes is None or max_edges is None:
            try:
                from ..config import load_config
                utils_cfg = load_config().get("utils", {}).get("subgraph_processor", {})
                if similarity_threshold is None:
                    similarity_threshold = utils_cfg.get("similarity_threshold", 0.8)
                if max_nodes is None:
                    max_nodes = utils_cfg.get("max_nodes", 1000)
                if max_edges is None:
                    max_edges = utils_cfg.get("max_edges", 2000)
            except Exception:
                similarity_threshold = similarity_threshold or 0.8
                max_nodes = max_nodes or 1000
                max_edges = max_edges or 2000
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.max_nodes = max_nodes
        self.max_edges = max_edges
    
    def load_subgraph(self, subgraph_data: Dict[str, Any]) -> Optional[nx.DiGraph]:
        """
        Load a subgraph from System 1 output data.
        
        Args:
            subgraph_data: Dictionary containing subgraph data from System 1 output
                Expected keys:
                - connection_subgraph: Dict with nodes, edges, node_attributes, edge_attributes
                - matched_node_ids: List of matched node IDs
                - found_paths: List of paths between nodes
                - kg_results: Dict with results from multiple KGs (for dual-KG mode)
                
        Returns:
            NetworkX DiGraph if subgraph data is available, None otherwise
        """
        if not subgraph_data:
            return None
        
        # Try to load from connection_subgraph first (preferred method)
        connection_subgraph_data = subgraph_data.get("connection_subgraph")
        
        # Check if connection_subgraph is a valid dict (not a string representation)
        if connection_subgraph_data and isinstance(connection_subgraph_data, dict):
            # Reconstruct NetworkX graph from serialized data
            G = nx.DiGraph()
            
            # Add nodes with attributes
            nodes = connection_subgraph_data.get("nodes", [])
            node_attributes = connection_subgraph_data.get("node_attributes", {})
            
            for node in nodes:
                attrs = node_attributes.get(node, {})
                G.add_node(node, **attrs)
            
            # Add edges with attributes
            edges = connection_subgraph_data.get("edges", [])
            edge_attributes = connection_subgraph_data.get("edge_attributes", {})
            
            for edge in edges:
                if isinstance(edge, (list, tuple)) and len(edge) >= 2:
                    u, v = edge[0], edge[1]
                    # Edge attributes may be keyed by tuple or string representation
                    attrs = edge_attributes.get((u, v), {})
                    if not attrs:
                        # Fallback: JSON serialization converts tuple keys to strings
                        attrs = edge_attributes.get(str((u, v)), {})
                        if not attrs:
                            attrs = edge_attributes.get(f"{u},{v}", {})
                    G.add_edge(u, v, **attrs)
            
            if G.number_of_nodes() > 0:
                return G
        
        # Fallback: Reconstruct subgraph from found_paths and matched_node_ids
        # This handles cases where connection_subgraph wasn't properly serialized
        # (e.g., when using dual-KG mode with "separate" strategy)
        G = nx.DiGraph()
        
        # Collect all nodes from matched_node_ids
        matched_node_ids = subgraph_data.get("matched_node_ids", [])
        for node_id in matched_node_ids:
            if node_id:
                G.add_node(node_id)
        
        # Collect nodes and edges from found_paths
        found_paths = subgraph_data.get("found_paths", [])
        for path_info in found_paths:
            if isinstance(path_info, dict):
                path = path_info.get("path", [])
            elif isinstance(path_info, (list, tuple)):
                path = path_info
            else:
                continue
            
            if not path or len(path) < 2:
                continue
            
            # Add all nodes in the path
            for node in path:
                if node:
                    G.add_node(node)
            
            # Add edges along the path
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                if u and v:
                    G.add_edge(u, v)
        
        # Also check kg_results for dual-KG mode
        kg_results = subgraph_data.get("kg_results", {})
        if kg_results and isinstance(kg_results, dict):
            for kg_name, kg_result in kg_results.items():
                if not isinstance(kg_result, dict):
                    continue
                
                # Get result data (might be nested)
                result_data = kg_result.get("result", kg_result)
                
                # Collect matched nodes
                kg_matched_nodes = result_data.get("matched_node_ids", [])
                for node_id in kg_matched_nodes:
                    if node_id:
                        G.add_node(node_id)
                
                # Collect paths
                kg_paths = result_data.get("found_paths", [])
                for path_info in kg_paths:
                    if isinstance(path_info, dict):
                        path = path_info.get("path", [])
                    elif isinstance(path_info, (list, tuple)):
                        path = path_info
                    else:
                        continue
                    
                    if not path or len(path) < 2:
                        continue
                    
                    # Add all nodes in the path
                    for node in path:
                        if node:
                            G.add_node(node)
                    
                    # Add edges along the path
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i + 1]
                        if u and v:
                            G.add_edge(u, v)
        
        # Return graph if we have any nodes, otherwise None
        if G.number_of_nodes() > 0:
            return G
        
        return None
    
    @staticmethod
    def _get_node_text(subgraph: nx.DiGraph, node) -> str:
        """Return the best available text representation for *node*."""
        node_attrs = subgraph.nodes[node]
        for attr_key in ("title", "label", "name", "description"):
            if attr_key in node_attrs:
                return str(node_attrs[attr_key])
        return str(node)

    def filter_by_relevance(
        self,
        subgraph: nx.DiGraph,
        application_Y: str,
        properties_W: Dict[str, Any],
        constraints_U: List[str]
    ) -> nx.DiGraph:
        """
        Filter subgraph to retain only nodes/edges relevant to application, properties, and constraints.
        
        Args:
            subgraph: NetworkX graph to filter
            application_Y: Target application description
            properties_W: Dictionary with property requirements
            constraints_U: List of constraint strings
            
        Returns:
            Filtered NetworkX subgraph
        """
        if subgraph is None or subgraph.number_of_nodes() == 0:
            return nx.DiGraph()
        
        # Extract key terms for relevance matching
        required_properties = properties_W.get("required", [])
        all_keywords = [application_Y] + required_properties + constraints_U
        
        # Generate embeddings for keywords
        keyword_embeddings = self.embedding_model.encode(all_keywords, convert_to_numpy=True)
        
        # --- Batch-encode all node texts at once (major speedup) ---
        node_list = list(subgraph.nodes())
        node_texts = [self._get_node_text(subgraph, n) for n in node_list]
        node_embeddings_matrix = self.embedding_model.encode(node_texts, convert_to_numpy=True)

        # Score nodes by relevance
        node_scores = {}
        relevant_nodes = set()
        
        for idx, node in enumerate(node_list):
            node_embedding = node_embeddings_matrix[idx]
            
            max_similarity = 0.0
            for keyword_emb in keyword_embeddings:
                norm_product = np.linalg.norm(node_embedding) * np.linalg.norm(keyword_emb)
                if norm_product == 0:
                    continue
                similarity = np.dot(node_embedding, keyword_emb) / norm_product
                max_similarity = max(max_similarity, similarity)
            
            node_scores[node] = max_similarity
            
            if max_similarity >= self.similarity_threshold:
                relevant_nodes.add(node)
        
        # Also include nodes from matched_node_ids and found_paths if available
        # (These are already known to be relevant)
        # This would be passed separately if needed
        
        # Include nodes connected to relevant nodes (neighbors)
        extended_nodes = set(relevant_nodes)
        for node in relevant_nodes:
            extended_nodes.update(subgraph.predecessors(node))
            extended_nodes.update(subgraph.successors(node))
        
        # If too many nodes, keep top-scoring ones
        if len(extended_nodes) > self.max_nodes:
            scored_extended = [(n, node_scores.get(n, 0.0)) for n in extended_nodes]
            scored_extended.sort(key=lambda x: x[1], reverse=True)
            extended_nodes = {n for n, _ in scored_extended[:self.max_nodes]}
        
        # Create filtered subgraph
        filtered_subgraph = subgraph.subgraph(list(extended_nodes)).copy()
        
        # Limit edges if needed
        if filtered_subgraph.number_of_edges() > self.max_edges:
            # Keep edges connecting highest-scoring nodes
            edge_scores = []
            for u, v in filtered_subgraph.edges():
                score = node_scores.get(u, 0.0) + node_scores.get(v, 0.0)
                edge_scores.append(((u, v), score))
            
            edge_scores.sort(key=lambda x: x[1], reverse=True)
            top_edges = [edge for edge, _ in edge_scores[:self.max_edges]]
            
            filtered_subgraph = nx.DiGraph()
            for node in extended_nodes:
                filtered_subgraph.add_node(node, **subgraph.nodes[node])
            for u, v in top_edges:
                if u in filtered_subgraph and v in filtered_subgraph:
                    filtered_subgraph.add_edge(u, v, **subgraph[u][v])
        
        return filtered_subgraph
    
    def extract_key_nodes(self, subgraph: nx.DiGraph, properties_W: Dict[str, Any]) -> Set[str]:
        """
        Extract nodes related to required properties.
        
        Args:
            subgraph: NetworkX graph to search
            properties_W: Dictionary with property requirements
            
        Returns:
            Set of node IDs related to properties
        """
        if subgraph is None:
            return set()
        
        required_properties = properties_W.get("required", [])
        property_nodes = set()
        
        # Generate embeddings for properties
        if required_properties:
            property_embeddings = self.embedding_model.encode(required_properties, convert_to_numpy=True)
            
            # Batch-encode all nodes at once
            node_list = list(subgraph.nodes())
            node_texts = [self._get_node_text(subgraph, n) for n in node_list]
            node_embeddings_matrix = self.embedding_model.encode(node_texts, convert_to_numpy=True)
            
            for idx, node in enumerate(node_list):
                node_embedding = node_embeddings_matrix[idx]
                for prop_emb in property_embeddings:
                    norm_product = np.linalg.norm(node_embedding) * np.linalg.norm(prop_emb)
                    if norm_product == 0:
                        continue
                    similarity = np.dot(node_embedding, prop_emb) / norm_product
                    if similarity >= self.similarity_threshold:
                        property_nodes.add(node)
                        break
        
        return property_nodes
    
    def extract_application_nodes(self, subgraph: nx.DiGraph, application_Y: str) -> Set[str]:
        """
        Extract nodes related to the target application.
        
        Args:
            subgraph: NetworkX graph to search
            application_Y: Target application description
            
        Returns:
            Set of node IDs related to the application
        """
        if subgraph is None:
            return set()
        
        application_nodes = set()
        app_embedding = self.embedding_model.encode([application_Y], convert_to_numpy=True)[0]
        
        # Batch-encode all nodes at once
        node_list = list(subgraph.nodes())
        node_texts = [self._get_node_text(subgraph, n) for n in node_list]
        node_embeddings_matrix = self.embedding_model.encode(node_texts, convert_to_numpy=True)
        
        for idx, node in enumerate(node_list):
            node_embedding = node_embeddings_matrix[idx]
            norm_product = np.linalg.norm(node_embedding) * np.linalg.norm(app_embedding)
            if norm_product == 0:
                continue
            similarity = np.dot(node_embedding, app_embedding) / norm_product
            
            if similarity >= self.similarity_threshold:
                application_nodes.add(node)
        
        return application_nodes
    
    def get_relevant_subgraph(
        self,
        subgraph: nx.DiGraph,
        application_Y: str,
        properties_W: Dict[str, Any],
        constraints_U: List[str]
    ) -> nx.DiGraph:
        """
        Get a filtered subgraph containing only relevant nodes and edges.
        This is a convenience method that combines filtering operations.
        
        Args:
            subgraph: NetworkX graph to filter
            application_Y: Target application description
            properties_W: Dictionary with property requirements
            constraints_U: List of constraint strings
            
        Returns:
            Filtered NetworkX subgraph
        """
        return self.filter_by_relevance(subgraph, application_Y, properties_W, constraints_U)
    
    def prune_irrelevant_edges(self, subgraph: nx.DiGraph, relevant_nodes: Set[str]) -> nx.DiGraph:
        """
        Remove edges that don't connect relevant nodes.
        
        Args:
            subgraph: NetworkX graph to prune
            relevant_nodes: Set of node IDs to keep
            
        Returns:
            Pruned NetworkX subgraph
        """
        if subgraph is None:
            return nx.DiGraph()
        
        pruned = subgraph.subgraph(list(relevant_nodes)).copy()
        return pruned

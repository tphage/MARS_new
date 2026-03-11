"""Material Grounding - Maps lab materials to knowledge graph nodes"""

import networkx as nx
import numpy as np
from typing import List, Dict, Any, Optional, Set
from GraphReasoning import find_best_fitting_node_list
from ..config import load_config


class MaterialGrounding:
    """
    Grounds lab materials in the global knowledge graph by finding matching nodes
    and retrieving relevant relationships.
    """
    
    def __init__(
        self,
        knowledge_graph: nx.DiGraph,
        node_embeddings: Dict[str, np.ndarray],
        embedding_model,
        embedding_tokenizer,
        n_samples: int = None,
        similarity_threshold: float = None
    ):
        """
        Initialize the material grounding module.
        
        Args:
            knowledge_graph: Global knowledge graph (NetworkX DiGraph)
            node_embeddings: Dictionary mapping node IDs to embedding vectors
            embedding_model: Model for generating embeddings
            embedding_tokenizer: Tokenizer for generating embeddings
            n_samples: Number of top matching nodes to retrieve per material (default: None, uses config value)
            similarity_threshold: Minimum similarity threshold for matching (default: None, uses config value)
        """
        # Load config
        config = load_config()
        agent_config = config.get("agents", {}).get("material_grounding", {})
        
        # Use config defaults if not provided
        if n_samples is None:
            n_samples = agent_config.get("n_samples", 5)
        if similarity_threshold is None:
            similarity_threshold = agent_config.get("similarity_threshold", 0.8)
        
        self.knowledge_graph = knowledge_graph
        self.node_embeddings = node_embeddings
        self.embedding_model = embedding_model
        self.embedding_tokenizer = embedding_tokenizer
        self.n_samples = n_samples
        self.similarity_threshold = similarity_threshold
    
    def ground_material(
        self,
        material_name: str,
        material_class: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Find matching nodes in the global KG for a given material.
        
        Args:
            material_name: Name of the material to ground
            material_class: Optional material class/family (deprecated, not used)
            
        Returns:
            List of dictionaries containing:
                - node_id: Matched node ID
                - similarity: Similarity score
                - node_data: Node attributes from KG
        """
        # Use material name as search query
        search_query = material_name
        
        # Find best matching nodes
        matched_nodes = find_best_fitting_node_list(
            search_query,
            self.node_embeddings,
            self.embedding_tokenizer,
            self.embedding_model,
            N_samples=self.n_samples,
            similarity_threshold=self.similarity_threshold
        )
        
        # Format results
        formatted_matches = []
        for node_id, similarity in matched_nodes:
            node_info = {
                "node_id": node_id,
                "similarity": float(similarity),
                "material_name": material_name
            }
            
            # Add node attributes if available
            if self.knowledge_graph is not None and node_id in self.knowledge_graph.nodes():
                node_attrs = self.knowledge_graph.nodes[node_id]
                node_info["node_data"] = dict(node_attrs)
            else:
                node_info["node_data"] = {}
            
            formatted_matches.append(node_info)
        
        return formatted_matches
    
    def ground_material_database(self, material_db) -> Dict[str, List[Dict[str, Any]]]:
        """
        Ground all materials in a material database.
        
        Args:
            material_db: MaterialDatabase instance
            
        Returns:
            Dictionary mapping material_id -> list of matched KG node dictionaries
        """
        grounding_map = {}
        
        for material in material_db.get_all_materials():
            material_id = material["material_id"]
            material_name = material["material_name"]
            
            matched_nodes = self.ground_material(material_name)
            grounding_map[material_id] = matched_nodes
        
        return grounding_map
    
    def retrieve_material_relationships(
        self,
        material_node_ids: List[str],
        property_node_ids: Optional[List[str]] = None
    ) -> nx.DiGraph:
        """
        Retrieve all relationships connecting material nodes to property nodes.
        
        Args:
            material_node_ids: List of material node IDs in the KG
            property_node_ids: Optional list of property node IDs to focus on
            
        Returns:
            NetworkX subgraph containing material-property relationships
        """
        if not material_node_ids:
            return nx.DiGraph()
        
        # Find all edges connecting material nodes
        relationship_edges = []
        material_node_set = set(material_node_ids)
        
        if property_node_ids:
            property_node_set = set(property_node_ids)
        else:
            property_node_set = None
        
        # Traverse graph to find relationships
        for material_node in material_node_ids:
            if material_node not in self.knowledge_graph:
                continue
            
            # Get all neighbors (both predecessors and successors)
            neighbors = list(self.knowledge_graph.predecessors(material_node))
            neighbors.extend(self.knowledge_graph.successors(material_node))
            
            for neighbor in neighbors:
                # If property nodes specified, only include those
                if property_node_set and neighbor not in property_node_set:
                    continue
                
                # Add edge if it exists
                if self.knowledge_graph.has_edge(material_node, neighbor):
                    relationship_edges.append((material_node, neighbor))
                elif self.knowledge_graph.has_edge(neighbor, material_node):
                    relationship_edges.append((neighbor, material_node))
        
        # Create subgraph with all nodes and edges
        all_nodes = set(material_node_ids)
        for u, v in relationship_edges:
            all_nodes.add(u)
            all_nodes.add(v)
        
        if property_node_ids:
            all_nodes.update(property_node_ids)
        
        relationship_subgraph = self.knowledge_graph.subgraph(list(all_nodes)).copy()
        
        return relationship_subgraph
    
    def find_material_nodes(
        self,
        material_name: str,
        material_class: Optional[str] = None
    ) -> List[str]:
        """
        Find material nodes in the KG (simplified version returning just node IDs).
        
        Args:
            material_name: Name of the material
            material_class: Optional material class (deprecated, not used)
            
        Returns:
            List of matched node IDs
        """
        matched_nodes = self.ground_material(material_name)
        return [node["node_id"] for node in matched_nodes]
    
    def get_property_relationships(
        self,
        material_node_ids: List[str],
        property_keywords: List[str]
    ) -> nx.DiGraph:
        """
        Extract material-property edges from the KG.
        
        Args:
            material_node_ids: List of material node IDs
            property_keywords: List of property keywords to search for
            
        Returns:
            Subgraph containing material-property relationships
        """
        # Find property nodes
        property_node_ids = []
        for prop_keyword in property_keywords:
            matched_nodes = find_best_fitting_node_list(
                prop_keyword,
                self.node_embeddings,
                self.embedding_tokenizer,
                self.embedding_model,
                N_samples=self.n_samples,
                similarity_threshold=self.similarity_threshold
            )
            property_node_ids.extend([node_id for node_id, _ in matched_nodes])
        
        # Retrieve relationships
        return self.retrieve_material_relationships(material_node_ids, property_node_ids)
    
    def merge_into_subgraph(
        self,
        base_subgraph: nx.DiGraph,
        material_relationships: nx.DiGraph
    ) -> nx.DiGraph:
        """
        Merge material relationships into a filtered subgraph.
        
        Args:
            base_subgraph: Base filtered subgraph
            material_relationships: Subgraph containing material relationships
            
        Returns:
            Merged NetworkX subgraph
        """
        # Create a copy of base subgraph
        merged = base_subgraph.copy()
        
        # Add nodes and edges from material relationships
        for node in material_relationships.nodes():
            if node not in merged:
                node_attrs = material_relationships.nodes[node]
                merged.add_node(node, **node_attrs)
            else:
                # Merge attributes
                base_attrs = merged.nodes[node]
                new_attrs = material_relationships.nodes[node]
                base_attrs.update(new_attrs)
        
        for u, v in material_relationships.edges():
            if not merged.has_edge(u, v):
                edge_attrs = material_relationships[u][v]
                merged.add_edge(u, v, **edge_attrs)
        
        return merged

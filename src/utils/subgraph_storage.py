"""Subgraph Storage - Persistent storage for knowledge graph subgraphs"""

import json
import logging
import networkx as nx
from typing import Optional, List, Dict
from pathlib import Path
import os

logger = logging.getLogger(__name__)


def _get_pipeline_logs_dir() -> str:
    """Return the pipeline_logs_dir from config, with a sensible fallback."""
    try:
        from ..config import load_config
        return load_config().get("logging", {}).get("pipeline_logs_dir", "./pipeline_logs")
    except (ImportError, FileNotFoundError, KeyError):
        return "./pipeline_logs"


def _find_project_root() -> Path:
    """
    Find the project root by looking for project markers (src/ + config/).
    
    Returns:
        Path to project root
    """
    # First try: package location (most reliable when imported as part of src/)
    try:
        pkg_root = Path(__file__).parent.parent.parent
        if (pkg_root / "src").exists() and (pkg_root / "config").exists():
            return pkg_root
    except Exception:
        pass

    current_dir = Path.cwd().resolve()

    # Check current directory
    if (current_dir / "src").exists() and (current_dir / "config").exists():
        return current_dir
    
    # Walk up the directory tree
    for parent in current_dir.parents:
        if (parent / "src").exists() and (parent / "config").exists():
            return parent
    
    # Last resort: use current directory
    return current_dir


class SubgraphStorage:
    """
    Handles persistent storage of knowledge graph subgraphs to disk.
    
    Subgraphs are saved as JSON files using NetworkX node-link format,
    allowing them to be loaded back as NetworkX DiGraph objects.
    """
    
    def __init__(self, storage_dir: Optional[str] = None) -> None:
        """
        Initialize the subgraph storage.
        
        Args:
            storage_dir: Directory where subgraphs will be stored. If None, uses
                        config logging.pipeline_logs_dir + /subgraphs/
        """
        if storage_dir is None:
            project_root = _find_project_root()
            logs_dir = _get_pipeline_logs_dir()
            if logs_dir.startswith("./"):
                logs_dir = logs_dir[2:]
            storage_dir = str(project_root / logs_dir / "subgraphs")
        else:
            # If relative path provided, resolve it relative to project root
            if not os.path.isabs(storage_dir):
                project_root = _find_project_root()
                if storage_dir.startswith("./"):
                    storage_dir = storage_dir[2:]
                storage_dir = str(project_root / storage_dir)
        
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    def save_subgraph(
        self,
        subgraph: nx.DiGraph,
        run_id: str,
        subgraph_type: str = "material_informed"
    ) -> Optional[str]:
        """
        Save a subgraph to disk.
        
        Args:
            subgraph: NetworkX DiGraph to save
            run_id: Unique identifier for this run (e.g., timestamp-based ID)
            subgraph_type: Type of subgraph (default: "material_informed")
            
        Returns:
            Path to the saved file, or None if the subgraph was empty
        """
        if subgraph is None or subgraph.number_of_nodes() == 0:
            logger.warning("Attempting to save empty subgraph, skipping")
            return None
        
        # Serialize subgraph to node-link format
        graph_data = nx.node_link_data(subgraph)
        
        # Create filename
        filename = f"{run_id}_{subgraph_type}.json"
        filepath = self.storage_dir / filename
        
        # Save to JSON
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    "_type": "networkx_graph",
                    "data": graph_data,
                    "run_id": run_id,
                    "subgraph_type": subgraph_type,
                    "num_nodes": subgraph.number_of_nodes(),
                    "num_edges": subgraph.number_of_edges()
                }, f, indent=2, default=str)
            
            logger.info("Saved subgraph to %s (%d nodes, %d edges)",
                        filepath, subgraph.number_of_nodes(), subgraph.number_of_edges())
            return str(filepath)
        except (IOError, OSError, TypeError) as e:
            logger.warning("Failed to save subgraph to %s: %s", filepath, e)
            return None
    
    def load_subgraph(
        self,
        run_id: str,
        subgraph_type: str = "material_informed"
    ) -> Optional[nx.DiGraph]:
        """
        Load a subgraph from disk.
        
        Args:
            run_id: Unique identifier for the run
            subgraph_type: Type of subgraph (default: "material_informed")
            
        Returns:
            NetworkX DiGraph if found, None otherwise
        """
        filename = f"{run_id}_{subgraph_type}.json"
        filepath = self.storage_dir / filename
        
        if not filepath.exists():
            return None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if it's a valid subgraph file
            if not isinstance(data, dict) or data.get("_type") != "networkx_graph":
                logger.warning("Invalid subgraph file format: %s", filepath)
                return None
            
            # Deserialize from node-link format
            graph_data = data.get("data", {})
            subgraph = nx.node_link_graph(graph_data, directed=True)
            
            logger.info("Loaded subgraph from %s (%d nodes, %d edges)",
                        filepath, subgraph.number_of_nodes(), subgraph.number_of_edges())
            return subgraph
        except (IOError, OSError, json.JSONDecodeError, TypeError) as e:
            logger.warning("Failed to load subgraph from %s: %s", filepath, e)
            return None
    
    def list_subgraphs(self) -> List[Dict[str, str]]:
        """
        List all available persisted subgraphs.
        
        Returns:
            List of dictionaries with information about each subgraph:
            - run_id: Run identifier
            - subgraph_type: Type of subgraph
            - filepath: Path to the file
            - num_nodes: Number of nodes (if available)
            - num_edges: Number of edges (if available)
        """
        subgraphs = []
        
        for filepath in self.storage_dir.glob("*_*.json"):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, dict) and data.get("_type") == "networkx_graph":
                    subgraphs.append({
                        "run_id": data.get("run_id", ""),
                        "subgraph_type": data.get("subgraph_type", ""),
                        "filepath": str(filepath),
                        "num_nodes": data.get("num_nodes", 0),
                        "num_edges": data.get("num_edges", 0)
                    })
            except (IOError, json.JSONDecodeError) as e:
                logger.debug("Skipping invalid subgraph file %s: %s", filepath, e)
                continue
        
        return subgraphs
    
    def delete_subgraph(
        self,
        run_id: str,
        subgraph_type: str = "material_informed"
    ) -> bool:
        """
        Delete a persisted subgraph.
        
        Args:
            run_id: Unique identifier for the run
            subgraph_type: Type of subgraph (default: "material_informed")
            
        Returns:
            True if deleted successfully, False otherwise
        """
        filename = f"{run_id}_{subgraph_type}.json"
        filepath = self.storage_dir / filename
        
        if not filepath.exists():
            return False
        
        try:
            filepath.unlink()
            return True
        except (IOError, OSError) as e:
            logger.warning("Failed to delete subgraph %s: %s", filepath, e)
            return False

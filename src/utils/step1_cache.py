"""Step 1 Cache - Caches expensive Step 1 operations for System 2"""

import hashlib
import json
import networkx as nx
from typing import Dict, Any, Optional
from pathlib import Path


class Step1Cache:
    """
    Cache for Step 1 (Material Substitution) results in System 2.
    
    Caches expensive operations that don't change between iterations:
    - Material grounding
    - Material-property relationship retrieval
    - KG material class extraction
    - Subgraph processing
    
    Cache key is based on deterministic inputs:
    - material_X, application_Y, properties_W, constraints_U, subgraph_data, material_db
    """
    
    def __init__(self, enable_persistence: bool = False, cache_dir: Optional[str] = None):
        """
        Initialize the cache.
        
        Args:
            enable_persistence: If True, persist cache to disk (default: False)
            cache_dir: Directory for persistent cache files (default: ./cache/)
        """
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        self.enable_persistence = enable_persistence
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./cache")
        
        if self.enable_persistence:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()
    
    def _generate_cache_key(
        self,
        material_X: str,
        application_Y: str,
        properties_W: Dict[str, Any],
        constraints_U: list,
        subgraph_data: Optional[Dict[str, Any]],
        material_db_path: Optional[str] = None
    ) -> str:
        """
        Generate a hash-based cache key from input parameters.
        
        Args:
            material_X: Material to replace
            application_Y: Application context
            properties_W: Property requirements dict
            constraints_U: List of constraints
            subgraph_data: Subgraph data from System 1
            material_db_path: Path to material database (for hashing)
            
        Returns:
            Hex digest string to use as cache key
        """
        # Create a deterministic representation of inputs
        key_parts = {
            "material_X": material_X,
            "application_Y": application_Y,
            "properties_W": self._normalize_dict(properties_W),
            "constraints_U": sorted(constraints_U) if constraints_U else [],
            "subgraph_data": self._normalize_dict(subgraph_data) if subgraph_data else None,
            "material_db_path": material_db_path
        }
        
        # Convert to JSON string for hashing
        key_json = json.dumps(key_parts, sort_keys=True, default=str)
        
        # Generate hash
        hash_obj = hashlib.sha256(key_json.encode('utf-8'))
        return hash_obj.hexdigest()
    
    def _normalize_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize a dictionary for consistent hashing.
        Sorts nested lists and dicts recursively.
        
        Args:
            d: Dictionary to normalize
            
        Returns:
            Normalized dictionary
        """
        if not isinstance(d, dict):
            return d
        
        normalized = {}
        for key in sorted(d.keys()):
            value = d[key]
            if isinstance(value, dict):
                normalized[key] = self._normalize_dict(value)
            elif isinstance(value, list):
                # Only sort lists of homogeneous, sortable types
                try:
                    normalized[key] = sorted(value) if all(isinstance(x, (str, int, float)) for x in value) else value
                except TypeError:
                    # Mixed types or non-sortable elements — keep original order
                    normalized[key] = value
            else:
                normalized[key] = value
        
        return normalized
    
    def _serialize_substitution_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Serialize substitution result for storage.
        Converts NetworkX graphs to node-link format.
        
        Args:
            result: Substitution result dict
            
        Returns:
            Serialized dict suitable for JSON storage
        """
        serialized = {}
        
        for key, value in result.items():
            if isinstance(value, nx.DiGraph) or isinstance(value, nx.Graph):
                # Convert NetworkX graph to node-link format
                serialized[key] = {
                    "_type": "networkx_graph",
                    "data": nx.node_link_data(value)
                }
            elif isinstance(value, dict):
                serialized[key] = self._serialize_dict(value)
            else:
                serialized[key] = value
        
        return serialized
    
    def _serialize_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively serialize a dictionary, handling NetworkX graphs."""
        serialized = {}
        for key, value in d.items():
            if isinstance(value, nx.DiGraph) or isinstance(value, nx.Graph):
                serialized[key] = {
                    "_type": "networkx_graph",
                    "data": nx.node_link_data(value)
                }
            elif isinstance(value, dict):
                serialized[key] = self._serialize_dict(value)
            else:
                serialized[key] = value
        return serialized
    
    def _deserialize_substitution_result(self, serialized: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deserialize substitution result from storage.
        Converts node-link format back to NetworkX graphs.
        
        Args:
            serialized: Serialized dict from storage
            
        Returns:
            Deserialized dict with NetworkX graphs restored
        """
        deserialized = {}
        
        for key, value in serialized.items():
            if isinstance(value, dict) and value.get("_type") == "networkx_graph":
                # Convert node-link format back to NetworkX graph
                graph_data = value["data"]
                graph = nx.node_link_graph(graph_data, directed=True)
                deserialized[key] = graph
            elif isinstance(value, dict):
                deserialized[key] = self._deserialize_dict(value)
            else:
                deserialized[key] = value
        
        return deserialized
    
    def _deserialize_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively deserialize a dictionary, handling NetworkX graphs."""
        deserialized = {}
        for key, value in d.items():
            if isinstance(value, dict) and value.get("_type") == "networkx_graph":
                graph_data = value["data"]
                graph = nx.node_link_graph(graph_data, directed=True)
                deserialized[key] = graph
            elif isinstance(value, dict):
                deserialized[key] = self._deserialize_dict(value)
            else:
                deserialized[key] = value
        return deserialized
    
    def get(
        self,
        material_X: str,
        application_Y: str,
        properties_W: Dict[str, Any],
        constraints_U: list,
        subgraph_data: Optional[Dict[str, Any]] = None,
        material_db_path: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached Step 1 result if available.
        
        Args:
            material_X: Material to replace
            application_Y: Application context
            properties_W: Property requirements dict
            constraints_U: List of constraints
            subgraph_data: Subgraph data from System 1
            material_db_path: Path to material database
            
        Returns:
            Cached substitution result dict, or None if not found
        """
        cache_key = self._generate_cache_key(
            material_X, application_Y, properties_W, constraints_U, subgraph_data, material_db_path
        )
        
        # Check memory cache first
        if cache_key in self._memory_cache:
            cached_data = self._memory_cache[cache_key]
            return self._deserialize_substitution_result(cached_data)
        
        # Check disk cache if persistence enabled
        if self.enable_persistence:
            cache_file = self.cache_dir / f"step1_cache_{cache_key}.json"
            if cache_file.exists():
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cached_data = json.load(f)
                    # Also store in memory cache for faster access
                    self._memory_cache[cache_key] = cached_data
                    return self._deserialize_substitution_result(cached_data)
                except Exception as e:
                    print(f"Warning: Failed to load cache from disk: {e}")
        
        return None
    
    def set(
        self,
        material_X: str,
        application_Y: str,
        properties_W: Dict[str, Any],
        constraints_U: list,
        substitution_result: Dict[str, Any],
        subgraph_data: Optional[Dict[str, Any]] = None,
        material_db_path: Optional[str] = None
    ):
        """
        Store Step 1 result in cache.
        
        Args:
            material_X: Material to replace
            application_Y: Application context
            properties_W: Property requirements dict
            constraints_U: List of constraints
            substitution_result: Result from run_material_substitution_step()
            subgraph_data: Subgraph data from System 1
            material_db_path: Path to material database
        """
        cache_key = self._generate_cache_key(
            material_X, application_Y, properties_W, constraints_U, subgraph_data, material_db_path
        )
        
        # Serialize result
        serialized = self._serialize_substitution_result(substitution_result)
        
        # Store in memory cache
        self._memory_cache[cache_key] = serialized
        
        # Persist to disk if enabled
        if self.enable_persistence:
            cache_file = self.cache_dir / f"step1_cache_{cache_key}.json"
            try:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(serialized, f, indent=2, default=str)
            except Exception as e:
                print(f"Warning: Failed to persist cache to disk: {e}")
    
    def clear(self):
        """Clear all cached entries (memory and disk)."""
        self._memory_cache.clear()
        
        if self.enable_persistence:
            for cache_file in self.cache_dir.glob("step1_cache_*.json"):
                try:
                    cache_file.unlink()
                except Exception as e:
                    print(f"Warning: Failed to delete cache file {cache_file}: {e}")
    
    def _load_from_disk(self):
        """Load cache entries from disk into memory cache."""
        if not self.cache_dir.exists():
            return
        
        for cache_file in self.cache_dir.glob("step1_cache_*.json"):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                # Extract cache key from filename
                cache_key = cache_file.stem.replace("step1_cache_", "")
                self._memory_cache[cache_key] = cached_data
            except Exception as e:
                print(f"Warning: Failed to load cache file {cache_file}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dict with cache statistics
        """
        stats = {
            "memory_entries": len(self._memory_cache),
            "persistence_enabled": self.enable_persistence
        }
        
        if self.enable_persistence and self.cache_dir.exists():
            disk_files = list(self.cache_dir.glob("step1_cache_*.json"))
            stats["disk_entries"] = len(disk_files)
            # Calculate total cache size
            total_size = sum(f.stat().st_size for f in disk_files)
            stats["total_cache_size_bytes"] = total_size
        else:
            stats["disk_entries"] = 0
            stats["total_cache_size_bytes"] = 0
        
        return stats

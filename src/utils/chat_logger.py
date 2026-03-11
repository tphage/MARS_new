"""Chat Logger for capturing all agent interactions"""

import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path


def _get_logging_config() -> dict:
    """Load logging config from config.yaml."""
    try:
        from ..config import load_config
        return load_config().get("logging", {})
    except Exception:
        return {}

def _find_project_root() -> Path:
    """
    Find project root in a way that's stable even when CWD is notebooks/.

    We primarily trust the package location (src/../..) which corresponds to the
    repository root in this project layout.
    """
    try:
        pkg_root = Path(__file__).resolve().parent.parent.parent
        if (pkg_root / "src").exists() and (pkg_root / "config").exists():
            return pkg_root
    except Exception:
        pass

    # Fallback: walk up from CWD
    current_dir = Path.cwd().resolve()
    if (current_dir / "src").exists() and (current_dir / "config").exists():
        return current_dir
    for parent in current_dir.parents:
        if (parent / "src").exists() and (parent / "config").exists():
            return parent
    return current_dir


def _resolve_log_dir_from_config() -> str:
    """
    Resolve logging directory to an absolute path under the project root.

    Config allows relative paths like "./pipeline_logs" which should always be
    interpreted relative to the project root, not the current working directory.
    """
    cfg = _get_logging_config()
    base = cfg.get("pipeline_logs_dir", "./pipeline_logs") or "./pipeline_logs"
    subdir = cfg.get("chats_subdir", "chats/") or ""

    # Normalize "./" prefix
    if isinstance(base, str) and base.startswith("./"):
        base = base[2:]
    if isinstance(subdir, str) and subdir.startswith("./"):
        subdir = subdir[2:]

    # If base is relative, anchor it at project root
    if isinstance(base, str) and not os.path.isabs(base):
        base = str(_find_project_root() / base)

    return os.path.join(base, subdir) if subdir else base


class ChatLogger:
    """
    Logger for capturing all agent interactions including LLM calls, RAG queries,
    KG queries, and agent-to-agent communications.
    """
    
    def __init__(self, run_id: str, pipeline: str = "unknown", log_dir: str = None):
        """
        Initialize the ChatLogger.
        
        Args:
            run_id: Unique identifier for this run
            pipeline: Name of the pipeline (e.g., "material_requirements", "material_discovery")
            log_dir: Directory to save log files. If None, uses config logging.pipeline_logs_dir + logging.chats_subdir
        """
        self.run_id = run_id
        self.pipeline = pipeline
        if log_dir is None:
            log_dir = _resolve_log_dir_from_config()
        else:
            # Resolve a relative path against project root to avoid CWD surprises
            if isinstance(log_dir, str) and log_dir.startswith("./"):
                log_dir = log_dir[2:]
            if isinstance(log_dir, str) and not os.path.isabs(log_dir):
                log_dir = str(_find_project_root() / log_dir)
        self.log_dir = log_dir
        self.start_time = datetime.utcnow()
        
        # Initialize log structure
        self.logs = {
            "run_id": run_id,
            "pipeline": pipeline,
            "timestamp": self.start_time.isoformat() + "Z",
            "interactions": [],
            "summary": {
                "total_interactions": 0,
                "llm_calls": 0,
                "rag_queries": 0,
                "kg_queries": 0,
                "agent_interactions": 0
            }
        }
        
        # Create log directory if it doesn't exist
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    
    def log_llm_call(
        self,
        agent_name: str,
        method_name: str,
        system_prompt: str,
        user_prompt: str,
        response: str,
        temperature: float = 0,
        model: str = None,
        **kwargs
    ):
        """
        Log an LLM interaction.
        
        Args:
            agent_name: Name of the agent making the call
            method_name: Name of the method making the call
            system_prompt: System prompt sent to LLM
            user_prompt: User prompt sent to LLM
            response: LLM response
            temperature: Temperature used for generation
            model: Model name (optional)
            **kwargs: Additional metadata
        """
        try:
            interaction = {
                "type": "llm",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "agent": agent_name,
                "method": method_name,
                "data": {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "response": response,
                    "temperature": temperature,
                    "model": model,
                    **kwargs
                }
            }
            
            self.logs["interactions"].append(interaction)
            self.logs["summary"]["total_interactions"] += 1
            self.logs["summary"]["llm_calls"] += 1
        except Exception as e:
            # Logging failures should not break the pipeline
            print(f"Warning: Failed to log LLM call: {e}")
    
    def log_rag_query(
        self,
        agent_name: str,
        query: str,
        keywords: List[str] = None,
        results: List[Dict[str, Any]] = None,
        num_results: int = 0,
        **kwargs
    ):
        """
        Log a RAG query and its results.
        
        Args:
            agent_name: Name of the agent making the query
            query: Query string (sentence)
            keywords: Optional list of keywords used for filtering
            results: List of retrieved documents with content, metadata, distances, IDs
            num_results: Number of results retrieved
            **kwargs: Additional metadata
        """
        try:
            # Summarise RAG results to keep chat logs manageable.
            # Full document content is truncated to keep log file sizes down.
            max_chars = _get_logging_config().get("max_content_chars_in_log", 300)
            compact_results = None
            if results and len(results) > 0:
                compact_results = []
                for r in results:
                    compact = {
                        "id": r.get("id"),
                        "source": r.get("source", r.get("metadata", {}).get("source", "")),
                        "distance": r.get("distance"),
                        "content_preview": str(r.get("content", ""))[:max_chars],
                    }
                    compact_results.append(compact)

            interaction = {
                "type": "rag",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "agent": agent_name,
                "method": kwargs.get("method_name", "analyze"),
                "data": {
                    "query": query,
                    "keywords": keywords or [],
                    "num_results": num_results or len(results) if results else 0,
                    "results": compact_results,  # Compacted results (content truncated)
                    **{k: v for k, v in kwargs.items() if k != "method_name"}
                }
            }
            
            self.logs["interactions"].append(interaction)
            self.logs["summary"]["total_interactions"] += 1
            self.logs["summary"]["rag_queries"] += 1
        except Exception as e:
            print(f"Warning: Failed to log RAG query: {e}")
    
    def log_kg_query(
        self,
        agent_name: str,
        method_name: str,
        keywords: List[str] = None,
        result: Dict[str, Any] = None,
        **kwargs
    ):
        """
        Log a knowledge graph query and its results.
        
        Args:
            agent_name: Name of the agent making the query
            method_name: Name of the method (e.g., "find_connections", "map_properties_to_materials")
            keywords: Keywords or properties used in the query
            result: Full result dictionary from KG query
            **kwargs: Additional metadata
        """
        try:
            # Extract key information from result
            summary = result.get("summary", {}) if result else {}
            found_paths = result.get("found_paths", []) if result else []
            
            # Store at most a handful of paths to keep logs reasonable
            max_paths = _get_logging_config().get("max_paths_in_log", 5)
            paths_to_store = found_paths[:max_paths] if found_paths else []

            interaction = {
                "type": "kg",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "agent": agent_name,
                "method": method_name,
                "data": {
                    "keywords": keywords or [],
                    "result_summary": summary,
                    "found_paths_sample": paths_to_store,
                    "total_paths": len(found_paths),
                    "keyword_to_nodes": result.get("keyword_to_nodes", {}) if result else {},
                    "matched_node_ids": result.get("matched_node_ids", []) if result else [],
                    **kwargs
                }
            }
            
            self.logs["interactions"].append(interaction)
            self.logs["summary"]["total_interactions"] += 1
            self.logs["summary"]["kg_queries"] += 1
        except Exception as e:
            print(f"Warning: Failed to log KG query: {e}")
    
    def log_agent_interaction(
        self,
        source_agent: str,
        target_agent: str,
        interaction_type: str,
        data: Dict[str, Any] = None,
        **kwargs
    ):
        """
        Log an agent-to-agent interaction.
        
        Args:
            source_agent: Name of the source agent
            target_agent: Name of the target agent
            interaction_type: Type of interaction (e.g., "data_passed", "result_received")
            data: Data passed between agents (may be summarized if large)
            **kwargs: Additional metadata
        """
        try:
            interaction = {
                "type": "agent_interaction",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "agent": source_agent,
                "target_agent": target_agent,
                "interaction_type": interaction_type,
                "data": data or {},
                **kwargs
            }
            
            self.logs["interactions"].append(interaction)
            self.logs["summary"]["total_interactions"] += 1
            self.logs["summary"]["agent_interactions"] += 1
        except Exception as e:
            print(f"Warning: Failed to log agent interaction: {e}")
    
    def update_run_id(self, new_run_id: str):
        """
        Update the run_id for this logger.
        
        Args:
            new_run_id: New run_id to use
        """
        self.run_id = new_run_id
        self.logs["run_id"] = new_run_id
    
    def get_logs(self) -> Dict[str, Any]:
        """
        Get the current logs dictionary.
        
        Returns:
            Dictionary containing all logged interactions
        """
        return self.logs
    
    def save(self, filename: str = None) -> str:
        """
        Save logs to a JSON file.
        
        Args:
            filename: Optional filename. If None, generates filename with system/pipeline name.
            
        Returns:
            Path to the saved log file
        """
        try:
            if filename is None:
                # Map pipeline names to system numbers for clarity
                pipeline_to_system = {
                    "material_requirements": "system1",
                    "material_discovery": "system2",
                    "manufacturability_assessment": "system3",
                }
                
                # Get system identifier from pipeline name
                system_id = pipeline_to_system.get(self.pipeline, "unknown")
                
                # Generate filename: system1_chat_log_material_requirements_{run_id}.json,
                # system2_chat_log_material_discovery_{run_id}.json,
                # system3_chat_log_manufacturability_assessment_{run_id}.json
                filename = f"{system_id}_chat_log_{self.pipeline}_{self.run_id}.json"
            
            # Ensure filename ends with .json
            if not filename.endswith(".json"):
                filename += ".json"
            
            filepath = os.path.join(self.log_dir, filename)
            
            # Update end time
            end_time = datetime.utcnow()
            self.logs["end_time"] = end_time.isoformat() + "Z"
            # Compute duration from stored datetime objects to avoid parsing issues
            duration = end_time - self.start_time
            self.logs["duration_seconds"] = duration.total_seconds()
            
            with open(filepath, 'w') as f:
                json.dump(self.logs, f, indent=2, default=str)
            
            return filepath
        except Exception as e:
            print(f"Warning: Failed to save chat log: {e}")
            return ""


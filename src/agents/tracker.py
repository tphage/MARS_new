"""RejectedCandidateTracker - Tracks rejected candidates to avoid repetition"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def _get_project_root_and_logging_config() -> Tuple[Path, dict]:
    """Get project root and logging config. Project root from package location; config from config.yaml."""
    project_root = Path(__file__).parent.parent.parent
    try:
        from ..config import load_config
        logging_cfg = load_config().get("logging", {})
    except (ImportError, FileNotFoundError, KeyError):
        logging_cfg = {}
    return project_root, logging_cfg


def _find_project_root() -> Path:
    """
    Find the project root by looking for pipeline_logs directory.
    Walks up from current directory to find the root.
    
    Returns:
        Path to project root
    """
    current_dir = Path.cwd().resolve()
    
    # Check current directory - use config dir name if available
    try:
        from ..config import load_config
        logs_dir = load_config().get("logging", {}).get("pipeline_logs_dir", "./pipeline_logs")
        logs_dir = logs_dir.replace("./", "") if logs_dir.startswith("./") else logs_dir
    except (ImportError, FileNotFoundError, KeyError):
        logs_dir = "pipeline_logs"
    if (current_dir / logs_dir).exists():
        return current_dir
    for parent in current_dir.parents:
        if (parent / logs_dir).exists():
            return parent
    
    # Fallback: assume we're in project root if pipeline_logs doesn't exist yet
    # This handles the case where pipeline_logs hasn't been created
    # We'll look for other project markers
    for parent in current_dir.parents:
        if (parent / "src").exists() and (parent / "config").exists():
            return parent
    
    # Last resort: use current directory
    return current_dir


class RejectedCandidateTracker:
    """
    Tracks rejected material candidates to avoid proposing them again in iterative pipelines.
    Persists data to JSON file for cross-session persistence.
    """
    
    def __init__(self, log_file: Optional[str] = None) -> None:
        """
        Initialize the tracker.
        
        Args:
            log_file: Path to JSON file for persistent storage. If None, automatically
                     finds project root and uses pipeline_logs/rejected_candidates.json
        """
        if log_file is None:
            project_root, logging_cfg = _get_project_root_and_logging_config()
            logs_dir = logging_cfg.get("pipeline_logs_dir", "./pipeline_logs")
            if logs_dir.startswith("./"):
                logs_dir = logs_dir[2:]
            rejected_file = logging_cfg.get("rejected_candidates_file", "rejected_candidates.json")
            log_file = str(project_root / logs_dir / rejected_file)
        else:
            # If relative path provided, resolve it relative to project root
            if not os.path.isabs(log_file):
                project_root = _find_project_root()
                # Handle both "./pipeline_logs/..." and "pipeline_logs/..." formats
                if log_file.startswith("./"):
                    log_file = log_file[2:]
                log_file = str(project_root / log_file)
        
        self.log_file = log_file
        self.rejected_candidates = []
        self._load()
    
    def _load(self) -> None:
        """Load rejected candidates from persistent storage."""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.rejected_candidates = data.get("rejected_candidates", [])
            except (IOError, json.JSONDecodeError, OSError) as e:
                logger.warning("Could not load rejected candidates from %s: %s", self.log_file, e)
                self.rejected_candidates = []
        else:
            # Create directory if it doesn't exist
            log_dir = os.path.dirname(self.log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            self.rejected_candidates = []
    
    def _save(self) -> None:
        """Save rejected candidates to persistent storage."""
        try:
            log_dir = os.path.dirname(self.log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "rejected_candidates": self.rejected_candidates
                }, f, indent=2)
        except (IOError, OSError) as e:
            logger.warning("Could not save rejected candidates to %s: %s", self.log_file, e)
    
    def add_rejected(
        self,
        candidate: str,
        constraints: Optional[List[str]] = None,
        reason: Optional[str] = None,
        source: Optional[str] = None,
    ) -> None:
        """
        Add a rejected candidate to the tracker.

        Args:
            candidate: Material name that was rejected
            constraints: List of constraints that were violated (optional)
            reason: Reason for rejection (optional)
            source: Optional source of rejection, e.g. "feasibility" or "manufacturability"
        """
        if not isinstance(candidate, str) or not candidate.strip():
            return

        candidate_lower = candidate.lower().strip()

        # Check if already rejected
        if self.is_rejected(candidate):
            # Update existing entry
            for entry in self.rejected_candidates:
                if entry.get("candidate", "").lower() == candidate_lower:
                    if constraints:
                        entry["constraints"] = constraints
                    if reason:
                        entry["reason"] = reason
                    if source is not None:
                        entry["source"] = source
                    break
        else:
            # Add new entry
            entry = {
                "candidate": candidate.strip(),
                "constraints": constraints or [],
                "reason": reason or "",
                "timestamp": None,
            }
            if source is not None:
                entry["source"] = source

            try:
                entry["timestamp"] = datetime.utcnow().isoformat() + "Z"
            except Exception:
                pass

            self.rejected_candidates.append(entry)

        self._save()
    
    def is_rejected(self, candidate: str) -> bool:
        """
        Check if a candidate has been rejected.
        
        Args:
            candidate: Material name to check
        
        Returns:
            bool: True if candidate is in rejected list
        """
        if not isinstance(candidate, str) or not candidate.strip():
            return False
        
        candidate_lower = candidate.lower().strip()
        
        for entry in self.rejected_candidates:
            if entry.get("candidate", "").lower() == candidate_lower:
                return True
        
        return False
    
    def get_all_rejected(self) -> List[str]:
        """
        Get list of all rejected candidate names.
        
        Returns:
            List[str]: List of rejected material names
        """
        return [entry.get("candidate", "") for entry in self.rejected_candidates if entry.get("candidate")]
    
    def get_all_rejection_details(self) -> List[Dict[str, Any]]:
        """
        Get all rejection entries with full details (candidate, constraints, reason, timestamp).
        
        Returns:
            List[Dict[str, Any]]: List of rejection entry dictionaries, each containing:
                - "candidate": Material name
                - "constraints": List of violated constraints
                - "reason": Reason for rejection
                - "timestamp": Timestamp of rejection
        """
        return [entry.copy() for entry in self.rejected_candidates]
    
    def get_rejection_info(self, candidate: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed rejection information for a candidate.
        
        Args:
            candidate: Material name
        
        Returns:
            Dict with rejection details, or None if not rejected
        """
        if not isinstance(candidate, str) or not candidate.strip():
            return None
        
        candidate_lower = candidate.lower().strip()
        
        for entry in self.rejected_candidates:
            if entry.get("candidate", "").lower() == candidate_lower:
                return entry.copy()
        
        return None
    
    def clear(self) -> None:
        """Clear all rejected candidates (use with caution)."""
        self.rejected_candidates = []
        self._save()


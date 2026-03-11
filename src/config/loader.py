"""YAML configuration loader with environment variable interpolation."""

import re
import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path

# Cache for loaded prompts to avoid repeated file reads
_prompts_cache: Optional[Dict[str, Any]] = None

# Pattern for ${VAR_NAME} or ${VAR_NAME:-default_value} in config strings
_ENV_VAR_PATTERN = re.compile(r"\$\{([^}:]+)(?::-([^}]*))?\}")


def _interpolate_env_vars(value: Any) -> Any:
    """Recursively replace ${VAR} and ${VAR:-default} placeholders with environment variable values."""
    if isinstance(value, str):
        def _replacer(match):
            var_name = match.group(1)
            default = match.group(2)  # None if no default specified
            env_val = os.environ.get(var_name)
            if env_val is not None:
                return env_val
            if default is not None:
                return default
            return match.group(0)  # leave unresolved placeholder as-is
        return _ENV_VAR_PATTERN.sub(_replacer, value)
    if isinstance(value, dict):
        return {k: _interpolate_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate_env_vars(item) for item in value]
    return value


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    String values may contain ``${ENV_VAR}`` or ``${ENV_VAR:-default}``
    placeholders that are resolved from the process environment.
    
    Args:
        config_path: Path to config YAML file. If None, looks for config/config.yaml
        
    Returns:
        Dictionary containing configuration values
    """
    if config_path is None:
        # Default to config/config.yaml relative to project root
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "config" / "config.yaml"
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    config = _interpolate_env_vars(config)
    return config


def load_prompts(prompts_path: Optional[str] = None, use_cache: bool = True) -> Dict[str, Any]:
    """
    Load system prompts from YAML file.
    
    Args:
        prompts_path: Path to prompts YAML file. If None, looks for config/prompts.yaml
        use_cache: If True, use cached prompts to avoid repeated file reads (default: True)
        
    Returns:
        Dictionary containing prompts organized by agent and method.
        Structure: {
            "llm": {"default": "..."},
            "agents": {
                "research_manager": {"default": "...", "answer_question": "...", ...},
                "research_assistant": {"default": "..."}
            }
        }
        
    Raises:
        FileNotFoundError: If prompts file doesn't exist
    """
    global _prompts_cache
    
    # Return cached prompts if available and caching is enabled
    if use_cache and _prompts_cache is not None:
        return _prompts_cache
    
    if prompts_path is None:
        # Default to config/prompts.yaml relative to project root
        project_root = Path(__file__).parent.parent.parent
        prompts_path = project_root / "config" / "prompts.yaml"
    
    # Load prompts.yaml - raise error if file doesn't exist
    if not os.path.exists(prompts_path):
        raise FileNotFoundError(
            f"Prompts configuration file not found: {prompts_path}. "
            "All system prompts must be defined in config/prompts.yaml"
        )
    
    with open(prompts_path, 'r', encoding='utf-8') as f:
        prompts = yaml.safe_load(f)
    
    # Cache the loaded prompts
    if use_cache:
        _prompts_cache = prompts
    
    return prompts


def clear_prompts_cache() -> None:
    """Clear the cached prompts. Useful for testing or reloading prompts."""
    global _prompts_cache
    _prompts_cache = None


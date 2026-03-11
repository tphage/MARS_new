"""LLM wrapper for gpt-oss compatibility"""

import logging
import time
from typing import List, Dict, Any, Optional, Callable
from openai import OpenAI
from ..config import load_prompts

logger = logging.getLogger(__name__)


def strip_after_message_marker(text: str) -> str:
    """Strip text after message marker for gpt-oss compatibility."""
    marker = "final<|message|>"  # "<|message|>"
    if marker in text:
        # keep only the part after the LAST marker
        text = text.rsplit(marker, 1)[-1]
    return text.strip()


def clean_messages_for_llm(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Clean messages for LLM by removing message markers."""
    cleaned = []
    for m in messages or []:
        m = m.copy()
        content = m.get("content")
        if isinstance(content, str):
            m["content"] = strip_after_message_marker(content)
        cleaned.append(m)
    return cleaned


class llm:
    """Minimal OpenAI-compatible LLM wrapper for local server routing."""
    
    def __init__(self, llm_config: Dict[str, Any]) -> None:
        """
        Initialize LLM wrapper.
        
        Args:
            llm_config: Dictionary with 'api_key', 'base_url', 'model' (or 'model_name'), 'max_tokens'
        """
        self.client = OpenAI(api_key=llm_config["api_key"], base_url=llm_config["base_url"])
        self.model = llm_config.get("model") or llm_config.get("model_name")
        if not self.model:
            raise ValueError(
                "llm_config must contain 'model' or 'model_name'. "
                "If passing raw config['llm'], map 'model_name' to 'model' first."
            )
        self.max_tokens = llm_config["max_tokens"]
    
    def generate_cli(
        self,
        system_prompt: Optional[str] = None,
        prompt: str = "Hello world! I am",
        temperature: Optional[float] = None,
        chat_logger: Optional[Any] = None,
        agent_name: Optional[str] = None,
        method_name: Optional[str] = None,
        **kwargs: Any
    ) -> str:
        """
        Generate text using the LLM.
        
        Args:
            system_prompt: System prompt for the LLM. If None, loads from config/prompts.yaml.
            prompt: User prompt
            temperature: Sampling temperature (default: None, uses config value)
            chat_logger: Optional ChatLogger instance for logging
            agent_name: Optional agent name for logging
            method_name: Optional method name for logging
            **kwargs: Additional arguments passed to LLM
            
        Returns:
            Generated text with message markers stripped
        """
        try:
            # Load config for default temperature
            from ..config import load_config
            config = load_config()
            llm_config_dict = config.get("llm", {})
            
            # Use config default if not provided
            if temperature is None:
                temperature = llm_config_dict.get("temperature", 0)
            
            # Store original system_prompt for logging
            original_system_prompt = system_prompt
            
            # If system_prompt is None, load from YAML
            if system_prompt is None:
                prompts = load_prompts()
                yaml_prompt = prompts.get("llm", {}).get("default")
                if yaml_prompt is None:
                    raise ValueError(
                        "Missing required prompt in config/prompts.yaml: llm.default. "
                        "All system prompts must be defined in the config file."
                    )
                system_prompt = yaml_prompt
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]

            # Retry logic for transient failures (connection errors, timeouts, 5xx)
            max_retries = llm_config_dict.get("max_retries", 3)
            retry_base_delay = llm_config_dict.get("retry_base_delay", 2)  # seconds
            last_exc = None
            for attempt in range(max_retries):
                try:
                    result = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=self.max_tokens,
                    )
                    text = result.choices[0].message.content
                    break
                except (ConnectionError, TimeoutError, OSError) as retry_exc:
                    last_exc = retry_exc
                    if attempt < max_retries - 1:
                        delay = retry_base_delay * (2 ** attempt)
                        logger.warning(
                            "LLM call attempt %d/%d failed (%s); retrying in %.1fs...",
                            attempt + 1, max_retries, retry_exc, delay,
                        )
                        time.sleep(delay)
                    else:
                        raise
                except Exception as api_exc:
                    # Check for server-side errors (status 5xx) that may be transient
                    exc_str = str(api_exc)
                    retry_codes = llm_config_dict.get("retry_status_codes", ["500", "502", "503", "504", "529"])
                    if any(code in exc_str for code in retry_codes):
                        last_exc = api_exc
                        if attempt < max_retries - 1:
                            delay = retry_base_delay * (2 ** attempt)
                            logger.warning(
                                "LLM call attempt %d/%d got server error (%s); retrying in %.1fs...",
                                attempt + 1, max_retries, api_exc, delay,
                            )
                            time.sleep(delay)
                        else:
                            raise
                    else:
                        raise  # Non-transient error, don't retry

            response = strip_after_message_marker(text)
            
            # Log the LLM call if chat_logger is provided
            if chat_logger is not None:
                try:
                    chat_logger.log_llm_call(
                        agent_name=agent_name or "unknown",
                        method_name=method_name or "generate_cli",
                        system_prompt=original_system_prompt or system_prompt,
                        user_prompt=prompt,
                        response=response,
                        temperature=temperature,
                        model=self.model,
                        **kwargs
                    )
                except Exception as e:
                    logger.warning("Failed to log LLM call: %s", e)
            
            return response
        except Exception as e:
            # Log the error for debugging — silent empty returns hide failures
            logger.error("LLM call failed (agent=%s, method=%s): %s",
                         agent_name or "unknown", method_name or "generate_cli", e)
            # Log error if chat_logger is provided
            if chat_logger is not None:
                try:
                    chat_logger.log_llm_call(
                        agent_name=agent_name or "unknown",
                        method_name=method_name or "generate_cli",
                        system_prompt=original_system_prompt,
                        user_prompt=prompt,
                        response=f"ERROR: {str(e)}",
                        temperature=temperature,
                        model=self.model,
                        error=str(e),
                        **kwargs
                    )
                except Exception:
                    pass
            # Return empty string for backward compatibility, but callers
            # should check for empty responses. Raising here would break
            # existing pipelines that catch-and-continue.
            return ''


def create_logged_generate_fn(generate_fn: Callable, chat_logger: Optional[Any] = None, agent_name: Optional[str] = None) -> Callable:
    """
    Create a wrapper around generate_fn that automatically logs LLM calls.
    
    Args:
        generate_fn: Original generate function (e.g., llm_instance.generate_cli)
        chat_logger: Optional ChatLogger instance
        agent_name: Name of the agent using this function
        
    Returns:
        Wrapped generate function that logs calls
    """
    if chat_logger is None:
        return generate_fn
    
    def logged_generate_fn(system_prompt=None, prompt="", temperature=None, method_name=None, **kwargs):
        # Extract method_name from kwargs if passed there instead of as a keyword arg
        if method_name is None:
            method_name = kwargs.pop("method_name", None)
        else:
            # Remove duplicate method_name from kwargs to avoid TypeError
            kwargs.pop("method_name", None)
        
        # Call original function with chat_logger
        return generate_fn(
            system_prompt=system_prompt,
            prompt=prompt,
            temperature=temperature,
            chat_logger=chat_logger,
            agent_name=agent_name,
            method_name=method_name,
            **kwargs
        )
    
    return logged_generate_fn


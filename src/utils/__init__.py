"""Utility functions and classes"""

from .llm_wrapper import llm, strip_after_message_marker, clean_messages_for_llm, create_logged_generate_fn
from .embeddings import TransformerEmbeddingFunction, custom_token_count_function
from .autogen_agent import AssistantAgent_gptoss, create_assistant_agent_gptoss
from .chat_logger import ChatLogger
from .material_database import MaterialDatabase
from .property_mapper import PropertyMapper
from .subgraph_processor import SubgraphProcessor
from .material_grounding import MaterialGrounding
from .step1_cache import Step1Cache
from .evaluation_export import build_evaluation_payload, save_evaluation_export
from .ablation_utils import (
    load_ablation_queries,
    extract_json_from_response,
    build_ablation_evaluation,
    save_ablation_result,
    format_rag_results_for_prompt,
    format_kg_results_for_prompt,
)

__all__ = [
    "llm",
    "strip_after_message_marker",
    "clean_messages_for_llm",
    "create_logged_generate_fn",
    "TransformerEmbeddingFunction",
    "custom_token_count_function",
    "AssistantAgent_gptoss",
    "create_assistant_agent_gptoss",
    "ChatLogger",
    "MaterialDatabase",
    "PropertyMapper",
    "SubgraphProcessor",
    "MaterialGrounding",
    "Step1Cache",
    "build_evaluation_payload",
    "save_evaluation_export",
    "load_ablation_queries",
    "extract_json_from_response",
    "build_ablation_evaluation",
    "save_ablation_result",
    "format_rag_results_for_prompt",
    "format_kg_results_for_prompt",
]


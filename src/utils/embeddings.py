"""Embedding utilities for ChromaDB and knowledge graphs"""

import importlib
from typing import Optional, Union, TypeVar
import numpy as np
import numpy.typing as npt
import torch
import networkx as nx
from chromadb.api.types import EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer


Embeddable = Union[str, nx.DiGraph]
D = TypeVar("D", bound=Embeddable, contravariant=True)


class TransformerEmbeddingFunction(EmbeddingFunction[D]):
    """ChromaDB-compatible embedding function wrapper.
    
    Provides normalization and dual-mode encoding (HF-style or SentenceTransformer).
    """
    
    def __init__(
        self,
        embedding_tokenizer,
        embedding_model,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize embedding function.
        
        Args:
            embedding_tokenizer: Tokenizer for embeddings (can be None for SentenceTransformer)
            embedding_model: Embedding model (SentenceTransformer or HuggingFace model)
            cache_dir: Optional cache directory
            device: Device for inference ("auto", "cuda:0", "cpu", etc.). If None, uses config embeddings.device
        """
        if device is None:
            try:
                from ..config import load_config
                device = load_config().get("embeddings", {}).get("device", "auto")
            except Exception:
                device = "auto"
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._device = device
        try:
            from transformers import AutoModel, AutoTokenizer
            self._torch = importlib.import_module("torch")
            # Use provided tokenizer/model; HF constructors intentionally not invoked here.
            self._tokenizer = embedding_tokenizer
            self._model = embedding_model
        except ImportError:
            raise ValueError(
                "The transformers and/or pytorch package is not installed. Please install it with "
                "pip install transformers or pip install torch"
            )

    @staticmethod
    def _normalize(vector: npt.NDArray) -> npt.NDArray:
        """L2-normalize embedding vectors to unit length for cosine similarity."""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def __call__(self, input: D) -> Embeddings:
        """Tokenize and embed input; return normalized embeddings as Python lists."""
        if self._tokenizer:
            inputs = self._tokenizer(
                input, padding=True, truncation=True, return_tensors="pt"
            ).to(self._device)
            outputs = self._model(**inputs)
            try:
                embeddings = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()
            except (AttributeError, RuntimeError):
                # Fallback for models that use hidden_states instead of last_hidden_state
                embeddings = outputs.hidden_states[-1].mean(dim=1).detach().to(torch.float).cpu().numpy()
        else:
            embeddings = self._model.encode(input, show_progress_bar=False)
        return [e.tolist() for e in self._normalize(embeddings)]


def custom_token_count_function(text: str, placeholder: str = '', embedding_model: Optional[SentenceTransformer] = None) -> int:
    """
    Token counting to estimate context budgets when assembling retrieval contexts.
    
    Args:
        text: Text to count tokens for
        placeholder: Unused placeholder
        embedding_model: SentenceTransformer model with tokenizer
        
    Returns:
        Number of tokens
    """
    if embedding_model is None:
        raise ValueError("embedding_model is required")
    inputs = embedding_model.tokenizer(text, return_tensors='pt', truncation=True)
    return len(inputs["input_ids"][0])


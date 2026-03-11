"""ResearchAnalyst Agent - Performs RAG retrieval (no LLM)"""

from typing import List, Dict, Any, Optional
from ..config import load_config


class ResearchAnalyst:
    """
    Research analyst agent that performs RAG (Retrieval-Augmented Generation) using ChromaDB.
    No LLM calls - only performs retrieval.
    """
    
    def __init__(self, collection, embedding_function=None, n_results: int = None, distance_threshold: float = None, chat_logger=None):
        """
        Initialize the ResearchAnalyst agent.
        
        Args:
            collection: ChromaDB collection object (must have query method)
            embedding_function: Optional embedding function. If None, uses collection's default
            n_results: Number of results to retrieve (default: None, uses config value)
            distance_threshold: Optional distance threshold for filtering results
            chat_logger: Optional ChatLogger instance for logging RAG queries
        """
        # Load config
        config = load_config()
        agent_config = config.get("agents", {}).get("research_analyst", {})
        
        # Use config default if not provided
        if n_results is None:
            n_results = agent_config.get("n_results", 5)
        
        self.collection = collection
        self.embedding_function = embedding_function
        self.n_results = n_results
        self.distance_threshold = distance_threshold
        self.chat_logger = chat_logger
        
        # Load RAG query multiplier from config
        self.rag_query_multiplier = agent_config.get("rag_query_multiplier", 20)
    
    def analyze(self, sentence: str, keywords: List[str] = None) -> Dict[str, Any]:
        """
        Analyze a sentence with optional keywords by performing RAG.
        
        Args:
            sentence: Input sentence
            keywords: Optional list of keywords (can be any number of keywords, including 0 or None)
            
        Returns:
            Dict containing:
                - "sentence": Original input sentence
                - "keywords": List of provided keywords (empty list if None)
                - "rag_results": List of retrieved documents from RAG
                - "num_results": Number of retrieved documents
        """
        if keywords is None:
            keywords = []
        
        if not isinstance(sentence, str):
            raise ValueError(f"Input sentence must be a string, got {type(sentence)}")
        
        if not sentence.strip():
            raise ValueError("Input sentence cannot be empty")
        
        if not isinstance(keywords, list):
            raise ValueError(f"Keywords must be a list, got {type(keywords)}")
        
        # Keywords are optional - allow empty list
        if len(keywords) > 0:
            # If keywords are provided, validate they are all strings and non-empty
            if not all(isinstance(kw, str) for kw in keywords):
                raise ValueError("All keywords must be strings")
            
            if not all(kw.strip() for kw in keywords):
                raise ValueError("Keywords cannot be empty strings")
        
        # Perform RAG retrieval
        try:
            # Query Chroma with a larger pool, then filter down ourselves
            results = self.collection.query(
                query_texts=[sentence],
                n_results=self.n_results * self.rag_query_multiplier,
                include=["documents", "metadatas", "distances"],
            )

            # Chroma returns lists per query; we only have one query_text
            documents_all = results.get("documents", [[]])[0] if results.get("documents") else []
            metadatas_all = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
            distances_all = results.get("distances", [[]])[0] if results.get("distances") else []
            ids_all = results.get("ids", [[]])[0] if results.get("ids") else [None] * len(documents_all)

            rag_results: List[Dict[str, Any]] = []

            # Iterate over all retrieved candidates
            for doc, meta, dist, _id in zip(documents_all, metadatas_all, distances_all, ids_all):
                # Keyword filtering (AND over all keywords, case-insensitive)
                if keywords and len(keywords) > 0:
                    doc_lower = doc.lower()
                    if not all(k.lower() in doc_lower for k in keywords):
                        continue

                # Distance threshold filtering, if enabled
                if self.distance_threshold is not None and dist is not None:
                    if dist > self.distance_threshold:
                        continue

                # Add formatted result
                rag_results.append(
                    {
                        "content": doc,
                        "distance": dist,
                        "id": _id,
                        "metadata": meta if meta is not None else {},
                    }
                )

                # Stop once we have enough results
                if len(rag_results) >= self.n_results:
                    break

            result = {
                "sentence": sentence,
                "keywords": keywords,
                "rag_results": rag_results,
                "num_results": len(rag_results)
            }
            
            # Log RAG query if chat_logger is provided
            if self.chat_logger is not None:
                try:
                    self.chat_logger.log_rag_query(
                        agent_name="research_analyst",
                        query=sentence,
                        keywords=keywords,
                        results=rag_results,
                        num_results=len(rag_results),
                        method_name="analyze"
                    )
                except Exception as e:
                    print(f"Warning: Failed to log RAG query: {e}")
            
            return result

        except Exception as e:
            raise RuntimeError(f"Error performing RAG retrieval: {str(e)}")
    
    def analyze_question(self, question: str) -> Dict[str, Any]:
        """
        Analyze a question without keywords by performing RAG.
        This method is designed for questions generated by ResearchManager that have no keywords.
        
        Args:
            question: Input question (string)
            
        Returns:
            Dict containing:
                - "question": Original input question
                - "keywords": Empty list (no keywords for questions)
                - "rag_results": List of retrieved documents from RAG
                - "num_results": Number of retrieved documents
        """
        if not isinstance(question, str):
            raise ValueError(f"Input question must be a string, got {type(question)}")
        
        if not question.strip():
            raise ValueError("Input question cannot be empty")
        
        # Perform RAG without keywords (questions have no keywords)
        result = self.analyze(question, keywords=[])
        result["question"] = question
        
        # Note: RAG logging is already handled in analyze() method, so no need to log again here
        return result


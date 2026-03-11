"""MultiAnalyst Agent - Wrapper for querying multiple ResearchAnalyst instances with source tagging"""

from typing import List, Dict, Any, Optional
from .research_analyst import ResearchAnalyst


class MultiAnalyst:
    """
    Wrapper class that queries multiple ResearchAnalyst instances and combines results
    with source tagging so the LLM can distinguish which results came from which source.
    """
    
    def __init__(self, analysts: Dict[str, ResearchAnalyst]):
        """
        Initialize the MultiAnalyst wrapper.
        
        Args:
            analysts: Dictionary mapping source names to ResearchAnalyst instances
                    e.g., {"patents": analyst_patents, "pfas_papers": analyst_pfas}
        """
        self.analysts = analysts
    
    def analyze(self, sentence: str, keywords: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Query all analysts and combine results with source tagging.
        
        Args:
            sentence: Input sentence to query
            keywords: Optional list of keywords for filtering
        
        Returns:
            Dict with combined results, where each rag_result includes a "source" field
        """
        all_rag_results = []
        
        for source_name, analyst in self.analysts.items():
            try:
                result = analyst.analyze(sentence, keywords)
                rag_results = result.get("rag_results", [])
                
                # Tag each result with its source
                for rag_result in rag_results:
                    rag_result["source"] = source_name
                    all_rag_results.append(rag_result)
            except Exception as e:
                print(f"Warning: Error querying {source_name} analyst: {e}")
        
        # Sort by distance (lower is better) and limit to top results
        all_rag_results.sort(key=lambda x: x.get("distance", float("inf")))
        
        return {
            "sentence": sentence,
            "keywords": keywords if keywords else [],
            "rag_results": all_rag_results,
            "num_results": len(all_rag_results),
            "sources": list(self.analysts.keys())
        }
    
    def analyze_question(self, question: str) -> Dict[str, Any]:
        """
        Query all analysts with a question and combine results with source tagging.
        
        Args:
            question: Input question to query
        
        Returns:
            Dict with combined results, where each rag_result includes a "source" field
        """
        all_rag_results = []
        
        for source_name, analyst in self.analysts.items():
            try:
                result = analyst.analyze_question(question)
                rag_results = result.get("rag_results", [])
                
                # Tag each result with its source
                for rag_result in rag_results:
                    rag_result["source"] = source_name
                    all_rag_results.append(rag_result)
            except Exception as e:
                print(f"Warning: Error querying {source_name} analyst: {e}")
        
        # Sort by distance (lower is better) and limit to top results
        all_rag_results.sort(key=lambda x: x.get("distance", float("inf")))
        
        return {
            "question": question,
            "keywords": [],
            "rag_results": all_rag_results,
            "num_results": len(all_rag_results),
            "sources": list(self.analysts.keys())
        }

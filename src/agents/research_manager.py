"""ResearchManager Agent - Generates questions and answers them using LLM"""

import json
import re
import warnings
import networkx as nx
from typing import List, Dict, Any, Optional
from ..config import load_prompts, load_config
from ..utils.parsing import parse_to_list, clean_material_name
from ..utils.ablation_utils import extract_json_from_response
from GraphReasoning import find_best_fitting_node_list


class ResearchManager:
    """
    Research Manager Agent that processes research queries using LLM.
    Generates research questions and answers them based on RAG results.
    Also includes methods for material discovery: propose_candidate, generate_validation_queries, validate_feasibility.
    """
    
    def __init__(self, name: str = "research_manager", system_message: str = None, generate_fn=None, chat_logger=None):
        """
        Initialize the ResearchManager agent.
        
        Args:
            name: Agent name (for identification/logging purposes)
            system_message: System message/prompt for the agent (optional)
            generate_fn: LLM generate function (required)
            chat_logger: Optional ChatLogger instance for logging interactions
        """
        self.name = name
        self.chat_logger = chat_logger
        
        # Load config and prompts from YAML
        config = load_config()
        prompts = load_prompts()
        agent_prompts = prompts.get("agents", {}).get("research_manager", {})
        agent_config = config.get("agents", {}).get("research_manager", {})
        
        # Get default system message from YAML - raise error if missing
        default_system_message = agent_prompts.get("default")
        if default_system_message is None:
            raise ValueError(
                "Missing required prompt in config/prompts.yaml: agents.research_manager.default. "
                "All system prompts must be defined in the config file."
            )
        
        self.system_message = system_message or default_system_message
        
        # Store agent prompts for use in other methods
        self._agent_prompts = agent_prompts
        
        # Load hyperparameters from config
        self.temperature = agent_config.get("temperature", 0)
        self.max_queries = agent_config.get("max_queries", 4)
        formatting_config = agent_config.get("formatting", {})
        self.max_chars_per_result = formatting_config.get("max_chars_per_result", 10000)
        self.max_chars_per_result_answer = formatting_config.get("max_chars_per_result_answer", 3000)
        self.max_chars_per_result_validation = formatting_config.get("max_chars_per_result_validation", 1500)
        self.max_chars_per_result_feasibility = formatting_config.get("max_chars_per_result_feasibility", 2000)
        self.max_paths = formatting_config.get("max_paths", 100)
        
        # Max prompt budget (approximate characters). Prevents exceeding context windows.
        self.max_prompt_chars = agent_config.get("max_prompt_chars", 120000)

        # Validation thresholds
        validation_config = agent_config.get("validation", {})
        self.min_answer_length = validation_config.get("min_answer_length", 10)
        self.min_evaluation_response_length = validation_config.get("min_evaluation_response_length", 5)
        self.min_answer_length_answered = validation_config.get("min_answer_length_answered", 50)
        self.max_candidate_name_length = validation_config.get("max_candidate_name_length", 100)
        self.yes_feasible_proximity_chars = validation_config.get("yes_feasible_proximity_chars", 50)

        # Context limits for prompts
        context_config = agent_config.get("context_limits", {})
        self.max_kg_nodes_in_context = context_config.get("max_kg_nodes_in_context", 5)
        self.max_rag_results_in_context = context_config.get("max_rag_results_in_context", 5)
        
        # Wrap generate_fn to add logging if chat_logger is provided
        if generate_fn is None:
            raise ValueError("generate_fn is required. Pass the LLM generate function.")
        
        if chat_logger is not None:
            # Create a wrapper that adds agent_name and method_name to calls
            original_generate_fn = generate_fn
            def logged_generate_fn(system_prompt=None, prompt="", temperature=0, method_name=None, **kwargs):
                # Allow agent_name to be overridden from kwargs, but default to name
                agent_name = kwargs.pop('agent_name', name)
                # Remove chat_logger from kwargs if present (already handled by wrapper)
                kwargs.pop('chat_logger', None)
                return original_generate_fn(
                    system_prompt=system_prompt,
                    prompt=prompt,
                    temperature=temperature,
                    chat_logger=chat_logger,
                    agent_name=agent_name,
                    method_name=method_name,
                    **kwargs
                )
            self._generate_fn = logged_generate_fn
        else:
            self._generate_fn = generate_fn
    
    def _format_rag_context(self, rag_results: List[Dict[str, Any]], max_chars_per_result: int = None) -> str:
        """Format RAG results into a context string for the prompt."""
        if not rag_results:
            return "No relevant documents found."
        
        # Use config default if not provided
        if max_chars_per_result is None:
            max_chars_per_result = self.max_chars_per_result
        
        context_parts = [f"Found {len(rag_results)} relevant documents:"]
        
        for i, result in enumerate(rag_results, 1):
            content = result.get('content', '')
            if len(content) > max_chars_per_result:
                content = content[:max_chars_per_result] + "..."
            
            context_parts.append(f"\n[Document {i}]")
            # Explicitly show source if available (for multi-source RAG)
            if result.get('source'):
                context_parts.append(f"Source: {result['source']}")
            if result.get('id'):
                context_parts.append(f"ID: {result['id']}")
            if result.get('metadata'):
                metadata_str = ", ".join([f"{k}: {v}" for k, v in result['metadata'].items()])
                if metadata_str:
                    context_parts.append(f"Metadata: {metadata_str}")
            context_parts.append(f"Content: {content}")
        
        return "\n".join(context_parts)
    
    def _truncate_prompt(self, prompt: str, label: str = "prompt") -> str:
        """Truncate a prompt string if it exceeds the configured character budget.

        This is a safety-net: it prevents context-window overflows when
        large subgraphs, many properties, or verbose RAG results are assembled
        into a single prompt.

        Args:
            prompt: The assembled prompt text.
            label:  Human-readable label for the warning message.

        Returns:
            The (possibly truncated) prompt string.
        """
        if len(prompt) > self.max_prompt_chars:
            warnings.warn(
                f"{label} exceeds budget ({len(prompt)} > {self.max_prompt_chars} chars); truncating."
            )
            prompt = prompt[: self.max_prompt_chars] + "\n\n[...truncated due to length...]"
        return prompt

    def _log_prompt_length(self, prompt: str, label: str = "prompt") -> None:
        """Log prompt character counts without truncating content."""
        prompt_len = len(prompt or "")
        if prompt_len > self.max_prompt_chars:
            warnings.warn(
                f"{label} length ({prompt_len} chars) exceeds configured max_prompt_chars="
                f"{self.max_prompt_chars}; sending untruncated prompt by design."
            )

    def _get_edge_label(self, edge_data: Dict[str, Any]) -> str:
        """Extract relationship label from edge data."""
        if not edge_data:
            return ""
        # Check common attribute names in priority order
        for attr in ["relation", "label", "type", "title", "name"]:
            if attr in edge_data:
                attr_value = edge_data[attr]
                # Check if value exists and is non-empty (handle None, empty strings, whitespace)
                if attr_value is not None:
                    label_str = str(attr_value).strip()
                    if label_str:  # Non-empty after stripping whitespace
                        return label_str
        return ""  # No label found
    
    def _format_kg_context(self, kg_results: Dict[str, Any], max_paths: int = None) -> str:
        """Format knowledge graph results into a context string for the prompt."""
        if not kg_results:
            return ""
        
        # Use config default if not provided
        if max_paths is None:
            max_paths = self.max_paths
        
        summary = kg_results.get('summary', {})
        if not summary.get('connections_found', False):
            return ""
        
        context_parts = []
        context_parts.append("Knowledge Graph Insights (PFAS Domain):")
        
        # Add summary statistics
        num_paths = summary.get('num_paths_found', 0)
        num_nodes = summary.get('subgraph_nodes', 0)
        num_edges = summary.get('subgraph_edges', 0)
        num_matched = summary.get('num_matched_nodes', 0)
        
        context_parts.append(f"- Found {num_paths} connections between concepts")
        context_parts.append(f"- Matched {num_matched} relevant nodes")
        context_parts.append(f"- Subgraph: {num_nodes} nodes, {num_edges} edges")
        
        # Add key relationships/paths — use the same formatter as propose_candidate
        found_paths = kg_results.get('found_paths', [])
        if found_paths:
            formatted_paths = self._format_kg_paths(found_paths, max_paths=max_paths)
            if formatted_paths:
                context_parts.append(formatted_paths)
        
        # Add matched nodes information
        keyword_to_nodes = kg_results.get('keyword_to_nodes', {})
        keyword_mappings = keyword_to_nodes.get('keyword_mappings', [])
        if keyword_mappings:
            context_parts.append(f"\nMatched concepts:")
            for km in keyword_mappings[:3]:  # Show top 3 keyword mappings
                keyword = km.get('keyword', '')
                matched_nodes = km.get('matched_nodes', [])
                if matched_nodes:
                    top_node = matched_nodes[0]
                    node_id = top_node.get('node_id', '')
                    node_data = top_node.get('node_data', {})
                    # Try to get a readable name from node data
                    node_name = node_data.get('title') or node_data.get('label') or node_data.get('name') or str(node_id)[:50]
                    context_parts.append(f"  - '{keyword}' → {node_name}")
        
        return "\n".join(context_parts)
    
    def _extract_keywords_from_text(self, text: str, max_keywords: int = 5, temperature: float = None) -> List[str]:
        """
        Extract keywords from text for knowledge graph querying using LLM.
        
        Args:
            text: Text to extract keywords from
            max_keywords: Maximum number of keywords to return (default: 5)
            temperature: Temperature for LLM generation (default: None, uses config value)
        
        Returns:
            List of keyword strings suitable for KG querying
        """
        if not text or not isinstance(text, str):
            return []
        
        # Use config default if not provided
        if temperature is None:
            temperature = self.temperature
        
        # Load user prompt template from YAML
        extract_keywords_user_prompt_template = self._agent_prompts.get("extract_keywords_from_text_user_prompt")
        if extract_keywords_user_prompt_template is None:
            raise ValueError(
                "Missing required prompt in config/prompts.yaml: agents.research_manager.extract_keywords_from_text_user_prompt. "
                "All system prompts must be defined in the config file."
            )
        
        # Load system prompt from YAML
        extract_keywords_system_prompt = self._agent_prompts.get("extract_keywords_from_text")
        if extract_keywords_system_prompt is None:
            raise ValueError(
                "Missing required prompt in config/prompts.yaml: agents.research_manager.extract_keywords_from_text. "
                "All system prompts must be defined in the config file."
            )
        
        # Format user prompt with text
        prompt = extract_keywords_user_prompt_template.format(text=text)
        
        # Generate keywords using the LLM
        content = self._generate_fn(
            system_prompt=extract_keywords_system_prompt,
            prompt=prompt,
            temperature=temperature,
            method_name="_extract_keywords_from_text",
            **{}
        )
        
        # Parse the response into a list of keywords/phrases
        keywords = self._parse_to_list(content)
        
        # Limit to max_keywords
        return keywords[:max_keywords]
    
    def _find_paths_in_subgraph(
        self,
        keywords: List[str],
        subgraph: nx.DiGraph,
        node_embeddings: Dict[str, Any],
        embedding_tokenizer,
        embedding_model,
        n_samples: int = 5,
        similarity_threshold: float = 0.8,
        max_depth: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find paths between keywords in a subgraph.
        
        Maps keywords to nodes using semantic matching, then finds paths between matched nodes.
        
        Args:
            keywords: List of keyword strings
            subgraph: NetworkX DiGraph subgraph to search
            node_embeddings: Dictionary mapping node IDs to embedding vectors
            embedding_tokenizer: Tokenizer for generating embeddings
            embedding_model: Model for generating embeddings
            n_samples: Number of top matching nodes per keyword (default: 5)
            similarity_threshold: Minimum similarity threshold for matching (default: 0.8)
            max_depth: Maximum path depth (default: 10)
        
        Returns:
            List of path dictionaries with 'path', 'edges', 'source', 'target', 'length'
        """
        if not keywords or len(keywords) < 2:
            return []
        
        if subgraph is None or subgraph.number_of_nodes() == 0:
            return []
        
        # Step 1: Map keywords to nodes in subgraph
        matched_node_ids = []
        for keyword in keywords:
            if not isinstance(keyword, str) or not keyword.strip():
                continue
            
            # Find best matching nodes for this keyword
            matched_nodes = find_best_fitting_node_list(
                keyword,
                node_embeddings,
                embedding_tokenizer,
                embedding_model,
                N_samples=n_samples,
                similarity_threshold=similarity_threshold
            )
            
            # Filter to only nodes that exist in subgraph
            for node_id, similarity in matched_nodes:
                if node_id in subgraph.nodes():
                    matched_node_ids.append(node_id)
                    break  # Use best match only
        
        # Remove duplicates while preserving order
        seen = set()
        unique_node_ids = []
        for node_id in matched_node_ids:
            if node_id not in seen:
                seen.add(node_id)
                unique_node_ids.append(node_id)
        
        if len(unique_node_ids) < 2:
            return []
        
        # Step 2: Find paths between matched nodes
        found_paths = []
        try:
            undirected_graph = subgraph.to_undirected()
            
            for i in range(len(unique_node_ids)):
                for j in range(i + 1, len(unique_node_ids)):
                    source = unique_node_ids[i]
                    target = unique_node_ids[j]
                    
                    try:
                        path = nx.shortest_path(undirected_graph, source, target)
                        
                        # Extract edge metadata
                        edges_metadata = []
                        for k in range(len(path) - 1):
                            u, v = path[k], path[k + 1]
                            edge_info = {"source": u, "target": v}
                            
                            # Get edge data from subgraph (check both directions)
                            # Prefer forward direction, but use reverse if forward doesn't exist
                            edge_data = {}
                            if subgraph.has_edge(u, v):
                                edge_attrs = subgraph[u][v]
                                edge_data = dict(edge_attrs)
                            elif subgraph.has_edge(v, u):
                                # Edge exists in reverse direction - get its data
                                # Note: The relation label should still be valid even in reverse
                                edge_attrs = subgraph[v][u]
                                edge_data = dict(edge_attrs)
                            
                            # If edge_data is still empty, the edge might exist in undirected
                            # but not in directed graph, or has no attributes
                            edge_info["edge_data"] = edge_data
                            
                            edges_metadata.append(edge_info)
                        
                        found_paths.append({
                            "source": source,
                            "target": target,
                            "path": path,
                            "length": len(path) - 1,
                            "edges": edges_metadata
                        })
                    except nx.NetworkXNoPath:
                        pass
        except Exception as e:
            warnings.warn(f"Error finding paths in subgraph: {e}", UserWarning)
        
        return found_paths
    
    def _format_kg_paths(self, found_paths: List[Dict[str, Any]], max_paths: int = None) -> str:
        """
        Format knowledge graph paths into a readable string.
        
        Args:
            found_paths: List of path dictionaries with 'path', 'edges', etc.
            max_paths: Maximum number of paths to show (default: None, uses config value)
        
        Returns:
            Formatted string with paths
        """
        if not found_paths:
            return ""
        
        # Use config default if not provided
        if max_paths is None:
            max_paths = self.max_paths
        
        context_parts = []
        context_parts.append(f"\nKey relationships (showing first {min(max_paths, len(found_paths))}):")
        
        for i, path_info in enumerate(found_paths[:max_paths], 1):
            path = path_info.get('path', [])
            if path and len(path) >= 2:
                # Get edges if available
                edges = path_info.get('edges', [])
                has_edges = edges and len(edges) > 0
                
                # Format path with edge labels if edges are available
                if has_edges:
                    path_parts = []
                    for j, node in enumerate(path):
                        # Add node (full string, no truncation)
                        path_parts.append(str(node))
                        
                        # Add edge label if available (between nodes)
                        if j < len(path) - 1:  # Not the last node
                            if j < len(edges):
                                edge_info = edges[j]
                                edge_data = edge_info.get('edge_data', {})
                                edge_label = self._get_edge_label(edge_data)
                                if edge_label:
                                    path_parts.append(f"[{edge_label}]")
                                elif edge_data:
                                    # Edge data exists but no label found - edge exists but lacks relation attribute
                                    path_parts.append("[connected]")
                                else:
                                    # No edge data at all
                                    path_parts.append("[]")
                            else:
                                path_parts.append("[]")
                    
                    path_str = " → ".join(path_parts)
                else:
                    # Fall back to original format if edges are not available
                    path_str = " → ".join([str(node) for node in path])
                
                context_parts.append(f"  [{i}] {path_str}")
        
        return "\n".join(context_parts)
    
    def _parse_to_list(self, content: str) -> List[str]:
        """Parse LLM response content into a list of strings.

        Delegates to the shared ``parse_to_list`` utility so that parsing
        logic is maintained in one place.
        """
        return parse_to_list(content)
    
    def process(self, analysis_result: Dict[str, Any], temperature: float = None, max_items: int = None, **kwargs) -> List[str]:
        """
        Process a dictionary from ResearchAnalyst.analyze() and return a list of strings (questions).
        
        Args:
            analysis_result: Dictionary from ResearchAnalyst.analyze() containing:
                - "sentence": The input sentence (str)
                - "keywords": List of keywords (List[str])
                - "rag_results": List of retrieved documents (List[Dict])
                - "num_results": Number of results (int)
            temperature: Temperature for LLM generation (default: None, uses config value)
            max_items: Maximum number of items to return. If None, returns all items (default: None)
            **kwargs: Additional arguments passed to generate function
            
        Returns:
            List[str]: A list of strings (questions) generated by the LLM, parsed from the response
                       Limited to max_items if specified
        """
        # Use config default if not provided
        if temperature is None:
            temperature = self.temperature
        
        if not isinstance(analysis_result, dict):
            raise ValueError(f"analysis_result must be a dictionary, got {type(analysis_result)}")
        
        if 'sentence' not in analysis_result:
            raise ValueError("analysis_result must contain 'sentence' key")
        
        if 'keywords' not in analysis_result:
            raise ValueError("analysis_result must contain 'keywords' key")
        
        sentence = analysis_result['sentence']
        keywords = analysis_result['keywords']
        rag_results = analysis_result.get('rag_results', [])
        
        # Load user prompt template from YAML
        process_user_prompt_template = self._agent_prompts.get("process_user_prompt")
        if process_user_prompt_template is None:
            raise ValueError(
                "Missing required prompt in config/prompts.yaml: agents.research_manager.process_user_prompt. "
                "All system prompts must be defined in the config file."
            )
        
        # Format RAG context if available
        rag_context = ""
        if rag_results and len(rag_results) > 0:
            rag_context = f"Retrieved Information:\n{self._format_rag_context(rag_results)}\n"
        
        # Format keywords section
        keywords_section = ""
        if keywords and len(keywords) > 0:
            keywords_str = ", ".join(keywords)
            keywords_section = f"Keywords: {keywords_str}"
        
        # Format user prompt with dynamic content
        prompt = process_user_prompt_template.format(
            rag_context=rag_context,
            sentence=sentence,
            keywords_section=keywords_section
        )
        
        # Generate response using the LLM
        content = self._generate_fn(
            system_prompt=self.system_message,
            prompt=prompt,
            temperature=temperature,
            method_name="process",
            **kwargs
        )
        
        # Parse the response into a list of strings
        result_list = self._parse_to_list(content)
        
        # Limit to max_items if specified
        if max_items is not None and max_items > 0:
            result_list = result_list[:max_items]
        
        return result_list
    
    def answer_question(self, question: str, rag_results: List[Dict[str, Any]], temperature: float = None, kg_context: Optional[Dict[str, Any]] = None, **kwargs) -> str:
        """
        Answer a specific question based on retrieved RAG results and optional knowledge graph context.
        
        Args:
            question: The question to answer
            rag_results: List of RAG result dictionaries containing retrieved documents
            temperature: Temperature for LLM generation (default: None, uses config value)
            kg_context: Optional knowledge graph context dictionary from ResearchScientist.find_connections()
            **kwargs: Additional arguments passed to generate function
            
        Returns:
            str: A distilled answer to the question based on the retrieved information
        """
        # Use config default if not provided
        if temperature is None:
            temperature = self.temperature
        
        if not isinstance(question, str):
            raise ValueError(f"Question must be a string, got {type(question)}")
        
        if not question.strip():
            raise ValueError("Question cannot be empty")
        
        # Load user prompt template from YAML
        answer_user_prompt_template = self._agent_prompts.get("answer_question_user_prompt")
        if answer_user_prompt_template is None:
            raise ValueError(
                "Missing required prompt in config/prompts.yaml: agents.research_manager.answer_question_user_prompt. "
                "All system prompts must be defined in the config file."
            )
        
        # Format RAG context
        if rag_results and len(rag_results) > 0:
            rag_context = self._format_rag_context(rag_results, max_chars_per_result=self.max_chars_per_result_answer)
            rag_context_str = f"Retrieved Information (from documents):\n{rag_context}\n"
        else:
            rag_context_str = "No relevant documents were retrieved for this question.\n"
        
        # Format KG context
        kg_context_str = ""
        kg_instruction = ""
        if kg_context is not None:
            kg_formatted = self._format_kg_context(kg_context)
            if kg_formatted:
                kg_context_str = f"{kg_formatted}\n\nThe knowledge graph insights above provide cross-domain relationships that may help answer the question.\n"
                kg_instruction = "Consider both the document content and the knowledge graph relationships when formulating your answer."
        
        # Format user prompt with dynamic content
        prompt = answer_user_prompt_template.format(
            question=question,
            rag_context=rag_context_str,
            kg_context=kg_context_str,
            kg_instruction=kg_instruction
        )
        
        # Use a different system message for answering questions
        answer_system_message = self._agent_prompts.get("answer_question")
        if answer_system_message is None:
            raise ValueError(
                "Missing required prompt in config/prompts.yaml: agents.research_manager.answer_question. "
                "All system prompts must be defined in the config file."
            )
        
        # Generate answer using the LLM
        content = self._generate_fn(
            system_prompt=answer_system_message,
            prompt=prompt,
            temperature=temperature,
            method_name="answer_question",
            **kwargs
        )
        
        return content
    
    def _is_question_answered(self, question: str, answer: str, num_rag_documents: int, temperature: float = None) -> bool:
        """
        Use LLM to evaluate if a question was answered based on the answer content.
        
        Args:
            question: The original question string
            answer: The generated answer string
            num_rag_documents: Number of RAG documents retrieved
            temperature: Temperature for LLM evaluation (default: None, uses config value)
            
        Returns:
            bool: True if the question was answered (LLM returns 1), False otherwise (LLM returns 0)
        """
        # Use config default if not provided
        if temperature is None:
            temperature = self.temperature
        
        # If no documents were retrieved, question is not answered
        if num_rag_documents == 0:
            return False
        
        # If answer is empty or very short, question is not answered
        if not answer or len(answer.strip()) < self.min_answer_length:
            return False
        
        # Load user prompt from YAML
        evaluation_user_prompt_template = self._agent_prompts.get("evaluate_answer_user_prompt")
        if evaluation_user_prompt_template is None:
            raise ValueError(
                "Missing required prompt in config/prompts.yaml: agents.research_manager.evaluate_answer_user_prompt. "
                "All system prompts must be defined in the config file."
            )
        
        # Format user prompt with dynamic content
        evaluation_prompt = evaluation_user_prompt_template.format(
            question=question,
            answer=answer
        )

        evaluation_system_message = self._agent_prompts.get("evaluate_answer")
        if evaluation_system_message is None:
            raise ValueError(
                "Missing required prompt in config/prompts.yaml: agents.research_manager.evaluate_answer. "
                "All system prompts must be defined in the config file."
            )

        try:
            # Call LLM for evaluation
            evaluation_response = self._generate_fn(
                system_prompt=evaluation_system_message,
                prompt=evaluation_prompt,
                temperature=temperature,
                method_name="_is_question_answered"
            )
            
            # Parse the response to extract 1 or 0
            evaluation_response = evaluation_response.strip()
            
            # Look for 1 or 0 in the response
            if evaluation_response and evaluation_response[0] in ['1', '0']:
                return evaluation_response[0] == '1'
            
            first_one = evaluation_response.find('1')
            first_zero = evaluation_response.find('0')
            
            if first_one != -1 and (first_zero == -1 or first_one < first_zero):
                return True
            elif first_zero != -1:
                return False
            else:
                # Fallback: if response doesn't contain 1 or 0, use heuristics
                if len(evaluation_response) < self.min_evaluation_response_length:
                    return False
                return False
                
        except Exception as e:
            # If LLM evaluation fails, fall back to basic heuristics
            print(f"      Warning: LLM evaluation failed ({str(e)}), using fallback")
            return len(answer.strip()) >= self.min_answer_length_answered and num_rag_documents > 0
    
    def summarize_rejection_lessons(
        self,
        rejection_details: List[Dict[str, Any]],
        temperature: float = None,
        **kwargs
    ) -> str:
        """
        Summarize lessons learned from rejected materials using LLM.
        
        Args:
            rejection_details: List of rejection entry dictionaries, each containing:
                - "candidate": Material name
                - "constraints": List of violated constraints
                - "reason": Reason for rejection
                - "timestamp": Timestamp of rejection
            temperature: Temperature for LLM generation (default: None, uses config value)
            **kwargs: Additional arguments passed to generate function
        
        Returns:
            str: Concise summary (200-300 words) highlighting:
                - Common constraint violations
                - Material characteristics to avoid
                - Guidance for next proposal
        """
        # Use config default if not provided
        if temperature is None:
            temperature = self.temperature
        
        if not rejection_details:
            return ""
        
        # Load user prompt template from YAML
        summarize_user_prompt_template = self._agent_prompts.get("summarize_rejection_lessons_user_prompt")
        if summarize_user_prompt_template is None:
            raise ValueError(
                "Missing required prompt in config/prompts.yaml: agents.research_manager.summarize_rejection_lessons_user_prompt. "
                "All system prompts must be defined in the config file."
            )
        
        # Build rejection details string
        rejection_details_parts = []
        for i, entry in enumerate(rejection_details, 1):
            candidate = entry.get("candidate", "Unknown")
            constraints = entry.get("constraints", [])
            reason = entry.get("reason", "")
            
            rejection_details_parts.append(f"Rejected Material {i}: {candidate}")
            if constraints:
                rejection_details_parts.append(f"  Constraints Violated:")
                for constraint in constraints:
                    rejection_details_parts.append(f"    - {constraint}")
            if reason:
                rejection_details_parts.append(f"  Reason: {reason}")
            rejection_details_parts.append("")
        
        rejection_details_str = "\n".join(rejection_details_parts)
        
        # Format user prompt with dynamic content
        prompt = summarize_user_prompt_template.format(
            rejection_details=rejection_details_str
        )
        
        # Use specialized system message for rejection summarization
        summarization_system_message = self._agent_prompts.get("summarize_rejection_lessons")
        if summarization_system_message is None:
            raise ValueError(
                "Missing required prompt in config/prompts.yaml: agents.research_manager.summarize_rejection_lessons. "
                "All system prompts must be defined in the config file."
            )
        
        # Generate summary
        summary = self._generate_fn(
            system_prompt=summarization_system_message,
            prompt=prompt,
            temperature=temperature,
            method_name="summarize_rejection_lessons",
            **kwargs
        )
        
        return summary.strip()
    
    def propose_candidate(
        self,
        property_mapping: Dict[str, Any],
        application_Y: str,
        rejected_candidates: List[str] = None,
        rejection_lessons: Optional[str] = None,
        material_db=None,
        temperature: float = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Propose a candidate material Z_i based on property mapping and KG insights.
        
        Args:
            property_mapping: Result from ResearchScientist.map_properties_to_materials()
            application_Y: Target application description
            rejected_candidates: List of material names that have been rejected (to avoid repetition)
            rejection_lessons: Optional summary of lessons learned from previous rejections
            temperature: Temperature for LLM generation (default: None, uses config value)
            **kwargs: Additional arguments passed to generate function
        
        Returns:
            Dict containing:
                - "material_name": Proposed material name
                - "justification": Reasoning for the proposal
                - "kg_nodes": List of KG node IDs that support this candidate
        """
        # Use config default if not provided
        if temperature is None:
            temperature = self.temperature
        
        if rejected_candidates is None:
            rejected_candidates = []
        
        # Load user prompt template from YAML
        propose_user_prompt_template = self._agent_prompts.get("propose_candidate_user_prompt")
        if propose_user_prompt_template is None:
            raise ValueError(
                "Missing required prompt in config/prompts.yaml: agents.research_manager.propose_candidate_user_prompt. "
                "All system prompts must be defined in the config file."
            )
        
        # Build property list
        property_mappings = property_mapping.get("property_mappings", [])
        property_list_parts = []
        for pm in property_mappings:
            prop_name = pm.get("property", "")
            target_val = pm.get("target_value", None)
            if target_val:
                property_list_parts.append(f"  - {prop_name}: {target_val}")
            else:
                property_list_parts.append(f"  - {prop_name}")
        property_list = "\n".join(property_list_parts)
        
        # Build KG insights section (key relationships only)
        material_classes = property_mapping.get("material_classes", [])
        kg_insights_section = ""
        kg_insights = property_mapping.get("kg_insights", {})
        if kg_insights:
            found_paths = kg_insights.get('found_paths', [])
            if found_paths:
                formatted_paths = self._format_kg_paths(found_paths)
                if formatted_paths:
                    kg_parts = [f"\nKnowledge Graph Insights:"]
                    kg_parts.append(formatted_paths)
                    # kg_parts.append("\nUse these insights to propose a specific material candidate.")
                    kg_insights_section = "\n".join(kg_parts)
        
        # Build rejection lessons section
        rejection_lessons_section = ""
        if rejection_lessons:
            rejection_lessons_section = f"\nLessons Learned from Previous Rejections:\n{rejection_lessons}\n\nUse these insights to avoid similar issues when proposing the next candidate."
        
        # Build rejected candidates section
        rejected_candidates_section = ""
        if rejected_candidates:
            rejected_parts = ["\nPreviously Rejected Candidates (DO NOT propose these):"]
            for rejected in rejected_candidates:
                rejected_parts.append(f"  - {rejected}")
            rejected_candidates_section = "\n".join(rejected_parts)
        
        # Build rejection lessons task line
        rejection_lessons_task = ""
        if rejection_lessons:
            rejection_lessons_task = "6. Incorporates the lessons learned from previous rejections"

        # Build material database section
        material_database_section = ""
        if material_db is not None:
            all_materials = material_db.get_all_materials()
            if all_materials:
                db_parts = ["\nAvailable Materials Database (Lab Inventory):"]
                for i, mat in enumerate(all_materials, 1):
                    db_parts.append(f"  {i}. {mat['material_name']} (ID: {mat['material_id']})")
                db_parts.append("\nYou may select one of these as your substitute, or combine one with a material from the retrieved sources above.")
                material_database_section = "\n".join(db_parts)

        # Format user prompt with dynamic content
        prompt = propose_user_prompt_template.format(
            application_Y=application_Y,
            property_list=property_list,
            kg_insights_section=kg_insights_section,
            material_database_section=material_database_section,
            rejection_lessons_section=rejection_lessons_section,
            rejected_candidates_section=rejected_candidates_section,
            rejection_lessons_task=rejection_lessons_task
        )
        
        # Use specialized system message for candidate proposal
        proposal_system_message = self._agent_prompts.get("propose_candidate")
        if proposal_system_message is None:
            raise ValueError(
                "Missing required prompt in config/prompts.yaml: agents.research_manager.propose_candidate. "
                "All system prompts must be defined in the config file."
            )
        
        # Safety: truncate prompt if it exceeds token budget
        prompt = self._truncate_prompt(prompt, label="propose_candidate prompt")
        
        print("=" * 70)
        print("For code tweaking purposes, we will now print the prompt for the propose candidate step and then the generated content.")
        print("=" * 70)
        print(f"System prompt:\n{proposal_system_message}\n")
        print(f"Prompt:\n{prompt}")
        print("=" * 70)
        
        # Generate proposal
        content = self._generate_fn(
            system_prompt=proposal_system_message,
            prompt=prompt,
            temperature=temperature,
            method_name="propose_candidate",
            **kwargs
        )
        
        print(f"Generated content:\n{content}")
        print("=" * 70)
        
        # Parse the response to extract structured information
        import re
        material_name = None
        justification = content
        
        # Try to extract structured fields with markdown handling
        lines = content.split('\n')
        for i, line in enumerate(lines):
            # Strip markdown formatting for matching
            line_clean = re.sub(r'\*+', '', line).strip()  # Remove markdown bold/italic markers
            line_lower = line_clean.lower()
            
            # Match various patterns: "Material Name:", "**Material Name:**", "Material:", "Candidate:", etc.
            material_patterns = [
                r'^material\s+name\s*:',
                r'^material\s*:',
                r'^candidate\s*:',
                r'^proposed\s+material\s*:',
                r'^material\s+candidate\s*:'
            ]
            
            for pattern in material_patterns:
                if re.match(pattern, line_lower):
                    # Extract material name after the colon
                    parts = line.split(':', 1)
                    if len(parts) > 1:
                        material_name = parts[1].strip()
                        # Clean markdown from material name
                        material_name = re.sub(r'\*+', '', material_name).strip()
                        break
            
            if material_name:
                break
            
            # Check for justification
            if line_lower.startswith('justification:'):
                justification = '\n'.join(lines[i:]).split(':', 1)[1].strip()
                break
        
        # If parsing failed, try to extract material name from first line or inline patterns
        if not material_name:
            # Try to find inline patterns like "**Material Name:** X" in first few lines
            for line in lines[:5]:
                # Match patterns like "**Material Name:** X" or "*Material Name*: X"
                inline_match = re.search(r'(?:\*+\s*)?(?:material\s+name|material|candidate)\s*:?\s*\*?\s*:?\s*\*?\s*([^*\n]+)', line, re.IGNORECASE)
                if inline_match:
                    material_name = inline_match.group(1).strip()
                    material_name = re.sub(r'\*+', '', material_name).strip()
                    break
            
            # Fallback: extract from first line, cleaning markdown
            if not material_name:
                first_line = lines[0].strip() if lines else ""
                # Remove markdown formatting
                first_line_clean = re.sub(r'\*+', '', first_line).strip()
                # Check if it looks like a material name (reasonable length, not too long)
                if first_line_clean and len(first_line_clean) < self.max_candidate_name_length and not first_line_clean.lower().startswith(('material', 'candidate', 'proposed', 'justification')):
                    material_name = first_line_clean
        
        # Extract KG nodes relevant to the *proposed* candidate (not just top generic classes)
        kg_node_ids = []
        if material_classes and material_name:
            candidate_lower = (material_name or "").lower()
            # First pass: find nodes whose name/title matches the proposed candidate
            for mc in material_classes:
                node_id = mc.get("node_id", "")
                node_data = mc.get("node_data", {})
                node_name = (
                    node_data.get("material_name")
                    or node_data.get("title")
                    or node_data.get("name")
                    or str(node_id)
                )
                if candidate_lower in node_name.lower() or node_name.lower() in candidate_lower:
                    kg_node_ids.append(node_id)
            # If no name match, fall back to first 3 material classes (better than all)
            if not kg_node_ids:
                kg_node_ids = [mc.get("node_id", "") for mc in material_classes[:3] if mc.get("node_id")]
        
        # Final cleaning pass on material_name using shared utility
        material_name = clean_material_name(material_name) or "Unknown"

        return {
            "material_name": material_name,
            "justification": justification,
            "kg_nodes": kg_node_ids[:self.max_kg_nodes_in_context]
        }
    
    def generate_validation_queries(
        self,
        candidate_Z: Dict[str, Any],
        properties_W: Dict[str, Any],
        constraints_U: List[str],
        temperature: float = None,
        kg_context: Optional[Dict[str, Any]] = None,
        subgraph: Optional[nx.DiGraph] = None,
        node_embeddings: Optional[Dict[str, Any]] = None,
        embedding_model=None,
        embedding_tokenizer=None,
        **kwargs
    ) -> List[str]:
        """
        Generate structured validation queries SQ_i to test candidate feasibility.
        
        Args:
            candidate_Z: Candidate material proposal from propose_candidate()
            properties_W: Property requirements dictionary
            constraints_U: List of constraint strings
            temperature: Temperature for LLM generation (default: None, uses config value)
            kg_context: Optional knowledge graph context for the candidate
            subgraph: Optional NetworkX subgraph for finding paths
            node_embeddings: Optional node embeddings dictionary for keyword matching
            embedding_model: Optional embedding model for keyword matching
            embedding_tokenizer: Optional embedding tokenizer for keyword matching
            **kwargs: Additional arguments passed to generate function
        
        Returns:
            List[str]: List of validation queries
        """
        # Use config default if not provided
        if temperature is None:
            temperature = self.temperature
        
        # Load user prompt template from YAML
        validation_queries_user_prompt_template = self._agent_prompts.get("generate_validation_queries_user_prompt")
        if validation_queries_user_prompt_template is None:
            raise ValueError(
                "Missing required prompt in config/prompts.yaml: agents.research_manager.generate_validation_queries_user_prompt. "
                "All system prompts must be defined in the config file."
            )
        
        # Build property list
        required_properties = properties_W.get("required", [])
        target_values = properties_W.get("target_values", {})
        property_list_parts = []
        for prop in required_properties:
            if prop in target_values:
                property_list_parts.append(f"  - {prop}: {target_values[prop]}")
            else:
                property_list_parts.append(f"  - {prop}")
        property_list = "\n".join(property_list_parts)
        
        # Build constraints section
        constraints_section = ""
        if constraints_U:
            constraint_parts = ["\nConstraints:"]
            for constraint in constraints_U:
                constraint_parts.append(f"  - {constraint}")
            constraints_section = "\n".join(constraint_parts)
        
        # Build KG context section
        kg_context_section = ""
        kg_task = ""
        kg_parts = []
        
        # Extract keywords from candidate name, justification, and properties
        candidate_name = candidate_Z.get('material_name', '')
        justification = candidate_Z.get('justification', '')
        required_properties = properties_W.get("required", [])
        
        # Combine text for keyword extraction
        text_for_keywords = f"{candidate_name} {justification} {' '.join(required_properties)}"
        keywords = self._extract_keywords_from_text(text_for_keywords, max_keywords=5)
        
        # Find paths in subgraph if available
        found_paths = []
        if subgraph is not None and node_embeddings is not None and embedding_model is not None and embedding_tokenizer is not None:
            if keywords and len(keywords) >= 2:
                try:
                    # Get similarity threshold and n_samples from config if available
                    config = load_config()
                    scientist_config = config.get("agents", {}).get("research_scientist", {})
                    similarity_threshold = scientist_config.get("similarity_threshold", 0.8)
                    n_samples = scientist_config.get("n_samples", 5)
                    
                    found_paths = self._find_paths_in_subgraph(
                        keywords=keywords,
                        subgraph=subgraph,
                        node_embeddings=node_embeddings,
                        embedding_tokenizer=embedding_tokenizer,
                        embedding_model=embedding_model,
                        n_samples=n_samples,
                        similarity_threshold=similarity_threshold
                    )
                except Exception as e:
                    warnings.warn(f"Error finding paths in subgraph for validation queries: {e}", UserWarning)
        
        # Build KG context section
        if kg_context or found_paths:
            kg_parts = ["\nKnowledge Graph Context:"]
            
            if kg_context:
                material_nodes = kg_context.get('material_nodes', [])
                if material_nodes:
                    kg_parts.append(f"  - Candidate material appears in KG with {len(material_nodes)} connected nodes")
                    if len(material_nodes) > 0:
                        nodes_str = ', '.join([str(n)[:30] for n in material_nodes[:3]])
                        kg_parts.append(f"  - Sample connected nodes: {nodes_str}")
            
            # Add formatted paths if found
            if found_paths:
                formatted_paths = self._format_kg_paths(found_paths)
                if formatted_paths:
                    kg_parts.append(formatted_paths)
            
            if kg_parts:
                kg_parts.append("  - Consider KG relationships when formulating validation questions")
                kg_context_section = "\n".join(kg_parts)
                kg_task = "4. Consider KG relationships and connections when formulating questions"
        
        # Format user prompt with dynamic content
        prompt = validation_queries_user_prompt_template.format(
            candidate_name=candidate_Z.get('material_name', 'Unknown'),
            justification=candidate_Z.get('justification', '')[:500],
            property_list=property_list,
            constraints_section=constraints_section,
            kg_context_section=kg_context_section,
            kg_task=kg_task
        )
        
        # Use system message for query generation
        query_system_message = self._agent_prompts.get("generate_validation_queries")
        if query_system_message is None:
            raise ValueError(
                "Missing required prompt in config/prompts.yaml: agents.research_manager.generate_validation_queries. "
                "All system prompts must be defined in the config file."
            )
        
        # Generate queries
        content = self._generate_fn(
            system_prompt=query_system_message,
            prompt=prompt,
            temperature=temperature,
            method_name="generate_validation_queries",
            **kwargs
        )
        
        # Parse into list
        queries = self._parse_to_list(content)

        # Post-process: strip residual markdown and filter out section headers.
        # The LLM sometimes emits a structured document with bold topic headers
        # (e.g. "**Chemical Resistance to Hydrocarbons**") interleaved with the
        # actual questions.  We keep only items that look like real queries.
        _QUESTION_RE = re.compile(
            r"\?|"
            r"^\s*(what|how|which|does|is|are|can|will|should|according|"
            r"provide|quantify|determine|compare|evaluate|measure|test|"
            r"describe|identify|report|find|verify)\b",
            re.IGNORECASE,
        )
        cleaned: List[str] = []
        for q in queries:
            raw = q.strip()
            # Strip bold/italic markdown wrappers (e.g. **text** or *text*)
            stripped = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", raw).strip()
            # If stripping removed ALL content, skip
            if not stripped:
                continue
            # Heuristic: if the raw item was entirely a bold heading (no verb /
            # question mark) treat it as a section header and discard it.
            if not _QUESTION_RE.search(stripped):
                continue
            cleaned.append(stripped)

        # Fall back to original list if post-processing removed everything
        if cleaned:
            queries = cleaned

        # Limit to reasonable number
        return queries[:6]
    
    def validate_feasibility(
        self,
        candidate_Z: Dict[str, Any],
        evidence_I: List[Dict[str, Any]],
        properties_W: Dict[str, Any],
        constraints_U: List[str],
        temperature: float = None,
        kg_evidence: Optional[Dict[str, Any]] = None,
        subgraph: Optional[nx.DiGraph] = None,
        node_embeddings: Optional[Dict[str, Any]] = None,
        embedding_model=None,
        embedding_tokenizer=None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Validate candidate feasibility based on retrieved evidence.
        
        Args:
            candidate_Z: Candidate material proposal
            evidence_I: List of evidence dictionaries, each containing:
                - "query": The validation query
                - "rag_results": List of retrieved documents
                - "answer": Answer to the query (optional)
            properties_W: Property requirements dictionary
            constraints_U: List of constraint strings
            temperature: Temperature for LLM evaluation (default: None, uses config value)
            kg_evidence: Optional knowledge graph evidence (paths, connections, relationships)
            subgraph: Optional NetworkX subgraph for finding paths
            node_embeddings: Optional node embeddings dictionary for keyword matching
            embedding_model: Optional embedding model for keyword matching
            embedding_tokenizer: Optional embedding tokenizer for keyword matching
            **kwargs: Additional arguments passed to generate function
        
        Returns:
            Dict containing:
                - "is_feasible": bool indicating if candidate is feasible
                - "constraints_violated": List of violated constraints
                - "reasoning": Detailed reasoning for the decision
        """
        # Use config default if not provided
        if temperature is None:
            temperature = self.temperature
        
        # Load user prompt template from YAML
        validate_feasibility_user_prompt_template = self._agent_prompts.get("validate_feasibility_user_prompt")
        if validate_feasibility_user_prompt_template is None:
            raise ValueError(
                "Missing required prompt in config/prompts.yaml: agents.research_manager.validate_feasibility_user_prompt. "
                "All system prompts must be defined in the config file."
            )
        
        # Build evidence list
        evidence_list_parts = []
        for i, ev in enumerate(evidence_I, 1):
            query = ev.get("query", "")
            answer = ev.get("answer", "")
            num_docs = len(ev.get("rag_results", []))
            
            evidence_list_parts.append(f"\n[Evidence {i}]")
            evidence_list_parts.append(f"Query: {query}")
            if answer:
                evidence_list_parts.append(f"Answer: {answer[:300]}{'...' if len(answer) > 300 else ''}")
            evidence_list_parts.append(f"Documents Retrieved: {num_docs}")
        evidence_list = "\n".join(evidence_list_parts)
        
        # Extract keywords from validation queries
        all_query_texts = []
        for ev in evidence_I:
            query = ev.get("query", "")
            if query:
                all_query_texts.append(query)
        
        # Combine all queries for keyword extraction
        combined_queries_text = " ".join(all_query_texts)
        keywords = self._extract_keywords_from_text(combined_queries_text, max_keywords=5)
        
        # Find paths in subgraph if available
        found_paths = []
        if subgraph is not None and node_embeddings is not None and embedding_model is not None and embedding_tokenizer is not None:
            if keywords and len(keywords) >= 2:
                try:
                    # Get similarity threshold and n_samples from config if available
                    config = load_config()
                    scientist_config = config.get("agents", {}).get("research_scientist", {})
                    similarity_threshold = scientist_config.get("similarity_threshold", 0.8)
                    n_samples = scientist_config.get("n_samples", 5)
                    
                    found_paths = self._find_paths_in_subgraph(
                        keywords=keywords,
                        subgraph=subgraph,
                        node_embeddings=node_embeddings,
                        embedding_tokenizer=embedding_tokenizer,
                        embedding_model=embedding_model,
                        n_samples=n_samples,
                        similarity_threshold=similarity_threshold
                    )
                except Exception as e:
                    warnings.warn(f"Error finding paths in subgraph for feasibility validation: {e}", UserWarning)
        
        # Build KG evidence section
        kg_evidence_section = ""
        kg_consideration = ""
        kg_parts = []
        
        if kg_evidence or found_paths:
            kg_parts = ["\nKnowledge Graph Evidence:"]
            
            if kg_evidence:
                paths_found = kg_evidence.get('paths_found', 0)
                nodes_connected = kg_evidence.get('nodes_connected', [])
                if paths_found > 0:
                    kg_parts.append(f"  - Found {paths_found} paths connecting candidate to required properties in KG")
                    if nodes_connected:
                        nodes_str = ', '.join([str(n)[:30] for n in nodes_connected[:5]])
                        kg_parts.append(f"  - Connected nodes: {nodes_str}")
                elif not found_paths:
                    kg_parts.append("  - No direct paths found in knowledge graph (may indicate limited KG coverage)")
            
            # Add formatted paths if found
            if found_paths:
                formatted_paths = self._format_kg_paths(found_paths)
                if formatted_paths:
                    kg_parts.append(formatted_paths)
            
            if kg_parts:
                kg_evidence_section = "\n".join(kg_parts)
                kg_consideration = "5. Consider KG evidence: paths and connections in the knowledge graph"
        
        # Build property list
        required_properties = properties_W.get("required", [])
        target_values = properties_W.get("target_values", {})
        property_list_parts = []
        for prop in required_properties:
            if prop in target_values:
                property_list_parts.append(f"  - {prop}: {target_values[prop]}")
            else:
                property_list_parts.append(f"  - {prop}")
        property_list = "\n".join(property_list_parts)
        
        # Build constraints section
        constraints_section = ""
        if constraints_U:
            constraint_parts = ["\nConstraints:"]
            for constraint in constraints_U:
                constraint_parts.append(f"  - {constraint}")
            constraints_section = "\n".join(constraint_parts)
        
        # Format user prompt with dynamic content
        prompt = validate_feasibility_user_prompt_template.format(
            candidate_name=candidate_Z.get('material_name', 'Unknown'),
            evidence_list=evidence_list,
            kg_evidence_section=kg_evidence_section,
            property_list=property_list,
            constraints_section=constraints_section,
            kg_consideration=kg_consideration
        )
        
        # Use system message for feasibility validation
        feasibility_system_message = self._agent_prompts.get("validate_feasibility")
        if feasibility_system_message is None:
            raise ValueError(
                "Missing required prompt in config/prompts.yaml: agents.research_manager.validate_feasibility. "
                "All system prompts must be defined in the config file."
            )
        
        # Safety: truncate prompt if it exceeds token budget
        prompt = self._truncate_prompt(prompt, label="validate_feasibility prompt")
        
        # Generate evaluation
        content = self._generate_fn(
            system_prompt=feasibility_system_message,
            prompt=prompt,
            temperature=temperature,
            method_name="validate_feasibility",
            **kwargs
        )
        
        # Parse response
        is_feasible = False
        constraints_violated = []
        reasoning = content
        
        # First, try to parse structured format (with or without markdown)
        lines = content.split('\n')
        for line in lines:
            # Remove markdown formatting (**, *, etc.) for parsing
            line_clean = line.replace('**', '').replace('*', '').strip()
            if not line_clean:
                continue
            line_upper = line_clean.upper()
            
            if 'FEASIBLE:' in line_upper:
                # Extract text after FEASIBLE:
                parts = line_clean.split(':', 1)
                if len(parts) > 1:
                    feasible_text = parts[1].strip().upper()
                    is_feasible = feasible_text in ['YES', 'TRUE', '1']
            if 'CONSTRAINTS_VIOLATED:' in line_upper:
                parts = line_clean.split(':', 1)
                if len(parts) > 1:
                    constraints_text = parts[1].strip()
                    if constraints_text.upper() not in ['NONE', 'N/A', '']:
                        constraints_violated = [c.strip() for c in constraints_text.split(',') if c.strip()]
            if 'REASONING:' in line_upper:
                parts = line_clean.split(':', 1)
                if len(parts) > 1:
                    reasoning = parts[1].strip()
        
        # Fallback: if we didn't find FEASIBLE in structured format, try to infer from content
        if not any('FEASIBLE:' in line.replace('**', '').replace('*', '').upper() for line in lines):
            # Look for explicit YES/NO in the content
            content_upper = content.upper()
            if 'FEASIBLE' in content_upper and 'YES' in content_upper:
                # Check if YES appears near FEASIBLE
                feasible_idx = content_upper.find('FEASIBLE')
                yes_idx = content_upper.find('YES', feasible_idx)
                if yes_idx != -1 and yes_idx - feasible_idx < self.yes_feasible_proximity_chars:
                    is_feasible = True
        
        # Extract constraints from reasoning if not explicitly listed
        if not constraints_violated and not is_feasible:
            # Try to infer constraints from reasoning
            for constraint in constraints_U:
                if constraint.lower() in reasoning.lower():
                    constraints_violated.append(constraint)
        
        return {
            "is_feasible": is_feasible,
            "constraints_violated": constraints_violated,
            "reasoning": reasoning
        }

    def generate_process_retrieval_queries(
        self,
        candidate_name: str,
        material_class: str,
        application_Y: str,
        justification: str = "",
        max_queries: int = None,
        temperature: float = None,
        process_analyst=None,
        **kwargs
    ) -> List[str]:
        """
        Generate exactly 4 process-oriented search queries for RAG retrieval using the LLM.
        First performs initial RAG retrieval to gather context about the material and manufacturing,
        then uses that context to inform query generation.
        Returns list of exactly 4 query strings.
        
        Raises:
            ValueError: If candidate_name is missing, empty, or "Unknown", or if process_analyst is None
            RuntimeError: If RAG retrieval fails or LLM generation fails
            ValueError: If JSON parsing fails or insufficient queries are generated
        """
        # Use config defaults if not provided
        if max_queries is None:
            max_queries = self.max_queries
        if temperature is None:
            temperature = self.temperature
        
        # Step 0: Strict input validation
        if not candidate_name or candidate_name.strip() == "" or candidate_name.strip().lower() == "unknown":
            raise ValueError(
                f"candidate_name is required and must be non-empty and not 'Unknown', got: {repr(candidate_name)}"
            )
        
        if process_analyst is None:
            raise ValueError("process_analyst is required but was None")
        
        # Step 1: Perform initial RAG retrieval to gather context
        # No try-except: let exceptions propagate - RAG failure should crash
        initial_query = f"Manufacturing and synthesis processes for {candidate_name}"
        if material_class:
            initial_query += f" ({material_class})"
        if application_Y:
            initial_query += f" for {application_Y} applications"
        
        rag_result = process_analyst.analyze_question(initial_query)
        initial_rag_results = rag_result.get("rag_results", [])[:self.max_rag_results_in_context]
        
        # Step 2: Format RAG context if available
        rag_context = ""
        if initial_rag_results:
            formatted_rag = self._format_rag_context(initial_rag_results, max_chars_per_result=self.max_chars_per_result_validation)
            rag_context = f"\n\nInitial Context from Knowledge Sources:\n{formatted_rag}\n\nUse this context to inform what specific queries would be most valuable to retrieve more detailed information."
        
        # Step 3: Load prompt and format with context
        prompts = load_prompts()
        prompt_config = prompts.get("pipelines", {}).get("manufacturability_assessment", {})
        system_prompt = prompt_config.get("process_retrieval_queries")
        if system_prompt is None:
            raise RuntimeError(
                "Missing required prompt in config/prompts.yaml: "
                "pipelines.manufacturability_assessment.process_retrieval_queries"
            )
        
        user_prompt = system_prompt.format(
            candidate_name=candidate_name,
            material_class=material_class or "",
            application_Y=application_Y or "",
            justification=justification or "",
            rag_context=rag_context
        )
        
        # Step 4: Generate queries using LLM
        # No try-except: let exceptions propagate - LLM failure should crash
        self._log_prompt_length(user_prompt, label="generate_process_retrieval_queries prompt")
        content = self._generate_fn(
            system_prompt="",
            prompt=user_prompt,
            temperature=temperature,
            method_name="generate_process_retrieval_queries",
            **kwargs
        )
        
        if not content or not content.strip():
            raise RuntimeError("LLM generate_process_retrieval_queries returned empty or None content")
        
        # Step 5: Parse JSON response
        content_clean = content.strip()
        queries = []
        
        json_start = content_clean.find("{")
        json_end = content_clean.rfind("}") + 1
        if json_start < 0 or json_end <= json_start:
            raise ValueError(
                f"Failed to find JSON object in LLM response. Response: {content_clean[:500]}"
            )
        
        try:
            data = json.loads(content_clean[json_start:json_end])
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse JSON from LLM response: {e}. Response: {content_clean[:500]}"
            ) from e
        
        # Extract required field - raise KeyError if missing (no defaults)
        try:
            raw = data["queries"]
        except KeyError:
            raise KeyError(
                f"Missing required field 'queries' in JSON response from generate_process_retrieval_queries. "
                f"Available keys: {list(data.keys())}"
            )
        
        if not isinstance(raw, list):
            raise ValueError(f"Expected 'queries' to be a list, got {type(raw)}: {raw}")
        
        queries = [str(q).strip() for q in raw if q and str(q).strip()]
        
        # Step 6: Validate exactly max_queries (default 4) queries
        if len(queries) < max_queries:
            raise ValueError(
                f"Expected exactly {max_queries} queries, but only got {len(queries)}: {queries}"
            )
        elif len(queries) > max_queries:
            # If we have more, take the first max_queries
            queries = queries[:max_queries]
        
        return queries

    def extract_material_constituents_for_manufacturing(
        self,
        candidate_Z: Dict[str, Any],
        application_Y: str = "",
        constraints_U: Optional[List[str]] = None,
        temperature: float = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Extract constituent materials and combination modes for manufacturing retrieval."""
        if temperature is None:
            temperature = self.temperature

        material_name = candidate_Z.get("material_name")
        if not material_name or material_name.strip() == "" or material_name.strip().lower() == "unknown":
            raise ValueError(
                f"candidate_Z must have a valid 'material_name' field (non-empty and not 'Unknown'), "
                f"got: {repr(material_name)}"
            )

        material_class = candidate_Z.get("material_class", "")
        justification = candidate_Z.get("justification", "")
        constraints_str = "\n".join(f"  - {c}" for c in (constraints_U or []))

        prompts = load_prompts()
        prompt_config = prompts.get("pipelines", {}).get("manufacturability_assessment", {})
        system_prompt = prompt_config.get("extract_material_constituents")
        if system_prompt is None:
            raise ValueError(
                "Missing required prompt in config/prompts.yaml: "
                "pipelines.manufacturability_assessment.extract_material_constituents"
            )
        user_prompt_template = prompt_config.get("extract_material_constituents_user_prompt")
        if user_prompt_template is None:
            raise RuntimeError(
                "Missing required prompt in config/prompts.yaml: "
                "pipelines.manufacturability_assessment.extract_material_constituents_user_prompt"
            )

        user_prompt = user_prompt_template.format(
            material_name=material_name,
            material_class=material_class or "",
            application_Y=application_Y or "",
            justification=justification or "",
            constraints=constraints_str,
        )
        self._log_prompt_length(user_prompt, label="extract_material_constituents prompt")

        content = self._generate_fn(
            system_prompt=system_prompt,
            prompt=user_prompt,
            temperature=temperature,
            method_name="extract_material_constituents_for_manufacturing",
            **kwargs
        )
        if not content or not content.strip():
            raise RuntimeError("LLM extract_material_constituents_for_manufacturing returned empty content")

        content_clean = content.strip()
        json_start = content_clean.find("{")
        json_end = content_clean.rfind("}") + 1
        if json_start < 0 or json_end <= json_start:
            raise ValueError(
                "Failed to find JSON object in LLM response for "
                "extract_material_constituents_for_manufacturing."
            )
        try:
            data = json.loads(content_clean[json_start:json_end])
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse JSON from extract_material_constituents_for_manufacturing: {e}"
            ) from e

        try:
            is_composite = data["is_composite"]
            constituents = data["constituents"]
            composition_notes = data["composition_notes"]
            combination_modes = data["combination_modes"]
        except KeyError as e:
            raise KeyError(
                f"Missing required field '{e.args[0]}' in extract_material_constituents_for_manufacturing. "
                f"Available keys: {list(data.keys())}"
            ) from e

        if not isinstance(is_composite, bool):
            raise ValueError(
                f"extract_material_constituents_for_manufacturing: 'is_composite' must be bool, got {type(is_composite)}"
            )
        if not isinstance(constituents, list):
            raise ValueError(
                f"extract_material_constituents_for_manufacturing: 'constituents' must be list, got {type(constituents)}"
            )
        if not isinstance(combination_modes, list):
            raise ValueError(
                f"extract_material_constituents_for_manufacturing: 'combination_modes' must be list, got {type(combination_modes)}"
            )
        if composition_notes is None:
            composition_notes = ""
        if not isinstance(composition_notes, str):
            composition_notes = str(composition_notes)

        normalized_constituents = []
        seen = set()
        for c in constituents:
            c_str = str(c).strip()
            if not c_str:
                continue
            c_key = c_str.lower()
            if c_key in seen:
                continue
            seen.add(c_key)
            normalized_constituents.append(c_str)

        if not normalized_constituents:
            normalized_constituents = [material_name]

        normalized_combination_modes = []
        seen_modes = set()
        for mode in combination_modes:
            mode_str = str(mode).strip()
            if not mode_str:
                continue
            mode_key = mode_str.lower()
            if mode_key in seen_modes:
                continue
            seen_modes.add(mode_key)
            normalized_combination_modes.append(mode_str)

        return {
            "is_composite": is_composite,
            "constituents": normalized_constituents,
            "composition_notes": composition_notes,
            "combination_modes": normalized_combination_modes,
        }

    def generate_decomposition_process_queries(
        self,
        decomposition: Dict[str, Any],
        candidate_Z: Dict[str, Any],
        application_Y: str = "",
        temperature: float = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Generate structured retrieval queries for constituents and composite combination modes."""
        if temperature is None:
            temperature = self.temperature

        material_name = candidate_Z.get("material_name")
        if not material_name or material_name.strip() == "" or material_name.strip().lower() == "unknown":
            raise ValueError(
                f"candidate_Z must have a valid 'material_name' field (non-empty and not 'Unknown'), "
                f"got: {repr(material_name)}"
            )
        material_class = candidate_Z.get("material_class", "")
        justification = candidate_Z.get("justification", "")

        prompts = load_prompts()
        prompt_config = prompts.get("pipelines", {}).get("manufacturability_assessment", {})
        system_prompt = prompt_config.get("generate_decomposition_process_queries")
        if system_prompt is None:
            raise ValueError(
                "Missing required prompt in config/prompts.yaml: "
                "pipelines.manufacturability_assessment.generate_decomposition_process_queries"
            )
        user_prompt_template = prompt_config.get("generate_decomposition_process_queries_user_prompt")
        if user_prompt_template is None:
            raise RuntimeError(
                "Missing required prompt in config/prompts.yaml: "
                "pipelines.manufacturability_assessment.generate_decomposition_process_queries_user_prompt"
            )

        decomposition_json = json.dumps(decomposition or {}, indent=2)
        user_prompt = user_prompt_template.format(
            material_name=material_name,
            material_class=material_class or "",
            application_Y=application_Y or "",
            justification=justification or "",
            decomposition_json=decomposition_json,
        )
        self._log_prompt_length(user_prompt, label="generate_decomposition_process_queries prompt")

        content = self._generate_fn(
            system_prompt=system_prompt,
            prompt=user_prompt,
            temperature=temperature,
            method_name="generate_decomposition_process_queries",
            **kwargs
        )
        if not content or not content.strip():
            raise RuntimeError("LLM generate_decomposition_process_queries returned empty content")

        content_clean = content.strip()
        json_start = content_clean.find("{")
        json_end = content_clean.rfind("}") + 1
        if json_start < 0 or json_end <= json_start:
            raise ValueError("Failed to find JSON object in LLM response for generate_decomposition_process_queries")
        try:
            data = json.loads(content_clean[json_start:json_end])
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from generate_decomposition_process_queries: {e}") from e

        raw_queries = data.get("queries")
        if not isinstance(raw_queries, list):
            raise ValueError(
                f"Expected 'queries' to be a list in generate_decomposition_process_queries, got {type(raw_queries)}"
            )

        query_specs: List[Dict[str, Any]] = []
        seen = set()
        for item in raw_queries:
            if not isinstance(item, dict):
                continue
            query = str(item.get("query", "")).strip()
            if not query:
                continue
            query_type = str(item.get("query_type", "constituent")).strip().lower()
            if query_type not in {"constituent", "combination"}:
                raise ValueError(
                    "generate_decomposition_process_queries: query_type must be 'constituent' or 'combination', "
                    f"got {query_type}"
                )
            constituent = str(item.get("constituent", "")).strip()
            is_combination_query = item.get("is_combination_query", query_type == "combination")
            if not isinstance(is_combination_query, bool):
                raise ValueError(
                    "generate_decomposition_process_queries: is_combination_query must be bool, "
                    f"got {type(is_combination_query)}"
                )
            if query_type == "constituent" and not constituent:
                raise ValueError("generate_decomposition_process_queries: constituent query missing constituent field")
            if query_type == "combination":
                constituent = ""
                is_combination_query = True
            key = (query.lower(), query_type, constituent.lower())
            if key in seen:
                continue
            seen.add(key)
            query_specs.append({
                "query": query,
                "query_type": query_type,
                "constituent": constituent,
                "is_combination_query": is_combination_query,
            })

        if not query_specs:
            raise RuntimeError("generate_decomposition_process_queries produced no valid queries")

        return query_specs

    def generate_feasibility_questions(
        self,
        candidate_Z: Dict[str, Any],
        properties_W: Dict[str, Any],
        application_Y: str,
        constraints_U: List[str],
        retrieved_rag_results: List[Dict[str, Any]] = None,
        num_questions: int = 4,
        temperature: float = None,
        evidence_coverage: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[str]:
        """
        Generate feasibility assessment questions for a candidate material using the LLM.

        Uses initial process retrieval evidence as context to inform targeted question generation.

        Args:
            candidate_Z: Candidate material dict with material_name, material_class, etc.
            properties_W: Required properties from System 1
            application_Y: Target application
            constraints_U: User constraints
            retrieved_rag_results: Evidence from Step 1 (process retrieval) used as context
            num_questions: Number of questions to generate (default: 4)
            temperature: LLM temperature

        Returns:
            List of question strings (exactly num_questions)

        Raises:
            ValueError: If candidate_Z is invalid or JSON parsing fails
            RuntimeError: If LLM generation fails
        """
        if temperature is None:
            temperature = self.temperature

        # Validate candidate
        material_name = candidate_Z.get("material_name")
        if not material_name or material_name.strip() == "" or material_name.strip().lower() == "unknown":
            raise ValueError(
                f"candidate_Z must have a valid 'material_name' field (non-empty and not 'Unknown'), "
                f"got: {repr(material_name)}"
            )

        material_class = candidate_Z.get("material_class", "")
        required = properties_W.get("required", []) if properties_W else []
        required_str = ", ".join(required) if required else "None"
        constraints_str = "\n".join(f"  - {c}" for c in (constraints_U or []))

        # Format initial RAG context
        rag_context = ""
        if retrieved_rag_results:
            rag_context = self._format_rag_context(
                retrieved_rag_results, max_chars_per_result=self.max_chars_per_result_feasibility
            )
        else:
            rag_context = "No initial evidence available."

        # Load prompts
        prompts = load_prompts()
        prompt_config = prompts.get("pipelines", {}).get("manufacturability_assessment", {})

        system_prompt = prompt_config.get("generate_feasibility_questions")
        if system_prompt is None:
            raise ValueError(
                "Missing required prompt in config/prompts.yaml: "
                "pipelines.manufacturability_assessment.generate_feasibility_questions"
            )
        system_prompt = system_prompt.format(num_questions=num_questions)

        user_prompt_template = prompt_config.get("generate_feasibility_questions_user_prompt")
        if user_prompt_template is None:
            raise RuntimeError(
                "Missing required prompt in config/prompts.yaml: "
                "pipelines.manufacturability_assessment.generate_feasibility_questions_user_prompt"
            )

        user_prompt = user_prompt_template.format(
            material_name=material_name,
            material_class=material_class or "",
            application_Y=application_Y or "",
            required_properties=required_str,
            constraints=constraints_str,
            rag_context=rag_context,
            evidence_coverage=json.dumps(evidence_coverage or {}, indent=2),
            num_questions=num_questions,
        )
        self._log_prompt_length(user_prompt, label="generate_feasibility_questions prompt")

        # LLM call
        content = self._generate_fn(
            system_prompt=system_prompt,
            prompt=user_prompt,
            temperature=temperature,
            method_name="generate_feasibility_questions",
            **kwargs
        )

        if not content or not content.strip():
            raise RuntimeError("LLM generate_feasibility_questions returned empty or None content")

        # Parse JSON
        content_clean = content.strip()
        json_start = content_clean.find("{")
        json_end = content_clean.rfind("}") + 1
        if json_start < 0 or json_end <= json_start:
            raise ValueError(
                f"Failed to find JSON object in LLM response for generate_feasibility_questions. "
                f"Response: {content_clean[:500]}"
            )

        try:
            data = json.loads(content_clean[json_start:json_end])
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse JSON from LLM response for generate_feasibility_questions: {e}. "
                f"Response: {content_clean[:500]}"
            ) from e

        try:
            raw = data["questions"]
        except KeyError:
            raise KeyError(
                f"Missing required field 'questions' in JSON response from generate_feasibility_questions. "
                f"Available keys: {list(data.keys())}"
            )

        if not isinstance(raw, list):
            raise ValueError(f"Expected 'questions' to be a list, got {type(raw)}: {raw}")

        questions = [str(q).strip() for q in raw if q and str(q).strip()]

        if len(questions) < num_questions:
            raise ValueError(
                f"Expected exactly {num_questions} questions, but only got {len(questions)}: {questions}"
            )
        elif len(questions) > num_questions:
            questions = questions[:num_questions]

        return questions

    def answer_feasibility_question(
        self,
        question: str,
        rag_results: List[Dict[str, Any]],
        candidate_Z: Dict[str, Any],
        temperature: float = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Answer a single feasibility question using retrieved RAG sources.

        Args:
            question: Single feasibility question string
            rag_results: RAG results retrieved specifically for this question
            candidate_Z: Candidate material dict
            temperature: LLM temperature

        Returns:
            Dict with 'answer' (str), 'confidence' (str), 'evidence_used' (list of str)

        Raises:
            ValueError: If inputs are invalid or JSON parsing fails
            RuntimeError: If LLM generation fails
        """
        if temperature is None:
            temperature = self.temperature

        material_name = candidate_Z.get("material_name", "")
        material_class = candidate_Z.get("material_class", "")

        rag_context = self._format_rag_context(
            rag_results, max_chars_per_result=self.max_chars_per_result_feasibility
        )

        # Load prompts
        prompts = load_prompts()
        prompt_config = prompts.get("pipelines", {}).get("manufacturability_assessment", {})

        system_prompt = prompt_config.get("answer_feasibility_question")
        if system_prompt is None:
            raise ValueError(
                "Missing required prompt in config/prompts.yaml: "
                "pipelines.manufacturability_assessment.answer_feasibility_question"
            )

        user_prompt_template = prompt_config.get("answer_feasibility_question_user_prompt")
        if user_prompt_template is None:
            raise RuntimeError(
                "Missing required prompt in config/prompts.yaml: "
                "pipelines.manufacturability_assessment.answer_feasibility_question_user_prompt"
            )

        user_prompt = user_prompt_template.format(
            material_name=material_name,
            material_class=material_class or "",
            question=question,
            rag_context=rag_context,
        )
        self._log_prompt_length(user_prompt, label="answer_feasibility_question prompt")

        # LLM call
        content = self._generate_fn(
            system_prompt=system_prompt,
            prompt=user_prompt,
            temperature=temperature,
            method_name="answer_feasibility_question",
            **kwargs
        )

        if not content or not content.strip():
            raise RuntimeError("LLM answer_feasibility_question returned empty or None content")

        # Parse JSON
        content_clean = content.strip()
        json_start = content_clean.find("{")
        json_end = content_clean.rfind("}") + 1
        if json_start < 0 or json_end <= json_start:
            raise ValueError(
                f"Failed to find JSON object in LLM response for answer_feasibility_question. "
                f"Response: {content_clean[:500]}"
            )

        try:
            data = json.loads(content_clean[json_start:json_end])
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse JSON from LLM response for answer_feasibility_question: {e}. "
                f"Response: {content_clean[:500]}"
            ) from e

        # Extract fields
        try:
            answer = data["answer"]
            confidence = data["confidence"]
            evidence_used = data["evidence_used"]
        except KeyError as e:
            raise KeyError(
                f"Missing required field '{e.args[0]}' in JSON response from answer_feasibility_question. "
                f"Available keys: {list(data.keys())}"
            ) from e

        allowed_confidence = {"high", "medium", "low", "insufficient_evidence"}
        if confidence not in allowed_confidence:
            raise ValueError(
                f"Invalid confidence value from answer_feasibility_question: {confidence}. "
                f"Expected one of {sorted(allowed_confidence)}"
            )
        if not isinstance(answer, str):
            raise ValueError(
                f"answer_feasibility_question: 'answer' must be str, got {type(answer)}"
            )
        if not isinstance(evidence_used, list):
            raise ValueError(
                f"answer_feasibility_question: 'evidence_used' must be list, got {type(evidence_used)}"
            )

        return {
            "answer": answer,
            "confidence": confidence,
            "evidence_used": evidence_used,
        }

    def assess_manufacturability_feasibility(
        self,
        question_answers: List[Dict[str, Any]],
        candidate_Z: Dict[str, Any],
        properties_W: Dict[str, Any],
        application_Y: str,
        constraints_U: List[str],
        temperature: float = None,
        evidence_coverage: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Assess whether the candidate material can be manufactured at lab scale.

        Aggregates question-answer pairs from the feasibility Q&A step into a final
        feasibility decision.

        Args:
            question_answers: List of dicts, each with 'question', 'answer', 'confidence',
                              'evidence_used', and optionally 'num_rag_documents'
            candidate_Z: Candidate material dict
            properties_W: Required properties from System 1
            application_Y: Target application
            constraints_U: User constraints
            temperature: LLM temperature

        Returns:
            Dict with feasible (bool), blocking_constraints (list of dicts), feedback_to_system2 (str).
        """
        # Use config default if not provided
        if temperature is None:
            temperature = self.temperature
        
        prompts = load_prompts()
        prompt_config = prompts.get("pipelines", {}).get("manufacturability_assessment", {})
        system_prompt = prompt_config.get("assess_feasibility")
        if system_prompt is None:
            raise ValueError(
                "Missing required prompt in config/prompts.yaml: pipelines.manufacturability_assessment.assess_feasibility"
            )

        # Validate inputs
        material_name = candidate_Z.get("material_name")
        if not material_name or material_name.strip() == "" or material_name.strip().lower() == "unknown":
            raise ValueError(
                f"candidate_Z must have a valid 'material_name' field (non-empty and not 'Unknown'), "
                f"got: {repr(material_name)}"
            )
        
        material_class = candidate_Z.get("material_class", "")
        required = properties_W.get("required", []) if properties_W else []
        required_str = ", ".join(required) if required else "None"
        constraints_str = "\n".join(f"  - {c}" for c in (constraints_U or []))

        # Format Q&A pairs into a readable string for the prompt
        qa_parts = []
        for i, qa in enumerate(question_answers, 1):
            q = qa.get("question", "")
            a = qa.get("answer", "")
            conf = qa.get("confidence", "unknown")
            evidence = qa.get("evidence_used", [])
            evidence_str = ", ".join(evidence) if evidence else "None"
            qa_parts.append(
                f"Q{i}: {q}\n"
                f"A{i}: {a}\n"
                f"Confidence: {conf}\n"
                f"Evidence used: {evidence_str}"
            )
        question_answers_str = "\n\n".join(qa_parts)
        
        # Load user prompt template
        prompts = load_prompts()
        prompt_config = prompts.get("pipelines", {}).get("manufacturability_assessment", {})
        user_prompt_template = prompt_config.get("assess_feasibility_user_prompt")
        if user_prompt_template is None:
            raise RuntimeError(
                "Missing required prompt in config/prompts.yaml: "
                "pipelines.manufacturability_assessment.assess_feasibility_user_prompt"
            )
        
        user_prompt = user_prompt_template.format(
            material_name=material_name,
            material_class=material_class or "",
            application_Y=application_Y or "",
            required_properties=required_str,
            constraints=constraints_str,
            question_answers=question_answers_str,
            evidence_coverage=json.dumps(evidence_coverage or {}, indent=2),
        )
        self._log_prompt_length(user_prompt, label="assess_manufacturability_feasibility prompt")

        # No try-except: let exceptions propagate - LLM failure should crash
        content = self._generate_fn(
            system_prompt=system_prompt,
            prompt=user_prompt,
            temperature=temperature,
            method_name="assess_manufacturability_feasibility",
            **kwargs
        )
        
        if not content or not content.strip():
            raise RuntimeError("LLM assess_manufacturability_feasibility returned empty or None content")

        feasible = False
        blocking_constraints = []
        feedback_to_system2 = ""

        # Parse JSON - raise exception on failure instead of falling back
        content_clean = content.strip()
        json_start = content_clean.find("{")
        json_end = content_clean.rfind("}") + 1
        if json_start < 0 or json_end <= json_start:
            raise ValueError(
                f"Failed to find JSON object in LLM response for assess_manufacturability_feasibility. "
                f"Response: {content_clean[:500]}"
            )
        
        try:
            data = json.loads(content_clean[json_start:json_end])
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse JSON from LLM response for assess_manufacturability_feasibility: {e}. "
                f"Response: {content_clean[:500]}"
            ) from e
        
        # Extract required fields - raise KeyError if missing (no defaults)
        try:
            feasible = data["feasible"]
            feedback_to_system2 = data["feedback_to_system2"]
            raw_constraints = data["blocking_constraints"]
        except KeyError as e:
            raise KeyError(
                f"Missing required field '{e.args[0]}' in JSON response from assess_manufacturability_feasibility. "
                f"Available keys: {list(data.keys())}"
            ) from e
        
        if not isinstance(feasible, bool):
            raise ValueError(
                "assess_manufacturability_feasibility: 'feasible' must be bool, "
                f"got {type(feasible)}"
            )
        if not isinstance(feedback_to_system2, str):
            raise ValueError(
                "assess_manufacturability_feasibility: 'feedback_to_system2' must be str, "
                f"got {type(feedback_to_system2)}"
            )
        if not isinstance(raw_constraints, list):
            raise ValueError(
                "assess_manufacturability_feasibility: 'blocking_constraints' must be list, "
                f"got {type(raw_constraints)}"
            )

        for c in raw_constraints:
            if isinstance(c, dict):
                # Extract required constraint fields - raise KeyError if missing
                try:
                    blocking_constraints.append({
                        "type": c["type"],
                        "severity": c["severity"],
                        "description": c["description"],
                        "suggested_mitigation": c.get("suggested_mitigation"),  # Optional field
                        "evidence_pointers": c.get("evidence_pointers"),  # Optional field
                    })
                except KeyError as e:
                    raise KeyError(
                        f"Missing required field '{e.args[0]}' in blocking_constraint dict. "
                        f"Available keys: {list(c.keys())}"
                    ) from e
            else:
                blocking_constraints.append({
                    "type": "missing_critical_info",
                    "severity": "hard",
                    "description": str(c),
                    "suggested_mitigation": None,
                    "evidence_pointers": None,
                })
        if feasible and not feedback_to_system2:
            feedback_to_system2 = "Manufacturable at lab scale."

        return {
            "feasible": feasible,
            "blocking_constraints": blocking_constraints,
            "feedback_to_system2": feedback_to_system2,
        }

    def synthesize_process_recipe(
        self,
        retrieved_rag_results: List[Dict[str, Any]],
        candidate_Z: Dict[str, Any],
        application_Y: str,
        temperature: float = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Synthesize a high-level lab-scale process recipe from retrieved evidence.
        Returns process_recipe (list of step dicts), evidence.
        """
        # Use config default if not provided
        if temperature is None:
            temperature = self.temperature
        
        prompts = load_prompts()
        prompt_config = prompts.get("pipelines", {}).get("manufacturability_assessment", {})
        system_prompt = prompt_config.get("synthesize_recipe")
        if system_prompt is None:
            raise ValueError(
                "Missing required prompt in config/prompts.yaml: pipelines.manufacturability_assessment.synthesize_recipe"
            )

        # Validate inputs
        material_name = candidate_Z.get("material_name")
        if not material_name or material_name.strip() == "" or material_name.strip().lower() == "unknown":
            raise ValueError(
                f"candidate_Z must have a valid 'material_name' field (non-empty and not 'Unknown'), "
                f"got: {repr(material_name)}"
            )
        
        rag_context = self._format_rag_context(retrieved_rag_results, max_chars_per_result=self.max_chars_per_result_feasibility)
        material_class = candidate_Z.get("material_class", "")

        # Load user prompt template
        prompts = load_prompts()
        prompt_config = prompts.get("pipelines", {}).get("manufacturability_assessment", {})
        user_prompt_template = prompt_config.get("synthesize_recipe_user_prompt")
        if user_prompt_template is None:
            raise RuntimeError(
                "Missing required prompt in config/prompts.yaml: "
                "pipelines.manufacturability_assessment.synthesize_recipe_user_prompt"
            )
        
        user_prompt = user_prompt_template.format(
            material_name=material_name,
            material_class=material_class,
            application_Y=application_Y or "",
            rag_context=rag_context
        )
        self._log_prompt_length(user_prompt, label="synthesize_process_recipe prompt")

        # No try-except: let exceptions propagate - LLM failure should crash
        content = self._generate_fn(
            system_prompt=system_prompt,
            prompt=user_prompt,
            temperature=temperature,
            method_name="synthesize_process_recipe",
            **kwargs
        )
        
        if not content or not content.strip():
            raise RuntimeError("LLM synthesize_process_recipe returned empty or None content")

        process_recipe = []
        evidence = []

        # Parse JSON (LLMs often wrap output in ```json fences or add prose)
        if isinstance(content, dict):
            data = content
        else:
            if not isinstance(content, str):
                content = str(content)
            data = extract_json_from_response(content)
            if data is None:
                preview = content[:800] + ("…" if len(content) > 800 else "")
                raise ValueError(
                    "Failed to parse JSON from LLM response for synthesize_process_recipe "
                    f"(no valid JSON object). Response preview:\n{preview}"
                )
        if not isinstance(data, dict):
            raise ValueError(
                f"Expected a JSON object for synthesize_process_recipe, got {type(data).__name__}: {data!r}"
            )

        # Extract required fields - raise KeyError if missing (no defaults)
        try:
            evidence = data["evidence"]
            raw_steps = data["process_recipe"]
        except KeyError as e:
            raise KeyError(
                f"Missing required field '{e.args[0]}' in JSON response from synthesize_process_recipe. "
                f"Available keys: {list(data.keys())}"
            ) from e
        
        if not isinstance(evidence, list):
            raise ValueError(
                f"Expected 'evidence' to be a list, got {type(evidence)}: {evidence}"
            )
        
        for i, s in enumerate(raw_steps, 1):
            if isinstance(s, dict):
                # Extract required step fields - raise KeyError if missing
                try:
                    process_recipe.append({
                        "step_index": s["step_index"],
                        "description": s["description"],
                        "conditions": s.get("conditions"),  # Optional field
                        "equipment_class": s.get("equipment_class"),  # Optional field
                        "inputs": s.get("inputs"),  # Optional field
                    })
                except KeyError as e:
                    raise KeyError(
                        f"Missing required field '{e.args[0]}' in process_recipe step {i}. "
                        f"Available keys: {list(s.keys())}"
                    ) from e
            else:
                process_recipe.append({
                    "step_index": i,
                    "description": str(s),
                    "conditions": None,
                    "equipment_class": None,
                    "inputs": None,
                })

        return {
            "process_recipe": process_recipe,
            "evidence": evidence,
        }


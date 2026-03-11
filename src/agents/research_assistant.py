"""ResearchAssistant Agent - Extracts material property keywords"""

import re
from typing import List, Dict, Any
from ..config import load_prompts, load_config
from ..utils.parsing import parse_to_list


class ResearchAssistant:
    """
    An agent that extracts keywords and key phrases for material properties from the original question
    and answered sub-questions.
    """
    
    def __init__(self, name: str = "research_assistant", system_message: str = None, generate_fn=None, chat_logger=None):
        """
        Initialize the ResearchAssistant agent.
        
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
        agent_prompts = prompts.get("agents", {}).get("research_assistant", {})
        agent_config = config.get("agents", {}).get("research_assistant", {})
        
        # Get default system message from YAML - raise error if missing
        default_system_message = agent_prompts.get("default")
        if default_system_message is None:
            raise ValueError(
                "Missing required prompt in config/prompts.yaml: agents.research_assistant.default. "
                "All system prompts must be defined in the config file."
            )
        
        self.system_message = system_message or default_system_message
        
        # Load hyperparameters from config
        self.temperature = agent_config.get("temperature", 0)
        
        if generate_fn is None:
            raise ValueError("generate_fn is required. Pass the LLM generate function.")
        
        # Wrap generate_fn to add logging if chat_logger is provided
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
    
    def _parse_to_list(self, content: str) -> List[str]:
        """Parse LLM response content into a list of keywords/phrases.

        Delegates to the shared ``parse_to_list`` utility so that parsing
        logic is maintained in one place.
        """
        return parse_to_list(content)
    
    def extract_keywords(self, original_question: str, question_answers: List[Dict[str, Any]], temperature: float = None, **kwargs) -> List[str]:
        """
        Extract keywords and key phrases for material properties from the original question and answered sub-questions.
        
        Args:
            original_question: The original research question/sentence
            question_answers: List of question-answer dictionaries. Only pairs with is_answered=True are used.
                            Each dict should have 'question', 'answer', 'is_answered' keys
            temperature: Temperature for LLM generation (default: None, uses config value)
            **kwargs: Additional arguments passed to generate function
            
        Returns:
            List[str]: List of keywords and key phrases for material properties
        """
        # Use config default if not provided
        if temperature is None:
            temperature = self.temperature
        
        if not isinstance(original_question, str):
            raise ValueError(f"original_question must be a string, got {type(original_question)}")
        
        if not original_question.strip():
            raise ValueError("original_question cannot be empty")
        
        if not isinstance(question_answers, list):
            raise ValueError(f"question_answers must be a list, got {type(question_answers)}")
        
        # Filter to only answered questions
        answered_qa_pairs = [qa for qa in question_answers if qa.get('is_answered', False)]
        
        if len(answered_qa_pairs) == 0:
            # If no answered questions, still try to extract from original question
            print("  Warning: No answered questions found. Extracting keywords from original question only.")
            answered_qa_pairs = []
        
        # Load user prompt template from YAML
        prompts = load_prompts()
        extract_keywords_user_prompt_template = prompts.get("agents", {}).get("research_assistant", {}).get("extract_keywords_user_prompt")
        if extract_keywords_user_prompt_template is None:
            raise ValueError(
                "Missing required prompt in config/prompts.yaml: agents.research_assistant.extract_keywords_user_prompt. "
                "All system prompts must be defined in the config file."
            )
        
        # Build QA pairs section
        qa_pairs_section = ""
        if answered_qa_pairs:
            qa_parts = [f"\nBased on the following {len(answered_qa_pairs)} answered research questions and their answers:\n"]
            for i, qa in enumerate(answered_qa_pairs, 1):
                question = qa.get('question', '')
                answer = qa.get('answer', '')
                qa_parts.append(f"\n[Question {i}]")
                qa_parts.append(f"Q: {question}")
                qa_parts.append(f"A: {answer}")
            qa_pairs_section = "\n".join(qa_parts)
        else:
            qa_pairs_section = "\nExtract keywords and key phrases from the original question above."
        
        # Format user prompt with dynamic content
        prompt = extract_keywords_user_prompt_template.format(
            original_question=original_question,
            qa_pairs_section=qa_pairs_section
        )
        
        # Generate keywords using the LLM
        content = self._generate_fn(
            system_prompt=self.system_message,
            prompt=prompt,
            temperature=temperature,
            method_name="extract_keywords",
            **kwargs
        )
        
        # Parse the response into a list of keywords/phrases
        keywords = self._parse_to_list(content)
        
        return keywords

    def extract_constraints(self, original_question: str, question_answers: List[Dict[str, Any]], temperature: float = None, **kwargs) -> List[str]:
        """
        Extract hard constraints for substitute material selection from the original question and answered sub-questions.

        Hard constraints are non-negotiable requirements whose violation immediately disqualifies a material
        (e.g., "Must not contain PFAS", "Continuous service ≥ 250 °C").

        Args:
            original_question: The original research question/sentence
            question_answers: List of question-answer dictionaries. Only pairs with is_answered=True are used.
            temperature: Temperature for LLM generation (default: None, uses config value)
            **kwargs: Additional arguments passed to generate function

        Returns:
            List[str]: List of hard constraint strings
        """
        if temperature is None:
            temperature = self.temperature

        if not isinstance(original_question, str) or not original_question.strip():
            return []
        if not isinstance(question_answers, list):
            return []

        answered_qa_pairs = [qa for qa in question_answers if qa.get("is_answered", False)]

        # Load prompts
        prompts = load_prompts()
        agent_prompts = prompts.get("agents", {}).get("research_assistant", {})

        system_prompt = agent_prompts.get("extract_constraints")
        if system_prompt is None:
            # Gracefully degrade: if the prompt doesn't exist, return empty list
            print("Warning: Missing prompt agents.research_assistant.extract_constraints in prompts.yaml – skipping constraint extraction")
            return []

        user_prompt_template = agent_prompts.get("extract_constraints_user_prompt")
        if user_prompt_template is None:
            print("Warning: Missing prompt agents.research_assistant.extract_constraints_user_prompt in prompts.yaml – skipping constraint extraction")
            return []

        # Build QA section (same structure as extract_keywords)
        if answered_qa_pairs:
            qa_parts = [f"\nBased on the following {len(answered_qa_pairs)} answered research questions and their answers:\n"]
            for i, qa in enumerate(answered_qa_pairs, 1):
                qa_parts.append(f"\n[Question {i}]")
                qa_parts.append(f"Q: {qa.get('question', '')}")
                qa_parts.append(f"A: {qa.get('answer', '')}")
            qa_pairs_section = "\n".join(qa_parts)
        else:
            qa_pairs_section = "\nExtract constraints from the original question above."

        prompt = user_prompt_template.format(
            original_question=original_question,
            qa_pairs_section=qa_pairs_section,
        )

        content = self._generate_fn(
            system_prompt=system_prompt,
            prompt=prompt,
            temperature=temperature,
            method_name="extract_constraints",
            **kwargs,
        )

        constraints = self._parse_to_list(content)
        return constraints

"""Stage 7: Generate - Generate answers from grounded context using LLM.

This module provides functionality to generate answers from grounded context
produced by Stage 6 (Ground) using an LLM. It implements the recommended
approach for SupportDesk RAG: direct prompt with explicit citation instructions.

Key features:
- Grounded answer generation with citation markers
- Configurable model selection (GPT-4o-mini, GPT-4o, etc.)
- Temperature control for hallucination reduction
- Token tracking and cost estimation
- Ollama/OpenAI mode support via model_selector
"""
import os
from typing import List, Dict, Optional
from dataclasses import dataclass

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class GeneratedAnswer:
    """Generated answer from LLM with metadata.

    Args:
        answer: The generated answer text
        citations: List of citation dictionaries referencing source tickets
        prompt_tokens: Number of tokens in the prompt
        completion_tokens: Number of tokens in the completion
        total_tokens: Total tokens used (prompt + completion)
        model: The model name used for generation
    """
    answer: str
    citations: List[Dict]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str


class Config:
    """Configuration manager for Stage 7: Generate."""

    _instance = None
    _config = None

    def __new__(cls, config_path: str = "config/config.yaml"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._config = cls._load_config(config_path)
        return cls._instance

    @staticmethod
    def _load_config(config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            import yaml
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {
                "generate": {
                    "model": "gpt-4o-mini",
                    "temperature": 0.2,
                    "max_tokens": 1000,
                    "enable_citation_check": True
                }
            }

    def get(self, *keys, default=None):
        """Get config value using dot notation."""
        value = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def get_generate_config(self) -> Dict:
        """Get generate-specific configuration."""
        return self._config.get("generate", {})


class AnswerGenerator:
    """Generate answers from grounded context using LLM.

    This class implements Stage 7: Generate for the SupportDesk RAG system.
    It generates grounded answers from the context produced by Stage 6 (Ground)
    using an LLM with citation tracking.

    The generator follows the recommended approach for SupportDesk RAG:
    - Direct prompt with explicit citation instructions
    - Low temperature to reduce hallucination
    - Citation markers in answer format [CITATION:ticket_id]

    Model Selection:
    - Automatically uses Ollama models when USE_OLLAMA_MODELS=true
    - Uses OpenAI models when USE_OLLAMA_MODELS=false or not set
    - Configurable via environment variables:
      - USE_OLLAMA_MODELS (default: true)
      - OLLAMA_CHAT_MODEL (default: llama3)
      - OPENAI_CHAT_MODEL (default: gpt-3.5-turbo)
      - OPENAI_API_KEY (required for OpenAI mode)

    Usage:
        ```python
        from supportdesk_rag import AnswerGenerator, GroundedContext

        generator = AnswerGenerator(model="gpt-4o-mini", temperature=0.2)

        # Generate answer with grounded context
        answer = generator.generate_answer(
            question="How do I reset my password?",
            context=context.context_text,
            citations=context.citations
        )

        print(f"Answer: {answer.answer}")
        print(f"Citations: {answer.citations}")
        print(f"Tokens used: {answer.total_tokens}")
        ```
    """

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        """
        Initialize answer generator.

        Args:
            model: LLM model name (default: from config or gpt-4o-mini)
            temperature: Temperature for generation (default: from config or 0.2)
            max_tokens: Maximum tokens in response (default: from config or 1000)
        """
        from supportdesk_rag.model_selector import get_llm_model_name, get_client_kwargs

        config = Config()

        # Use model_selector for consistent LLM model selection
        self.model_name = model or get_llm_model_name() or "gpt-4o-mini"
        self.temperature = temperature or config.get_generate_config().get("temperature", 0.2)
        self.max_tokens = max_tokens or config.get_generate_config().get("max_tokens", 1000)

        # Initialize OpenAI client with appropriate settings
        try:
            import openai
            client_kwargs = get_client_kwargs()
            self.client = openai.OpenAI(**client_kwargs)
        except ImportError:
            raise ImportError(
                "openai package is required for AnswerGenerator. "
                "Install with: pip install openai"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}")

    def generate_answer(
        self,
        question: str,
        context: str,
        citations: List[Dict],
        system_prompt: Optional[str] = None
    ) -> GeneratedAnswer:
        """
        Generate grounded answer from context.

        Args:
            question: User's incident question
            context: Grounded context from Stage 6 (Ground)
            citations: Citation metadata for tracing
            system_prompt: Optional custom system prompt

        Returns:
            GeneratedAnswer with answer text and metadata
        """
        # Default system prompt enforces grounded responses
        default_system_prompt = """
You are a support desk assistant answering incident tickets.
Your answer MUST be grounded ONLY in the provided context.
Follow these rules STRICTLY:

1. Use ONLY information from the provided context
2. Add citation markers like [CITATION:ticket_id] after each claim
3. If the context doesn't contain enough information, say "I cannot find sufficient information in the ticket data to answer this question"
4. Do NOT invent, infer, or hallucinate information
5. Keep answers concise and actionable
6. If multiple citations support the same point, list them all: [CITATION:ticket1][CITATION:ticket2]

Context format:
[CITATION:ticket_id]
Passage content...

Question format:
User question

Answer format:
Your answer with [CITATION:...] markers embedded.
"""

        effective_system_prompt = system_prompt or default_system_prompt

        # Build user prompt
        user_prompt = f"""Context:
{context}

---

Question:
{question}

Answer:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": effective_system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                n=1,
                stop=None
            )

            answer = response.choices[0].message.content
            usage = response.usage

            return GeneratedAnswer(
                answer=answer,
                citations=citations,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                model=self.model_name
            )

        except Exception as e:
            raise RuntimeError(f"Answer generation failed: {e}")

    def generate_with_relevance_check(
        self,
        question: str,
        context: str,
        citations: List[Dict],
        relevance_threshold: float = -2.0
    ) -> GeneratedAnswer:
        """
        Generate answer only if context has enough relevant passages.

        Args:
            question: User's incident question
            context: Grounded context from Stage 6 (Ground)
            citations: Citation metadata for tracing
            relevance_threshold: Minimum average rerank score required

        Returns:
            GeneratedAnswer with answer text and metadata (or low-relevance response)
        """
        # Check average relevance score
        if citations:
            avg_score = sum(c.get("score", 0) for c in citations) / len(citations)
            if avg_score < relevance_threshold:
                return GeneratedAnswer(
                    answer="I cannot find sufficient relevant information in the ticket data to answer this question.",
                    citations=[],
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    model=self.model_name
                )

        return self.generate_answer(question, context, citations)


# Factory function
def create_answer_generator(**kwargs) -> AnswerGenerator:
    """Create an AnswerGenerator instance with optional parameters.

    Args:
        **kwargs: Arguments for AnswerGenerator initialization

    Returns:
        AnswerGenerator instance
    """
    return AnswerGenerator(**kwargs)


# Backward compatibility aliases
SupportDeskGenerator = AnswerGenerator
SupportDeskGeneratedAnswer = GeneratedAnswer

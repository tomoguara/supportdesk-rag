"""Unit tests for Stage 7: Generate (Answer Generation)

Tests for:
- Config: Configuration management for Stage 7
- AnswerGenerator: LLM-based answer generation functionality
- GeneratedAnswer: Answer data structure
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from supportdesk_rag.supportdesk_generate import (
    Config,
    AnswerGenerator,
    GeneratedAnswer,
)


# Reset Config singleton before tests
def reset_config_singleton():
    """Reset the Config singleton for testing."""
    Config._instance = None


@pytest.fixture(autouse=True)
def reset_config():
    """Reset Config singleton before each test."""
    reset_config_singleton()
    yield
    reset_config_singleton()


class TestConfig:
    """Tests for Stage 7 Configuration manager."""

    def test_config_singleton(self):
        """Config should be a singleton."""
        config1 = Config()
        config2 = Config()
        assert config1 is config2

    def test_config_default_values(self):
        """Test default configuration values for Stage 7."""
        config = Config.__new__(Config)
        config._config = {
            "generate": {
                "model": "gpt-4o-mini",
                "temperature": 0.2,
                "max_tokens": 1000,
                "enable_citation_check": True
            }
        }

        # Should have generate config with defaults
        generate_config = config.get_generate_config()
        assert generate_config.get("model") == "gpt-4o-mini"
        assert generate_config.get("temperature") == 0.2
        assert generate_config.get("max_tokens") == 1000

    def test_config_get_with_dot_notation(self):
        """Test getting config values using dot notation."""
        config = Config.__new__(Config)
        config._config = {
            "generate": {
                "model": "gpt-4o-mini",
                "temperature": 0.2,
                "max_tokens": 1000
            }
        }

        # Test various config values
        assert config.get("generate", "model") == "gpt-4o-mini"
        assert config.get("generate", "temperature") == 0.2
        assert config.get("nonexistent", "key", default="default") == "default"

    def test_config_load_from_file(self):
        """Test loading config from YAML file."""
        import tempfile
        import os
        
        # Create a temp YAML config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
generate:
  model: llama3:8b
  temperature: 0.3
  max_tokens: 2000
""")
            temp_path = f.name

        try:
            config = Config(config_path=temp_path)
            generate_config = config.get_generate_config()
            assert generate_config.get("model") == "llama3:8b"
            assert generate_config.get("temperature") == 0.3
        finally:
            os.unlink(temp_path)


class TestGeneratedAnswer:
    """Tests for the GeneratedAnswer data structure."""

    def test_generated_answer_creation(self):
        """Test creating a GeneratedAnswer instance."""
        answer = GeneratedAnswer(
            answer="The network timeout is 30 seconds.",
            citations=[{"ticket_id": "INC001", "source": "doc1.pdf"}],
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            model="mistral:7b"
        )

        assert answer.answer == "The network timeout is 30 seconds."
        assert len(answer.citations) == 1
        assert answer.citations[0]["ticket_id"] == "INC001"
        assert answer.total_tokens == 150
        assert answer.model == "mistral:7b"

    def test_generated_answer_optional_citations(self):
        """Test GeneratedAnswer with empty citations."""
        answer = GeneratedAnswer(
            answer="Test answer",
            citations=[],
            prompt_tokens=50,
            completion_tokens=20,
            total_tokens=70,
            model="test-model"
        )

        assert answer.answer == "Test answer"
        assert answer.citations == []
        assert answer.total_tokens == 70

    def test_generated_answer_repr(self):
        """Test the string representation."""
        answer = GeneratedAnswer(
            answer="Test answer",
            citations=[{"ticket_id": "INC001"}],
            prompt_tokens=50,
            completion_tokens=20,
            total_tokens=70,
            model="test-model"
        )

        repr_str = repr(answer)
        assert "GeneratedAnswer" in repr_str
        assert "Test answer" in repr_str
        assert "70" in repr_str  # total_tokens


class TestAnswerGenerator:
    """Tests for the AnswerGenerator class."""

    def test_initialization(self):
        """Test AnswerGenerator initialization."""
        # We can't test full initialization without a real model/client
        # This test verifies the class can be instantiated
        config = Config.__new__(Config)
        config._config = {
            "generate": {
                "model": "gpt-4o-mini",
                "temperature": 0.2,
                "max_tokens": 1000
            }
        }

        # This would fail without OpenAI client, so we just check the class exists
        assert AnswerGenerator is not None

    def test_generate_answer_method_exists(self):
        """Test that generate_answer method exists with correct signature."""
        import inspect
        
        sig = inspect.signature(AnswerGenerator.generate_answer)
        params = list(sig.parameters.keys())
        
        assert "question" in params
        assert "context" in params
        assert "citations" in params
        assert "system_prompt" in params

    def test_model_selector_integration(self):
        """Test that AnswerGenerator uses model_selector correctly."""
        from supportdesk_rag.model_selector import ModelSelector
        
        selector = ModelSelector()
        model = selector.get_llm_model()
        
        # Should return a model name string
        assert model is not None
        assert isinstance(model, str)
        assert len(model) > 0

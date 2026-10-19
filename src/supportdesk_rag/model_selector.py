"""
Model Selection Module

This module provides automatic model selection based on the USE_OLLAMA_MODELS
environment variable. When Ollama mode is enabled, it uses the OpenAI-compatible
endpoint at OLLAMA_BASE_URL (default: http://localhost:11434/v1).

Configuration is read from .env file with the following variables:

USE_OLLAMA_MODELS (bool, default: true)
  - true: Use Ollama models via OpenAI-compatible endpoint
  - false: Use OpenAI cloud models

OLLAMA_BASE_URL (str, default: http://localhost:11434/v1)
  - Base URL for Ollama's OpenAI-compatible API

OLLAMA_EMBEDDING_MODEL (str, default: nomic-embed-text)
  - Model name for embeddings when USE_OLLAMA_MODELS=true

OLLAMA_CHAT_MODEL (str, default: llama3)
  - Model name for LLM when USE_OLLAMA_MODELS=true

OPENAI_API_KEY (required for OpenAI mode)
  - Your OpenAI API key

OPENAI_EMBEDDING_MODEL (str, default: text-embedding-3-small)
  - Model name for embeddings when USE_OLLAMA_MODELS=false

OPENAI_CHAT_MODEL (str, default: gpt-3.5-turbo)
  - Model name for LLM when USE_OLLAMA_MODELS=false
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load .env file
load_dotenv()


class ModelSelector:
    """Centralized model selection based on USE_OLLAMA_MODELS configuration."""
    
    def __init__(self):
        self._use_ollama = self._parse_bool(
            os.getenv("USE_OLLAMA_MODELS", "true")
        )
    
    @staticmethod
    def _parse_bool(value: str) -> bool:
        """Parse string to boolean, case-insensitive."""
        if isinstance(value, bool):
            return value
        return value.lower() in ("true", "1", "yes")
    
    @property
    def use_ollama(self) -> bool:
        """Return whether Ollama mode is enabled."""
        return self._use_ollama
    
    def get_embedding_model(self, default: Optional[str] = None) -> str:
        """
        Get the appropriate embedding model based on mode.
        
        Args:
            default: Fallback model name if none specified
            
        Returns:
            Model name string
        """
        if self._use_ollama:
            return os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        return os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    
    def get_llm_model(self, default: Optional[str] = None) -> str:
        """
        Get the appropriate LLM model based on mode.
        
        Args:
            default: Fallback model name if none specified
            
        Returns:
            Model name string
        """
        if self._use_ollama:
            return os.getenv("OLLAMA_CHAT_MODEL", "llama3")
        return os.getenv("OPENAI_CHAT_MODEL", "gpt-3.5-turbo")
    
    def get_base_url(self) -> Optional[str]:
        """
        Get the base URL for API calls.
        
        Returns:
            Base URL string for Ollama-compatible API, or None for OpenAI
        """
        if self._use_ollama:
            return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        return None
    
    def get_openai_api_key(self) -> str:
        """
        Get the API key (required for OpenAI mode, optional for Ollama).
        
        Returns:
            API key string
            
        Raises:
            ValueError: If in OpenAI mode and no API key is set
        """
        if not self._use_ollama:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY environment variable is required when "
                    "USE_OLLAMA_MODELS=false"
                )
            return api_key
        # Ollama API key is optional
        return os.getenv("OPENAI_API_KEY", "dummy-key-for-ollama")
    
    def get_client_kwargs(self) -> dict:
        """Get kwargs for initializing OpenAI client."""
        base_url = self.get_base_url()
        api_key = self.get_openai_api_key()
        
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
            
        return kwargs


def create_model_selector() -> ModelSelector:
    """Factory function to create a ModelSelector instance."""
    return ModelSelector()


# Global singleton for easy access
_model_selector: Optional[ModelSelector] = None


def get_model_selector() -> ModelSelector:
    """Get the global ModelSelector instance."""
    global _model_selector
    if _model_selector is None:
        _model_selector = ModelSelector()
    return _model_selector


def use_ollama() -> bool:
    """Convenience function to check if Ollama mode is enabled."""
    return get_model_selector().use_ollama


def get_embedding_model_name() -> str:
    """Get the current embedding model name."""
    return get_model_selector().get_embedding_model()


def get_llm_model_name() -> str:
    """Get the current LLM model name."""
    return get_model_selector().get_llm_model()


def get_base_url() -> Optional[str]:
    """Get the current base URL for API calls."""
    return get_model_selector().get_base_url()


def get_client_kwargs() -> dict:
    """Get kwargs for initializing OpenAI client.
    
    Returns:
        Dictionary with api_key and optionally base_url for OpenAI client
    """
    selector = get_model_selector()
    base_url = selector.get_base_url()
    api_key = selector.get_openai_api_key()
    
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    
    return kwargs


def print_model_config() -> None:
    """Print the current model configuration to stdout.
    
    Displays the active mode, embedding model, LLM model, and API endpoint
    being used by the RAG system.
    """
    selector = get_model_selector()
    
    mode = "OLLAMA (local)" if selector.use_ollama else "OPENAI (cloud)"
    base_url = selector.get_base_url()
    embedding_model = selector.get_embedding_model()
    llm_model = selector.get_llm_model()
    
    print("=" * 60)
    print("SupportDesk RAG - Model Configuration")
    print("=" * 60)
    print(f"Mode:              {mode}")
    print(f"Embedding Model:   {embedding_model}")
    print(f"LLM Model:         {llm_model}")
    if base_url:
        print(f"API Base URL:      {base_url}")
    else:
        print(f"API Base URL:      https://api.openai.com/v1 (default)")
    print("=" * 60)

"""Unit tests for Stage 3: Storage (VectorStore)"""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from supportdesk_rag.supportdesk_store import (
    Config,
    VectorStore,
    StorePipeline,
    create_vector_store,
    create_store_pipeline,
    get_embedding_model_dimensions
)


class TestConfig:
    """Tests for Configuration manager"""

    def test_config_singleton(self):
        """Config should be a singleton"""
        config1 = Config()
        config2 = Config()
        assert config1 is config2

    def test_config_default_values(self):
        """Test default configuration values"""
        config = Config()

        # Should have qdrant config with defaults
        qdrant_config = config.get_qdrant_config()
        assert qdrant_config.get("host") == "localhost"
        assert qdrant_config.get("port") == 6333
        assert qdrant_config.get("collection_name") == "tickets"

    def test_config_get_with_dot_notation(self):
        """Test getting config values using dot notation"""
        config = Config()
        assert config.get("openai", "embedding_model") == "text-embedding-3-small"
        assert config.get("ingest", "chunk_size") == 1000


class TestGetEmbeddingModelDimensions:
    """Tests for get_embedding_model_dimensions function"""

    def test_text_embedding_3_small(self):
        """text-embedding-3-small should return 1536 dimensions"""
        with patch('supportdesk_rag.supportdesk_store.Config') as MockConfig:
            # Simulate get() method that handles section, key, default
            def mock_get(section=None, key=None, default=None):
                if section == "openai" and key == "embedding_model":
                    return "text-embedding-3-small"
                elif section == "openai" and key == "embedding_dimensions":
                    return None
                return default
            MockConfig.return_value.get.side_effect = mock_get
            dims = get_embedding_model_dimensions()
            assert dims == 1536

    def test_text_embedding_3_large(self):
        """text-embedding-3-large should return 3072 dimensions"""
        with patch('supportdesk_rag.supportdesk_store.Config') as MockConfig:
            def mock_get(section=None, key=None, default=None):
                if section == "openai" and key == "embedding_model":
                    return "text-embedding-3-large"
                elif section == "openai" and key == "embedding_dimensions":
                    return None
                return default
            MockConfig.return_value.get.side_effect = mock_get
            dims = get_embedding_model_dimensions()
            assert dims == 3072

    def test_text_embedding_ada_002(self):
        """text-embedding-ada-002 should return 1536 dimensions"""
        with patch('supportdesk_rag.supportdesk_store.Config') as MockConfig:
            def mock_get(section=None, key=None, default=None):
                if section == "openai" and key == "embedding_model":
                    return "text-embedding-ada-002"
                elif section == "openai" and key == "embedding_dimensions":
                    return None
                return default
            MockConfig.return_value.get.side_effect = mock_get
            dims = get_embedding_model_dimensions()
            assert dims == 1536

    def test_unknown_model_returns_default(self):
        """Unknown model should return explicit embedding_dimensions config value"""
        with patch('supportdesk_rag.supportdesk_store.Config') as MockConfig:
            def mock_get(section=None, key=None, default=None):
                if section == "openai" and key == "embedding_model":
                    return "unknown-model"
                elif section == "openai" and key == "embedding_dimensions":
                    # In real config, this is 4096
                    return None
                return default
            MockConfig.return_value.get.side_effect = mock_get
            dims = get_embedding_model_dimensions()
            # With no explicit dimensions and unknown model, returns default 4096 (Ollama model dim)
            assert dims == 4096


class TestStorePipeline:
    """Tests for StorePipeline class - the main integration point"""

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock VectorStore"""
        mock_store = Mock(spec=VectorStore)
        mock_store.dimensions = 1536
        mock_store.upsert_points = Mock()
        mock_store.count_points = Mock(return_value=100)
        mock_store.collection_name = "test_collection"
        return mock_store

    def test_store_chunks_with_valid_data(self, mock_vector_store):
        """Test storing chunks with valid embeddings"""
        chunks = [
            {
                "ticket_id": "TICK-001",
                "chunk_index": 0,
                "chunk_text": "Test chunk 1",
                "embedding": [0.1] * 1536,
                "metadata": {
                    "service": "network",
                    "severity": "high",
                    "ticket_type": "Incident",
                    "created_at": "2026-01-15",
                    "user_name": "John",
                    "user_email": "john@example.com"
                }
            },
            {
                "ticket_id": "TICK-002",
                "chunk_index": 0,
                "chunk_text": "Test chunk 2",
                "embedding": [0.2] * 1536,
                "metadata": {
                    "service": "database",
                    "severity": "medium",
                    "ticket_type": "Service Request",
                    "created_at": "2026-01-16",
                    "user_name": "Jane",
                    "user_email": "jane@example.com"
                }
            }
        ]

        pipeline = StorePipeline(vector_store=mock_vector_store)
        result = pipeline.store_chunks(chunks, batch_size=10)

        assert result["stored"] == 2
        assert result["collection"] == "test_collection"
        assert result["total_points"] == 100

    def test_store_chunks_empty_list(self, mock_vector_store):
        """Test storing empty chunk list"""
        pipeline = StorePipeline(vector_store=mock_vector_store)
        result = pipeline.store_chunks([])

        assert result["stored"] == 0

    def test_store_chunks_dimension_mismatch_warning(self, mock_vector_store, capsys):
        """Test dimension mismatch warning"""
        chunks = [
            {
                "embedding": [0.1] * 100,  # Wrong dimension
                "ticket_id": "TICK-001",
                "chunk_index": 0,
                "chunk_text": "Test chunk",
                "metadata": {}
            }
        ]

        mock_vector_store.dimensions = 1536
        mock_vector_store.upsert_points = Mock()
        mock_vector_store.count_points = Mock(return_value=1)

        pipeline = StorePipeline(vector_store=mock_vector_store)
        pipeline.store_chunks(chunks)

        # Should print warning about dimension mismatch
        captured = capsys.readouterr()
        assert "Warning" in captured.out or "mismatch" in captured.out.lower()


class TestFactoryFunctions:
    """Tests for factory functions"""

    def test_create_vector_store(self):
        """Test create_vector_store factory function"""
        with patch('supportdesk_rag.supportdesk_store.VectorStore') as MockStore:
            MockStore.return_value = Mock(spec=VectorStore)
            store = create_vector_store(host="localhost", port=6333)
            assert store is not None

    def test_create_store_pipeline(self):
        """Test create_store_pipeline factory function"""
        with patch('supportdesk_rag.supportdesk_store.VectorStore') as MockStore:
            MockStore.return_value = Mock(spec=VectorStore)
            pipeline = create_store_pipeline()
            assert pipeline is not None
            assert pipeline.vector_store is not None

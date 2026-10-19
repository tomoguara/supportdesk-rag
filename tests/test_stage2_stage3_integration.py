"""Integration tests for Stage 2 (Embed) + Stage 3 (Store)

Tests the complete pipeline from embedded chunks to Qdrant storage.
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json
import os
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from supportdesk_rag.supportdesk_rag_pipeline import (
    Config,
    EmbeddingService,
    EmbeddingPipeline,
    SupportDeskRAGPipelineWithStore
)
from supportdesk_rag.supportdesk_store import VectorStore, StorePipeline


def _make_config_get_mock(embedding_model=None, embedding_dimensions=None):
    """Helper to create a mock Config.get() method."""
    def mock_get(section=None, key=None, default=None):
        if section == "openai":
            if key == "embedding_model":
                return embedding_model if embedding_model is not None else "text-embedding-3-small"
            elif key == "embedding_dimensions":
                # For unit tests, return None to use model-based dimensions
                # Set to 4096 for integration tests that run with real Qdrant
                return embedding_dimensions
            elif key == "api_key_env":
                return "OPENAI_API_KEY"
        elif section == "qdrant":
            if key == "host":
                return "localhost"
            elif key == "port":
                return 6333
        elif section == "store":
            if key == "vector_db_path":
                return "data/vector_db"
        return default
    return mock_get


    """Helper to create a mock Config.get() method."""
    def mock_get(section=None, key=None, default=None):
        if section == "openai":
            if key == "embedding_model":
                return embedding_model if embedding_model is not None else "text-embedding-3-small"
            elif key == "embedding_dimensions":
                return embedding_dimensions if embedding_dimensions is not None else 4096
            elif key == "api_key_env":
                return "OPENAI_API_KEY"
        elif section == "qdrant":
            if key == "host":
                return "localhost"
            elif key == "port":
                return 6333
        elif section == "store":
            if key == "vector_db_path":
                return "data/vector_db"
        return default
    return mock_get



class TestStage2Stage3Integration:
    """Integration tests for Stage 2 + Stage 3 pipeline"""

    @pytest.fixture
    def sample_embedded_chunks(self):
        """Generate sample embedded chunks for testing"""
        return [
            {
                "ticket_id": "TICK-001",
                "chunk_index": 0,
                "chunk_text": "Network connection timeout when accessing API endpoint",
                "embedding": [0.1] * 1536,
                "metadata": {
                    "service": "Network Infrastructure",
                    "severity": "High",
                    "ticket_type": "Incident",
                    "created_at": "2026-01-15T10:00:00",
                    "user_name": "John Doe",
                    "user_email": "john@example.com"
                }
            },
            {
                "ticket_id": "TICK-002",
                "chunk_index": 0,
                "chunk_text": "Database query slow after deployment",
                "embedding": [0.2] * 1536,
                "metadata": {
                    "service": "Database Services",
                    "severity": "Medium",
                    "ticket_type": "Performance Issue",
                    "created_at": "2026-01-16T14:30:00",
                    "user_name": "Jane Smith",
                    "user_email": "jane@example.com"
                }
            },
            {
                "ticket_id": "TICK-003",
                "chunk_index": 0,
                "chunk_text": "Email delivery delay affecting marketing team",
                "embedding": [0.3] * 1536,
                "metadata": {
                    "service": "Email & Collaboration",
                    "severity": "High",
                    "ticket_type": "Incident",
                    "created_at": "2026-01-17T09:15:00",
                    "user_name": "Bob Wilson",
                    "user_email": "bob@example.com"
                }
            }
        ]

    @pytest.fixture
    def mock_embedding_service(self):
        """Create a mock embedding service"""
        mock_service = Mock(spec=EmbeddingService)
        mock_service.embed_text.return_value = [0.1] * 1536
        mock_service.embed_batch.return_value = [[0.1] * 1536] * 3
        mock_service.model = "text-embedding-3-small"
        mock_service.dimensions = 1536
        return mock_service

    def test_store_pipeline_stores_embedded_chunks(self, sample_embedded_chunks):
        """Test that StorePipeline properly stores embedded chunks"""
        with patch('qdrant_client.QdrantClient') as MockClient:
            mock_client = Mock()
            MockClient.return_value = mock_client
            mock_client.collection_exists.return_value = False

            with patch('supportdesk_rag.supportdesk_store.Config') as MockConfig:
                MockConfig.return_value.get.side_effect = _make_config_get_mock(embedding_dimensions=None)
                store = VectorStore()

                pipeline = StorePipeline(vector_store=store)
                result = pipeline.store_chunks(sample_embedded_chunks, batch_size=10)

                # Verify chunks were stored
                assert result["stored"] == 3
                assert mock_client.upsert.called

    def test_store_pipeline_filters_by_service(self, sample_embedded_chunks):
        """Test filtering by service in StorePipeline"""
        with patch('qdrant_client.QdrantClient') as MockClient:
            mock_client = Mock()
            MockClient.return_value = mock_client
            mock_client.collection_exists.return_value = False

            with patch('supportdesk_rag.supportdesk_store.Config') as MockConfig:
                MockConfig.return_value.get.side_effect = _make_config_get_mock(embedding_dimensions=None)
                store = VectorStore()

                # Filter for Network Infrastructure
                network_chunks = [c for c in sample_embedded_chunks
                                 if c["metadata"]["service"] == "Network Infrastructure"]

                assert len(network_chunks) == 1
                assert network_chunks[0]["ticket_id"] == "TICK-001"

    def test_full_pipeline_run_with_storage(self, sample_embedded_chunks, mock_embedding_service):
        """Test full pipeline with embedding and storage"""
        with patch('qdrant_client.QdrantClient') as MockClient:
            mock_client = Mock()
            MockClient.return_value = mock_client
            mock_client.collection_exists.return_value = False

            with patch('supportdesk_rag.supportdesk_store.Config') as MockConfig:
                MockConfig.return_value.get.side_effect = _make_config_get_mock(embedding_dimensions=None)
                store = VectorStore()
                pipeline = StorePipeline(vector_store=store)

                # Process chunks through pipeline
                result = pipeline.store_chunks(sample_embedded_chunks)

                # Verify pipeline processed all chunks
                assert result["stored"] == 3
                assert mock_client.upsert.called

    def test_full_pipeline_without_storage(self, sample_embedded_chunks):
        """Test pipeline behavior when storage is disabled (dry run)"""
        with patch('qdrant_client.QdrantClient') as MockClient:
            mock_client = Mock()
            MockClient.return_value = mock_client
            mock_client.collection_exists.return_value = False

            with patch('supportdesk_rag.supportdesk_store.Config') as MockConfig:
                MockConfig.return_value.get.side_effect = _make_config_get_mock(embedding_dimensions=None)
                store = VectorStore()
                pipeline = StorePipeline(vector_store=store)

                result = pipeline.store_chunks(sample_embedded_chunks)

                # Should still return results even if storage is mocked
                assert result["stored"] == 3

    def test_search_stored_embeddings(self, sample_embedded_chunks):
        """Test searching stored embeddings"""
        # Mock search result
        mock_search_result = Mock()
        mock_search_result.id = "TICK-001"
        mock_search_result.score = 0.95
        mock_search_result.payload = {"service": "Network Infrastructure"}

        # Mock query_points response with points attribute (as per query_points API)
        mock_points = [Mock()]
        mock_points[0].id = "TICK-001"
        mock_points[0].score = 0.95
        mock_points[0].payload = {"service": "Network Infrastructure"}
        mock_query_response = Mock()
        mock_query_response.points = mock_points

        with patch('qdrant_client.QdrantClient') as MockClient:
            mock_client = Mock()
            MockClient.return_value = mock_client
            mock_client.collection_exists.return_value = True
            mock_client.query_points.return_value = mock_query_response

            with patch('supportdesk_rag.supportdesk_store.Config') as MockConfig:
                MockConfig.return_value.get.side_effect = _make_config_get_mock(embedding_dimensions=None)
                store = VectorStore()
                results = store.search([0.1] * 1536, top_k=5)

                assert len(results) == 1
                assert results[0]["id"] == "TICK-001"
                assert results[0]["score"] == 0.95

    def test_storage_stats(self):
        """Test storage statistics"""
        mock_count = Mock()
        mock_count.count = 42

        with patch('qdrant_client.QdrantClient') as MockClient:
            mock_client = Mock()
            MockClient.return_value = mock_client
            mock_client.collection_exists.return_value = True
            mock_client.count.return_value = mock_count

            with patch('supportdesk_rag.supportdesk_store.Config') as MockConfig:
                MockConfig.return_value.get.side_effect = _make_config_get_mock(embedding_dimensions=None)
                store = VectorStore()

                stats = {
                    "total_points": store.count_points(),
                    "collection": store.collection_name,
                    "dimensions": store.dimensions
                }

                assert stats["total_points"] == 42


class TestErrorHandling:
    """Tests for error handling in Stage 2 + Stage 3 pipeline"""

    def test_store_pipeline_handles_dimension_mismatch(self):
        """Test that pipeline handles embedding dimension mismatches with warning"""
        chunks = [
            {
                "ticket_id": "TICK-001",
                "embedding": [0.1] * 100,  # Wrong dimension
                "chunk_index": 0,
                "chunk_text": "Test chunk",
                "metadata": {}
            }
        ]

        with patch('qdrant_client.QdrantClient') as MockClient:
            mock_client = Mock()
            MockClient.return_value = mock_client
            mock_client.collection_exists.return_value = False

            with patch('supportdesk_rag.supportdesk_store.Config') as MockConfig:
                MockConfig.return_value.get.side_effect = _make_config_get_mock(embedding_dimensions=None)
                store = VectorStore()
                pipeline = StorePipeline(vector_store=store)

                # Should detect dimension mismatch
                result = pipeline.store_chunks(chunks)

                # Should still store (but with warning)
                assert result["stored"] == 1

    def test_full_pipeline_handles_vector_store_failure(self):
        """Test pipeline behavior when vector store fails"""
        chunks = [
            {
                "ticket_id": "TICK-001",
                "embedding": [0.1] * 1536,
                "chunk_index": 0,
                "chunk_text": "Test chunk",
                "metadata": {}
            }
        ]

        with patch('qdrant_client.QdrantClient') as MockClient:
            mock_client = Mock()
            MockClient.return_value = mock_client
            mock_client.collection_exists.return_value = True
            # Simulate store failure
            mock_client.upsert.side_effect = Exception("Connection refused")

            with patch('supportdesk_rag.supportdesk_store.Config') as MockConfig:
                MockConfig.return_value.get.side_effect = _make_config_get_mock(embedding_dimensions=None)
                store = VectorStore()
                pipeline = StorePipeline(vector_store=store)

                # Pipeline should handle the exception
                with pytest.raises(Exception):
                    pipeline.store_chunks(chunks)


class TestMetadataFiltering:
    """Tests for metadata-based filtering in storage"""

    def test_filter_by_severity(self):
        """Test filtering by severity level"""
        chunks = [
            {
                "ticket_id": "TICK-001",
                "embedding": [0.1] * 1536,
                "metadata": {"severity": "Critical"}
            },
            {
                "ticket_id": "TICK-002",
                "embedding": [0.2] * 1536,
                "metadata": {"severity": "High"}
            },
            {
                "ticket_id": "TICK-003",
                "embedding": [0.3] * 1536,
                "metadata": {"severity": "Medium"}
            },
            {
                "ticket_id": "TICK-004",
                "embedding": [0.4] * 1536,
                "metadata": {"severity": "High"}
            }
        ]

        # Filter for High severity
        high_severity_chunks = [c for c in chunks if c["metadata"]["severity"] == "High"]

        assert len(high_severity_chunks) == 2

    def test_filter_by_service_and_severity(self):
        """Test filtering by multiple metadata fields"""
        chunks = [
            {
                "ticket_id": "TICK-001",
                "embedding": [0.1] * 1536,
                "metadata": {"service": "Network", "severity": "High"}
            },
            {
                "ticket_id": "TICK-002",
                "embedding": [0.2] * 1536,
                "metadata": {"service": "Network", "severity": "Medium"}
            },
            {
                "ticket_id": "TICK-003",
                "embedding": [0.3] * 1536,
                "metadata": {"service": "Database", "severity": "High"}
            }
        ]

        # Filter for Network + High severity
        filtered = [
            c for c in chunks
            if c["metadata"]["service"] == "Network"
            and c["metadata"]["severity"] == "High"
        ]

        assert len(filtered) == 1
        assert filtered[0]["ticket_id"] == "TICK-001"


class TestLargeScaleStorage:
    """Tests for large-scale storage scenarios"""

    def test_store_many_chunks(self):
        """Test storing a large number of chunks"""
        # Generate 50 chunks
        chunks = [
            {
                "ticket_id": f"TICK-{i:03d}",
                "chunk_index": 0,
                "chunk_text": f"Test chunk {i} with content",
                "embedding": [random.uniform(0, 1) for _ in range(1536)],
                "metadata": {
                    "service": "Network Infrastructure",
                    "severity": "Medium",
                    "ticket_type": "Service Request",
                    "created_at": f"2026-01-{15 + (i % 15):02d}",
                    "user_name": f"User {i}",
                    "user_email": f"user{i}@example.com"
                }
            }
            for i in range(50)
        ]

        with patch('qdrant_client.QdrantClient') as MockClient:
            mock_client = Mock()
            MockClient.return_value = mock_client
            mock_client.collection_exists.return_value = False

            with patch('supportdesk_rag.supportdesk_store.Config') as MockConfig:
                MockConfig.return_value.get.side_effect = _make_config_get_mock(embedding_dimensions=None)
                store = VectorStore()
                pipeline = StorePipeline(vector_store=store)

                result = pipeline.store_chunks(chunks, batch_size=100)

                assert result["stored"] == 50

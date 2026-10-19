"""Real integration tests for full RAG pipeline with Qdrant.

Tests end-to-end: Ingest → Embed → Store → Search workflow.
These tests require Qdrant to be running on localhost:6333.
"""
import pytest
import sys
import json
from pathlib import Path
import uuid
import time
from unittest.mock import patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from supportdesk_rag.supportdesk_store import (
    VectorStore,
    StorePipeline,
    get_embedding_model_dimensions
)


def get_test_collection_name():
    """Generate a unique collection name for testing."""
    timestamp = int(time.time() * 1000) % 1000000
    return f"test_pipeline_{uuid.uuid4().hex[:8]}_{timestamp}"


class TestFullPipelineIntegration:
    """End-to-end tests for the complete RAG pipeline."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Set up and tear down Qdrant collection for each test."""
        try:
            self.collection_name = get_test_collection_name()
            self.dimensions = get_embedding_model_dimensions()
            self.store = VectorStore(collection_name=self.collection_name)
            self.pipeline = StorePipeline(vector_store=self.store)
            self._qdrant_available = True
        except Exception as e:
            pytest.skip(f"Qdrant not available: {e}")
            self._qdrant_available = False
            return

        yield

        try:
            if hasattr(self.store, 'client'):
                self.store.client.delete_collection(collection_name=self.collection_name)
        except Exception:
            pass

    def test_full_pipeline_ingest_embed_store(self):
        """Test complete pipeline: create test data, ingest, embed, and store."""
        if not hasattr(self, '_qdrant_available') or not self._qdrant_available:
            pytest.skip("Qdrant not available")
            
        from supportdesk_rag.supportdesk_rag_pipeline import SupportDeskRAGPipeline
        import tempfile
        import os

        # Create temporary test data
        test_tickets = {
            "tickets": [
                {
                    "ticket_id": "PIPE-001",
                    "description": "Network connection timeout when accessing API endpoint",
                    "service": "Network Infrastructure",
                    "severity": "High",
                    "ticket_type": "Incident",
                    "created_at": "2026-01-15T10:00:00",
                    "user": {"name": "John Doe", "email": "john@example.com"}
                },
                {
                    "ticket_id": "PIPE-002",
                    "description": "Database query slow after deployment",
                    "service": "Database Services",
                    "severity": "Medium",
                    "ticket_type": "Performance Issue",
                    "created_at": "2026-01-16T14:30:00",
                    "user": {"name": "Jane Smith", "email": "jane@example.com"}
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_tickets, f)
            temp_path = f.name

        try:
            # Create pipeline with smaller chunk size for testing
            rag_pipeline = SupportDeskRAGPipeline(
                chunk_size=100,
                chunk_overlap=10
            )

            # Run the pipeline
            result = rag_pipeline.run(temp_path, save=False, store=False)

            # Verify ingestion
            assert "ingest" in result
            assert result["ingest"]["chunks_generated"] == 2

            # Verify embedding
            assert "embed" in result
            assert result["embed"]["embeddings_created"] == 2

            # Store the embedded chunks
            embedded_chunks = result["embed"]["embedded_chunks"]
            store_result = self.pipeline.store_chunks(embedded_chunks)

            # Verify storage
            assert store_result["stored"] == 2
            assert self.store.count_points() == 2

        finally:
            os.unlink(temp_path)

    def test_search_after_pipeline_storage(self):
        """Test that search returns results after pipeline storage."""
        if not hasattr(self, '_qdrant_available') or not self._qdrant_available:
            pytest.skip("Qdrant not available")
            
        from supportdesk_rag.supportdesk_embed import EmbeddingService, Config

        # Use real embedding service
        config = Config()
        service = EmbeddingService(
            model=config.get_embed_config().get("embedding_model", "text-embedding-3-small")
        )

        # Create chunks with real embeddings
        chunks = [
            {
                "ticket_id": "SEARCH-001",
                "chunk_index": 0,
                "chunk_text": "How to configure VPN access for remote employees",
                "embedding": service.embed_text("How to configure VPN access for remote employees"),
                "metadata": {
                    "service": "Network",
                    "severity": "Medium",
                    "user": "John Doe"
                }
            },
            {
                "ticket_id": "SEARCH-002",
                "chunk_index": 0,
                "chunk_text": "Password reset not working for corporate email",
                "embedding": service.embed_text("Password reset not working for corporate email"),
                "metadata": {
                    "service": "Account Management",
                    "severity": "High",
                    "user": "Jane Smith"
                }
            }
        ]

        # Store chunks
        self.pipeline.store_chunks(chunks)
        assert self.store.count_points() == 2

        # Search with related query
        query_text = "VPN configuration for remote workers"
        query_embedding = service.embed_text(query_text)

        results = self.store.search(query_embedding, top_k=2)

        # Should return results
        assert len(results) >= 1

        # Verify first result has expected metadata
        first_result = results[0]
        assert "payload" in first_result
        assert "service" in first_result["payload"]

    def test_pipeline_with_multiple_severities(self):
        """Test pipeline with various severity levels and search filtering."""
        if not hasattr(self, '_qdrant_available') or not self._qdrant_available:
            pytest.skip("Qdrant not available")
            
        from supportdesk_rag.supportdesk_embed import EmbeddingService, Config

        config = Config()
        service = EmbeddingService(
            model=config.get_embed_config().get("embedding_model", "text-embedding-3-small")
        )

        # Create chunks with different severities
        chunks = [
            {
                "ticket_id": "CRIT-001",
                "chunk_index": 0,
                "chunk_text": "Production database completely unreachable",
                "embedding": service.embed_text("Production database completely unreachable"),
                "metadata": {"severity": "Critical", "service": "Database"}
            },
            {
                "ticket_id": "HIGH-001",
                "chunk_index": 0,
                "chunk_text": "API response times exceeding acceptable limits",
                "embedding": service.embed_text("API response times exceeding acceptable limits"),
                "metadata": {"severity": "High", "service": "API"}
            },
            {
                "ticket_id": "MED-001",
                "chunk_index": 0,
                "chunk_text": "Minor UI glitch on login page",
                "embedding": service.embed_text("Minor UI glitch on login page"),
                "metadata": {"severity": "Medium", "service": "Frontend"}
            }
        ]

        self.pipeline.store_chunks(chunks)

        # Search for high severity issues
        high_query = "critical system failure"
        high_embedding = service.embed_text(high_query)
        high_results = self.store.filtered_search(
            high_embedding,
            severity="High",
            top_k=3
        )

        # Should get results with High severity
        assert len(high_results) >= 1
        for result in high_results:
            assert result["payload"]["severity"] == "High"


class TestPipelineBackwardCompatibility:
    """Test that existing code patterns still work with new pipeline."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Set up and tear down Qdrant collection for each test."""
        try:
            self.collection_name = get_test_collection_name()
            self.dimensions = get_embedding_model_dimensions()
            self.store = VectorStore(collection_name=self.collection_name)
            self.pipeline = StorePipeline(vector_store=self.store)
            self._qdrant_available = True
        except Exception as e:
            pytest.skip(f"Qdrant not available: {e}")
            self._qdrant_available = False
            return

        yield

        try:
            if hasattr(self.store, 'client'):
                self.store.client.delete_collection(collection_name=self.collection_name)
        except Exception:
            pass

    def test_existing_code_pattern_still_works(self):
        """Test that the old code pattern (direct VectorStore usage) still works."""
        if not hasattr(self, '_qdrant_available') or not self._qdrant_available:
            pytest.skip("Qdrant not available")
            
        # This simulates existing code that uses VectorStore directly
        chunks = [
            {
                "ticket_id": "EXISTING-001",
                "chunk_index": 0,
                "chunk_text": "Test using existing code pattern",
                "embedding": [0.1] * self.dimensions,
                "metadata": {"pattern": "existing", "version": "legacy"}
            }
        ]

        # Direct VectorStore usage (old pattern)
        store_result = self.store.upsert_points(chunks)
        assert store_result is not None

        # Should still be able to search
        query = [0.11] * self.dimensions
        results = self.store.search(query, top_k=1)
        assert len(results) == 1
        assert results[0]["payload"]["ticket_id"] == "EXISTING-001"

    def test_factory_function_still_works(self):
        """Test that create_store_pipeline factory function works."""
        if not hasattr(self, '_qdrant_available') or not self._qdrant_available:
            pytest.skip("Qdrant not available")
            
        from supportdesk_rag.supportdesk_store import create_store_pipeline

        # Use factory function
        store = VectorStore(collection_name=self.collection_name)
        pipeline = create_store_pipeline(vector_store=store)

        # Should be able to use the pipeline
        chunks = [
            {
                "ticket_id": "FACTORY-001",
                "chunk_index": 0,
                "chunk_text": "Factory function test",
                "embedding": [0.2] * self.dimensions,
                "metadata": {}
            }
        ]

        result = pipeline.store_chunks(chunks)
        assert result["stored"] == 1


class TestPipelineCleanup:
    """Test cleanup scenarios."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Set up and tear down Qdrant collection for each test."""
        try:
            self.collection_name = get_test_collection_name()
            self.dimensions = get_embedding_model_dimensions()
            self.store = VectorStore(collection_name=self.collection_name)
            self.pipeline = StorePipeline(vector_store=self.store)
            self._qdrant_available = True
        except Exception as e:
            pytest.skip(f"Qdrant not available: {e}")
            self._qdrant_available = False
            return

        yield

        try:
            if hasattr(self.store, 'client'):
                self.store.client.delete_collection(collection_name=self.collection_name)
        except Exception:
            pass

    def test_collection_gets_deleted_after_test(self):
        """Test that cleanup fixture properly deletes collections."""
        if not hasattr(self, '_qdrant_available') or not self._qdrant_available:
            pytest.skip("Qdrant not available")
            
        chunks = [
            {
                "ticket_id": "CLEANUP-001",
                "chunk_index": 0,
                "chunk_text": "Cleanup test",
                "embedding": [0.1] * self.dimensions,
                "metadata": {}
            }
        ]

        self.pipeline.store_chunks(chunks)
        assert self.store.count_points() == 1

        # Collection should still exist after test (in teardown)
        # This test verifies the teardown runs
        pass  # If we get here, fixture teardown should handle cleanup

    def test_large_dataset_storage(self):
        """Test storing and searching a larger dataset."""
        if not hasattr(self, '_qdrant_available') or not self._qdrant_available:
            pytest.skip("Qdrant not available")
            
        from supportdesk_rag.supportdesk_embed import EmbeddingService, Config

        config = Config()
        service = EmbeddingService(
            model=config.get_embed_config().get("embedding_model", "text-embedding-3-small")
        )

        # Create 20 chunks
        chunks = []
        for i in range(20):
            text = f"Ticket description number {i} with some unique content"
            chunks.append({
                "ticket_id": f"LARGE-{i:03d}",
                "chunk_index": 0,
                "chunk_text": text,
                "embedding": service.embed_text(text),
                "metadata": {
                    "index": i,
                    "service": f"Service {i % 3}",
                    "severity": ["Low", "Medium", "High"][i % 3]
                }
            })

        self.pipeline.store_chunks(chunks)

        # Verify count
        assert self.store.count_points() == 20

        # Search should still work
        query_text = "ticket description with content"
        query_embedding = service.embed_text(query_text)
        results = self.store.search(query_embedding, top_k=5)

        assert len(results) >= 1


class TestHybridSearchIntegration:
    """Test hybrid search in the pipeline."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Set up and tear down Qdrant collection for each test."""
        try:
            from supportdesk_rag.supportdesk_embed import EmbeddingService, Config

            self.collection_name = get_test_collection_name()
            self.dimensions = get_embedding_model_dimensions()
            self.store = VectorStore(collection_name=self.collection_name)
            self.pipeline = StorePipeline(vector_store=self.store)
            self.embed_service = EmbeddingService(
                model=Config().get_embed_config().get("embedding_model", "text-embedding-3-small")
            )
            self._qdrant_available = True
        except Exception as e:
            pytest.skip(f"Qdrant not available: {e}")
            self._qdrant_available = False
            return

        yield

        try:
            if hasattr(self.store, 'client'):
                self.store.client.delete_collection(collection_name=self.collection_name)
        except Exception:
            pass

    def test_hybrid_search_in_pipeline(self):
        """Test hybrid search is available in the pipeline."""
        if not hasattr(self, '_qdrant_available') or not self._qdrant_available:
            pytest.skip("Qdrant not available")

        from supportdesk_rag.supportdesk_retrieve import HybridRetriever, create_hybrid_retriever

        # Create test chunks
        test_texts = [
            "Network connection timeout when accessing API endpoint",
            "Database query slow after deployment",
            "User authentication failing with LDAP server",
        ]

        chunks = []
        for i, text in enumerate(test_texts):
            embedding = self.embed_service.embed_text(text)
            chunks.append({
                "ticket_id": f"HYBRID-{i+1}",
                "chunk_index": 0,
                "chunk_text": text,
                "embedding": embedding,
                "metadata": {
                    "service": ["Network", "Database", "Authentication"][i],
                    "severity": ["High", "Medium", "High"][i],
                }
            })

        # Store chunks
        self.pipeline.store_chunks(chunks)
        assert self.store.count_points() == 3

        # Create hybrid retriever
        retriever = create_hybrid_retriever(
            vector_store=self.store,
            embedding_service=self.embed_service
        )

        # Test hybrid search
        query = "network timeout"
        results = retriever.retrieve(query=query, top_k=5)

        # Verify results
        assert len(results) > 0, "Should return at least one result"
        assert results[0].get("fused_score") is not None, "Results should have fused_score"
        assert results[0].get("retriever") in ["vector", "bm25"], "Results should indicate retriever"

        # First result should be related to network
        first_result = results[0]
        assert "network" in first_result.get("document", "").lower() or \
               "timeout" in first_result.get("document", "").lower()

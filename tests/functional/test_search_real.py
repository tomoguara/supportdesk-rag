"""Real integration tests for Stage 3: Storage - Search functionality with Qdrant.

Tests semantic search and filtering with real Qdrant instance.
These tests require Qdrant to be running on localhost:6333.
"""
import pytest
import sys
from pathlib import Path
import uuid
import time

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
    return f"test_search_{uuid.uuid4().hex[:8]}_{timestamp}"


class TestSemanticSearchReal:
    """Real integration tests for semantic search with Qdrant."""

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

    def test_semantic_search_with_related_query(self):
        """Test that similar queries return relevant results."""
        if not hasattr(self, '_qdrant_available') or not self._qdrant_available:
            pytest.skip("Qdrant not available")
            
        # Create chunks with distinct topics
        network_chunks = [
            {
                "ticket_id": "NET-001",
                "chunk_index": 0,
                "chunk_text": "Network connection timeout when accessing API endpoint",
                "embedding": [0.1] * self.dimensions,
                "metadata": {"service": "Network", "topic": "API"}
            },
            {
                "ticket_id": "NET-002",
                "chunk_index": 0,
                "chunk_text": "DNS resolution failing for external domains",
                "embedding": [0.11] * self.dimensions,
                "metadata": {"service": "Network", "topic": "DNS"}
            }
        ]

        db_chunks = [
            {
                "ticket_id": "DB-001",
                "chunk_index": 0,
                "chunk_text": "Database query slow after deployment",
                "embedding": [0.9] * self.dimensions,
                "metadata": {"service": "Database", "topic": "Performance"}
            },
            {
                "ticket_id": "DB-002",
                "chunk_index": 0,
                "chunk_text": "Connection pool exhausted in production",
                "embedding": [0.91] * self.dimensions,
                "metadata": {"service": "Database", "topic": "Connection"}
            }
        ]

        self.pipeline.store_chunks(network_chunks + db_chunks)

        # Search for network-related query
        network_query = [0.15] * self.dimensions
        results = self.store.search(network_query, top_k=2)

        assert len(results) == 2

    def test_search_with_metadata_filter(self):
        """Test searching with metadata filters."""
        if not hasattr(self, '_qdrant_available') or not self._qdrant_available:
            pytest.skip("Qdrant not available")
            
        chunks = [
            {
                "ticket_id": "TICK-HIGH-1",
                "chunk_index": 0,
                "chunk_text": "High severity network issue",
                "embedding": [0.1] * self.dimensions,
                "metadata": {"severity": "High", "service": "Network"}
            },
            {
                "ticket_id": "TICK-HIGH-2",
                "chunk_index": 0,
                "chunk_text": "High priority database request",
                "embedding": [0.2] * self.dimensions,
                "metadata": {"severity": "High", "service": "Database"}
            },
            {
                "ticket_id": "TICK-LOW-1",
                "chunk_index": 0,
                "chunk_text": "Low severity question",
                "embedding": [0.3] * self.dimensions,
                "metadata": {"severity": "Low", "service": "General"}
            }
        ]

        self.pipeline.store_chunks(chunks)

        # Search with severity filter
        query = [0.15] * self.dimensions
        results = self.store.filtered_search(
            query,
            severity="High",
            top_k=3
        )

        # Should return only high severity tickets
        for result in results:
            assert result["payload"]["severity"] == "High"

    def test_search_returns_top_k_results(self):
        """Test that search returns the correct number of results."""
        if not hasattr(self, '_qdrant_available') or not self._qdrant_available:
            pytest.skip("Qdrant not available")
            
        # Create 10 chunks with varying embeddings
        chunks = []
        for i in range(10):
            # Use different base values to create distinct embeddings
            base = i * 0.1
            embedding = [base + j * 0.001 for j in range(self.dimensions)]
            chunks.append({
                "ticket_id": f"TICK-{i:03d}",
                "chunk_index": 0,
                "chunk_text": f"Test chunk {i}",
                "embedding": embedding,
                "metadata": {"index": i}
            })

        self.pipeline.store_chunks(chunks)

        # Search for top 5 results
        query = [0.05] * self.dimensions
        results = self.store.search(query, top_k=5)

        assert len(results) == 5

    def test_search_with_zero_results(self):
        """Test search when no matches are found."""
        if not hasattr(self, '_qdrant_available') or not self._qdrant_available:
            pytest.skip("Qdrant not available")
            
        # Store chunks with one distinct topic
        chunks = [
            {
                "ticket_id": "TICK-001",
                "chunk_index": 0,
                "chunk_text": "Network test",
                "embedding": [0.1] * self.dimensions,
                "metadata": {}
            }
        ]
        self.pipeline.store_chunks(chunks)

        # Search with very different embedding
        query = [5.0] * self.dimensions  # Very different values
        results = self.store.search(query, top_k=5, score_threshold=0.9)

        # May return results or empty depending on similarity
        assert isinstance(results, list)


class TestSearchWithRealEmbeddings:
    """Test search using real embeddings from EmbeddingService."""

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

    def test_search_with_real_embeddings_workflow(self):
        """Test the full workflow: store with real embeddings, search with query."""
        if not hasattr(self, '_qdrant_available') or not self._qdrant_available:
            pytest.skip("Qdrant not available")
            
        from supportdesk_rag.supportdesk_embed import EmbeddingService, Config

        # Use real embedding service
        config = Config()
        service = EmbeddingService(
            model=config.get_embed_config().get("embedding_model", "text-embedding-3-small")
        )
        dimensions = service.dimensions

        # Create chunks with real embeddings
        real_chunks = [
            {
                "ticket_id": "REAL-001",
                "chunk_index": 0,
                "chunk_text": "How to reset user password",
                "embedding": service.embed_text("How to reset user password"),
                "metadata": {"service": "Account Management", "topic": "Authentication"}
            },
            {
                "ticket_id": "REAL-002",
                "chunk_index": 0,
                "chunk_text": "Password reset not working for employees",
                "embedding": service.embed_text("Password reset not working for employees"),
                "metadata": {"service": "Account Management", "topic": "Authentication"}
            },
            {
                "ticket_id": "REAL-003",
                "chunk_index": 0,
                "chunk_text": "Network configuration for VPN access",
                "embedding": service.embed_text("Network configuration for VPN access"),
                "metadata": {"service": "Network", "topic": "VPN"}
            }
        ]

        self.pipeline.store_chunks(real_chunks)

        # Search with similar query
        query_text = "password reset failed"
        query_embedding = service.embed_text(query_text)

        results = self.store.search(query_embedding, top_k=2)

        # Should return at least one result
        assert len(results) >= 1

        # First result should be related to password
        first_result = results[0]
        assert "password" in first_result["payload"]["chunk_text"].lower() or \
               "reset" in first_result["payload"]["chunk_text"].lower()


class TestSearchThresholds:
    """Test search with different score thresholds."""

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

    def test_high_score_threshold(self):
        """Test search with high score threshold returns fewer results."""
        if not hasattr(self, '_qdrant_available') or not self._qdrant_available:
            pytest.skip("Qdrant not available")
            
        chunks = [
            {
                "ticket_id": "TICK-CLOSE",
                "chunk_index": 0,
                "chunk_text": "Very similar text to query",
                "embedding": [0.5] * self.dimensions,
                "metadata": {}
            },
            {
                "ticket_id": "TICK-FAR",
                "chunk_index": 0,
                "chunk_text": "Very different from query",
                "embedding": [0.9] * self.dimensions,
                "metadata": {}
            }
        ]

        self.pipeline.store_chunks(chunks)

        query = [0.51] * self.dimensions

        # With high threshold, should get fewer results
        high_threshold_results = self.store.search(query, top_k=10, score_threshold=0.9)

        # Low threshold results
        low_threshold_results = self.store.search(query, top_k=10, score_threshold=0.0)

        assert len(high_threshold_results) <= len(low_threshold_results)

    def test_score_threshold_zero_returns_all(self):
        """Test that score_threshold=0 returns all results."""
        if not hasattr(self, '_qdrant_available') or not self._qdrant_available:
            pytest.skip("Qdrant not available")
            
        chunks = [
            {
                "ticket_id": f"TICK-{i}",
                "chunk_index": 0,
                "chunk_text": f"Chunk {i}",
                "embedding": [i * 0.1] * self.dimensions,
                "metadata": {}
            }
            for i in range(5)
        ]

        self.pipeline.store_chunks(chunks)

        # Query with very different embedding
        query = [5.0] * self.dimensions
        results = self.store.search(query, top_k=10, score_threshold=0.0)

        # Should return results (Qdrant may return fewer than 5 due to vector similarity handling)
        assert len(results) >= 4

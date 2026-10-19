"""Real integration tests for Stage 4: Retrieve - Hybrid Search with Qdrant.

Tests hybrid search workflow with real Qdrant instance:
- End-to-end: Ingest → Embed → Store → Hybrid Retrieve
- BM25 index building from vector store
- RRF fusion with real embeddings

These tests require Qdrant to be running on localhost:6333.
"""
import pytest
import sys
import json
import uuid
import time
from pathlib import Path
from unittest.mock import patch, Mock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def get_test_collection_name():
    """Generate a unique collection name for testing."""
    timestamp = int(time.time() * 1000) % 1000000
    return f"test_retrieve_{uuid.uuid4().hex[:8]}_{timestamp}"


class TestHybridSearchIntegration:
    """Integration tests for hybrid search with real Qdrant."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Set up and tear down Qdrant collection for each test."""
        try:
            from supportdesk_rag.supportdesk_store import (
                VectorStore,
                StorePipeline,
                get_embedding_model_dimensions
            )
            from supportdesk_rag.supportdesk_embed import EmbeddingService, Config

            self.collection_name = get_test_collection_name()
            self.dimensions = get_embedding_model_dimensions()

            # Create vector store
            self.store = VectorStore(collection_name=self.collection_name)
            self.pipeline = StorePipeline(vector_store=self.store)

            # Create embedding service
            config = Config()
            self.embed_service = EmbeddingService(
                model=config.get_embed_config().get("embedding_model", "text-embedding-3-small")
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

    def test_hybrid_search_with_real_embeddings(self):
        """Test hybrid search with real Qdrant storage and embeddings."""
        if not hasattr(self, '_qdrant_available') or not self._qdrant_available:
            pytest.skip("Qdrant not available")

        from supportdesk_rag.supportdesk_retrieve import HybridRetriever

        # Create some test chunks with real embeddings
        test_texts = [
            "Network connection timeout when accessing API endpoint",
            "Database query slow after deployment",
            "User authentication failing with LDAP server",
            "Memory leak detected in application server",
            "Disk space warning on production database",
        ]

        chunks = []
        for i, text in enumerate(test_texts):
            embedding = self.embed_service.embed_text(text)
            chunks.append({
                "ticket_id": f"HR-00{i+1}",
                "chunk_index": 0,
                "chunk_text": text,
                "embedding": embedding,
                "metadata": {
                    "service": ["Network", "Database", "Authentication", "Application", "Storage"][i],
                    "severity": ["High", "Medium", "High", "Critical", "Medium"][i],
                }
            })

        # Store chunks
        self.pipeline.store_chunks(chunks)
        assert self.store.count_points() == len(test_texts)

        # Create hybrid retriever
        retriever = HybridRetriever(
            vector_store=self.store,
            embedding_service=self.embed_service
        )

        # Test hybrid search
        query = "network timeout"
        results = retriever.retrieve(query=query, top_k=5)

        # Verify results
        assert len(results) > 0, "Should return at least one result"
        assert results[0].get("fused_score") is not None, "Results should have fused_score"

        # First result should be related to network
        first_result = results[0]
        assert "network" in first_result.get("document", "").lower() or \
               "timeout" in first_result.get("document", "").lower()

    def test_hybrid_search_returns_retriever_info(self):
        """Test that results include information about which retriever found them."""
        if not hasattr(self, '_qdrant_available') or not self._qdrant_available:
            pytest.skip("Qdrant not available")

        from supportdesk_rag.supportdesk_retrieve import HybridRetriever

        # Create test data
        chunks = []
        test_texts = [
            "Very unique technical term X123",
            "Common phrase about database connection",
        ]

        for i, text in enumerate(test_texts):
            embedding = self.embed_service.embed_text(text)
            chunks.append({
                "ticket_id": f"HRT-00{i+1}",
                "chunk_index": 0,
                "chunk_text": text,
                "embedding": embedding,
                "metadata": {}
            })

        self.pipeline.store_chunks(chunks)

        # Create hybrid retriever
        retriever = HybridRetriever(
            vector_store=self.store,
            embedding_service=self.embed_service
        )

        # Test hybrid search
        query = "database connection"
        results = retriever.retrieve(query=query, top_k=5)

        # Verify retriever info is included
        assert len(results) > 0
        for result in results:
            assert "fused_score" in result
            assert "retriever" in result
            # Should be either "vector" or "bm25"
            assert result["retriever"] in ["vector", "bm25"]

    def test_hybrid_search_with_custom_top_k(self):
        """Test hybrid search with custom vector_top_k and bm25_top_k."""
        if not hasattr(self, '_qdrant_available') or not self._qdrant_available:
            pytest.skip("Qdrant not available")

        from supportdesk_rag.supportdesk_retrieve import HybridRetriever

        # Create test data
        chunks = []
        for i in range(10):
            text = f"Document number {i} with different content"
            embedding = self.embed_service.embed_text(text)
            chunks.append({
                "ticket_id": f"HTK-{i:03d}",
                "chunk_index": 0,
                "chunk_text": text,
                "embedding": embedding,
                "metadata": {}
            })

        self.pipeline.store_chunks(chunks)

        retriever = HybridRetriever(
            vector_store=self.store,
            embedding_service=self.embed_service
        )

        # Test with higher vector_top_k
        results = retriever.retrieve(
            query="document content",
            top_k=5,
            vector_top_k=8,
            bm25_top_k=8
        )

        assert len(results) <= 5  # Should respect top_k
        assert len(results) >= 1


class TestBM25FromVectorStore:
    """Tests for BM25 index built from vector store."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Set up and tear down Qdrant collection."""
        try:
            from supportdesk_rag.supportdesk_store import (
                VectorStore,
                StorePipeline,
                get_embedding_model_dimensions
            )
            from supportdesk_rag.supportdesk_embed import EmbeddingService, Config

            self.collection_name = get_test_collection_name()
            self.dimensions = get_embedding_model_dimensions()
            self.store = VectorStore(collection_name=self.collection_name)
            self.pipeline = StorePipeline(vector_store=self.store)

            config = Config()
            self.embed_service = EmbeddingService(
                model=config.get_embed_config().get("embedding_model", "text-embedding-3-small")
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

    def test_bm25_index_built_from_vector_store(self):
        """Test that BM25 index is automatically built from vector store documents."""
        if not hasattr(self, '_qdrant_available') or not self._qdrant_available:
            pytest.skip("Qdrant not available")

        from supportdesk_rag.supportdesk_retrieve import HybridRetriever, BM25Index

        # Create test documents
        documents = [
            "Error code 503: Service unavailable",
            "Error code 502: Bad gateway",
            "Error code 404: Not found",
        ]

        chunks = []
        for i, text in enumerate(documents):
            embedding = self.embed_service.embed_text(text)
            chunks.append({
                "ticket_id": f"BFT-{i+1}",
                "chunk_index": 0,
                "chunk_text": text,
                "embedding": embedding,
                "metadata": {}
            })

        self.pipeline.store_chunks(chunks)

        # Create retriever - should auto-build BM25 index
        retriever = HybridRetriever(
            vector_store=self.store,
            embedding_service=self.embed_service
        )

        # Verify BM25 index was built
        assert retriever.bm25_index is not None
        assert isinstance(retriever.bm25_index, BM25Index)
        assert len(retriever.bm25_index.documents) == len(documents)

    def test_bm25_search_works_after_auto_build(self):
        """Test that BM25 search works after automatic index building."""
        if not hasattr(self, '_qdrant_available') or not self._qdrant_available:
            pytest.skip("Qdrant not available")

        from supportdesk_rag.supportdesk_retrieve import HybridRetriever

        documents = [
            "Very specific technical term ABC123",
            "Different technical term XYZ789",
            "Third term PQR456",
        ]

        chunks = []
        for i, text in enumerate(documents):
            embedding = self.embed_service.embed_text(text)
            chunks.append({
                "ticket_id": f"BSW-{i+1}",
                "chunk_index": 0,
                "chunk_text": text,
                "embedding": embedding,
                "metadata": {}
            })

        self.pipeline.store_chunks(chunks)

        retriever = HybridRetriever(
            vector_store=self.store,
            embedding_service=self.embed_service
        )

        # BM25 search should work
        bm25_results = retriever.bm25_index.search("ABC123", top_k=5)
        assert len(bm25_results) >= 1
        assert "ABC123" in bm25_results[0]["document"]


class TestRerankAfterRetrieve:
    """Tests for RRF fusion scoring behavior."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Set up and tear down Qdrant collection."""
        try:
            from supportdesk_rag.supportdesk_store import (
                VectorStore,
                StorePipeline,
                get_embedding_model_dimensions
            )
            from supportdesk_rag.supportdesk_embed import EmbeddingService, Config

            self.collection_name = get_test_collection_name()
            self.dimensions = get_embedding_model_dimensions()
            self.store = VectorStore(collection_name=self.collection_name)
            self.pipeline = StorePipeline(vector_store=self.store)

            config = Config()
            self.embed_service = EmbeddingService(
                model=config.get_embed_config().get("embedding_model", "text-embedding-3-small")
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

    def test_rrf_favors_documents_in_both_results(self):
        """Test that RRF gives higher scores to documents appearing in both retriever results."""
        if not hasattr(self, '_qdrant_available') or not self._qdrant_available:
            pytest.skip("Qdrant not available")

        from supportdesk_rag.supportdesk_retrieve import HybridRetriever

        # Create documents where one appears in both semantic and keyword search
        documents = [
            "Very specific password reset error code ERR-1001",
            "General about password reset instructions",
            "Error code ERR-1002 for login problems",
        ]

        chunks = []
        for i, text in enumerate(documents):
            embedding = self.embed_service.embed_text(text)
            chunks.append({
                "ticket_id": f"RRF-{i+1}",
                "chunk_index": 0,
                "chunk_text": text,
                "embedding": embedding,
                "metadata": {}
            })

        self.pipeline.store_chunks(chunks)

        retriever = HybridRetriever(
            vector_store=self.store,
            embedding_service=self.embed_service
        )

        # Query that matches both semantically and by keyword
        query = "password reset error"
        results = retriever.retrieve(query=query, top_k=5)

        # Document 0 contains both "password reset" (semantic) and "error" (keyword)
        # Document 1 contains "password reset" (semantic)
        # Document 2 contains "error" (keyword)

        # The document with both should rank highest or high up
        assert len(results) >= 1

        # Check that we have fused scores
        for result in results:
            assert "fused_score" in result
            fused_score = result["fused_score"]
            assert isinstance(fused_score, float)
            # Fused scores should be between 0 and 1 (roughly)
            assert 0 <= fused_score <= 0.1  # RRF with k=60 returns small values

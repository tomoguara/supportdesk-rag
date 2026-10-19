"""Real integration tests for Stage 5: Re-rank - Cross-Encoder Re-ranking with Qdrant.

Tests re-ranking workflow with real Qdrant instance:
- End-to-end: Ingest → Embed → Store → Hybrid Retrieve → Re-rank
- Cross-encoder re-ranking with real embeddings

These tests require Qdrant to be running on localhost:6333.
"""

import pytest
import sys
import json
import uuid
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def get_test_collection_name():
    """Generate a unique collection name for testing."""
    timestamp = int(time.time() * 1000) % 1000000
    return f"test_rerank_{uuid.uuid4().hex[:8]}_{timestamp}"


class TestReRankIntegration:
    """Integration tests for re-ranking with real Qdrant."""

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

    def test_rerank_with_real_embeddings(self):
        """Test re-ranking with real Qdrant storage and embeddings."""
        if not hasattr(self, '_qdrant_available') or not self._qdrant_available:
            pytest.skip("Qdrant not available")

        from supportdesk_rag.supportdesk_rerank import ReRanker
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
                "ticket_id": f"RR-{i+1}",
                "chunk_index": 0,
                "chunk_text": text,
                "embedding": embedding,
                "metadata": {
                    "service": ["Network", "Database", "Authentication", "Application", "Storage"][i],
                    "severity": ["High", "Medium", "High", "Critical", "Low"][i],
                }
            })

        # Store chunks
        self.pipeline.store_chunks(chunks)
        assert self.store.count_points() == 5

        # Create hybrid retriever
        retriever = HybridRetriever(
            vector_store=self.store,
            embedding_service=self.embed_service
        )

        # Get hybrid search results
        query = "network timeout"
        hybrid_results = retriever.retrieve(query=query, top_k=5)

        assert len(hybrid_results) >= 1
        assert "fused_score" in hybrid_results[0]

        # Now test re-ranking
        reranker = ReRanker()

        reranked_results = reranker.rerank(
            query=query,
            documents=hybrid_results,
            top_k=3
        )

        # Verify results are re-ranked
        assert len(reranked_results) <= 3
        assert all("rerank_score" in r for r in reranked_results)

        # Results should be sorted by rerank_score (descending)
        scores = [r["rerank_score"] for r in reranked_results]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_improves_ranking_for_relevant_documents(self):
        """Test that re-ranking puts more relevant documents at top."""
        if not hasattr(self, '_qdrant_available') or not self._qdrant_available:
            pytest.skip("Qdrant not available")

        from supportdesk_rag.supportdesk_rerank import ReRanker
        from supportdesk_rag.supportdesk_retrieve import HybridRetriever

        # Create documents with varying relevance to query
        documents = [
            "How to reset admin password for the system",
            "Generic network troubleshooting guide",
            "Password reset error code ERR-1001",
        ]

        chunks = []
        for i, text in enumerate(documents):
            embedding = self.embed_service.embed_text(text)
            chunks.append({
                "ticket_id": f"PWRD-{i+1}",
                "chunk_index": 0,
                "chunk_text": text,
                "embedding": embedding,
                "metadata": {}
            })

        self.pipeline.store_chunks(chunks)
        assert self.store.count_points() == 3

        # Get hybrid results for password reset query
        query = "password reset"
        retriever = HybridRetriever(
            vector_store=self.store,
            embedding_service=self.embed_service
        )
        hybrid_results = retriever.retrieve(query=query, top_k=3)

        # Initial ranking (before re-rank)
        initial_scores = {r["id"]: r.get("fused_score", 0) for r in hybrid_results}

        # Re-rank
        reranker = ReRanker()
        reranked_results = reranker.rerank(query=query, documents=hybrid_results, top_k=3)

        # Check that rerank_score was added
        assert all("rerank_score" in r for r in reranked_results)

        # The document containing both "password" and "reset" should rank higher
        # After re-ranking, the most relevant document should be at top
        first_result = reranked_results[0]
        assert "password" in first_result.get("document", "").lower() or \
               "password" in first_result.get("payload", {}).get("chunk_text", "").lower()

    def test_rerank_with_top_k_parameter(self):
        """Test that re-ranking respects top_k parameter."""
        if not hasattr(self, '_qdrant_available') or not self._qdrant_available:
            pytest.skip("Qdrant not available")

        from supportdesk_rag.supportdesk_rerank import ReRanker
        from supportdesk_rag.supportdesk_retrieve import HybridRetriever

        # Create documents
        documents = [
            "Document one about networking",
            "Document two about databases",
            "Document three about security",
            "Document four about performance",
            "Document five about storage",
        ]

        chunks = []
        for i, text in enumerate(documents):
            embedding = self.embed_service.embed_text(text)
            chunks.append({
                "ticket_id": f"TOPK-{i+1}",
                "chunk_index": 0,
                "chunk_text": text,
                "embedding": embedding,
                "metadata": {}
            })

        self.pipeline.store_chunks(chunks)

        # Get hybrid results
        query = "networking"
        retriever = HybridRetriever(
            vector_store=self.store,
            embedding_service=self.embed_service
        )
        hybrid_results = retriever.retrieve(query=query, top_k=5)

        # Re-rank with different top_k values
        reranker = ReRanker()

        # top_k=2
        results2 = reranker.rerank(query=query, documents=hybrid_results, top_k=2)
        assert len(results2) == 2

        # top_k=3
        results3 = reranker.rerank(query=query, documents=hybrid_results, top_k=3)
        assert len(results3) == 3

        # top_k=10 (more than available)
        results10 = reranker.rerank(query=query, documents=hybrid_results, top_k=10)
        assert len(results10) == 5

    def test_rerank_with_min_score_filter(self):
        """Test rerank_with_metadata filters by minimum score."""
        if not hasattr(self, '_qdrant_available') or not self._qdrant_available:
            pytest.skip("Qdrant not available")

        from supportdesk_rag.supportdesk_rerank import ReRanker
        from supportdesk_rag.supportdesk_retrieve import HybridRetriever

        # Create test documents
        documents = [
            "Network connection timeout when accessing API endpoint",
            "Database query slow after deployment",
            "User authentication failing with LDAP server",
        ]

        chunks = []
        for i, text in enumerate(documents):
            embedding = self.embed_service.embed_text(text)
            chunks.append({
                "ticket_id": f"FILTER-{i+1}",
                "chunk_index": 0,
                "chunk_text": text,
                "embedding": embedding,
                "metadata": {}
            })

        self.pipeline.store_chunks(chunks)

        # Get hybrid results
        query = "network timeout"
        retriever = HybridRetriever(
            vector_store=self.store,
            embedding_service=self.embed_service
        )
        hybrid_results = retriever.retrieve(query=query, top_k=3)

        # Re-rank with strict min_score
        reranker = ReRanker(min_score=-3.0)
        results_strict = reranker.rerank_with_metadata(
            query=query,
            documents=hybrid_results,
            min_score=-3.0,
            top_k=10
        )

        # All results should have score >= -3.0
        assert all(r.get("rerank_score", float('-inf')) >= -3.0 for r in results_strict)

        # More lenient filter should include more or equal results
        reranker_lenient = ReRanker(min_score=-10.0)
        results_lenient = reranker_lenient.rerank_with_metadata(
            query=query,
            documents=hybrid_results,
            min_score=-10.0,
            top_k=10
        )

        assert len(results_lenient) >= len(results_strict)

    def test_rerank_pipeline_integration(self):
        """Test ReRankerPipeline integration with hybrid search."""
        if not hasattr(self, '_qdrant_available') or not self._qdrant_available:
            pytest.skip("Qdrant not available")

        from supportdesk_rag.supportdesk_rerank import ReRanker, ReRankerPipeline
        from supportdesk_rag.supportdesk_retrieve import HybridRetriever

        # Create test documents
        documents = [
            "Network connection timeout when accessing API endpoint",
            "Database query slow after deployment",
            "User authentication failing with LDAP server",
        ]

        chunks = []
        for i, text in enumerate(documents):
            embedding = self.embed_service.embed_text(text)
            chunks.append({
                "ticket_id": f"PIPELINE-{i+1}",
                "chunk_index": 0,
                "chunk_text": text,
                "embedding": embedding,
                "metadata": {}
            })

        self.pipeline.store_chunks(chunks)

        # Setup hybrid search and re-ranking
        query = "network timeout"
        retriever = HybridRetriever(
            vector_store=self.store,
            embedding_service=self.embed_service
        )
        hybrid_results = retriever.retrieve(query=query, top_k=3)

        # Re-rank using pipeline
        reranker = ReRanker()
        pipeline = ReRankerPipeline(reranker=reranker, enabled=True, top_k=2)

        results = pipeline.run(query=query, documents=hybrid_results, top_k=2)

        # Verify results
        assert len(results) <= 2
        assert all("rerank_score" in r for r in results)
        # Verify sorting
        scores = [r["rerank_score"] for r in results]
        assert scores == sorted(scores, reverse=True)


class TestReRankEmptyResults:
    """Tests for edge cases with empty or minimal results."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Set up and tear down Qdrant collection for each test."""
        try:
            from supportdesk_rag.supportdesk_store import VectorStore, get_embedding_model_dimensions
            self.collection_name = get_test_collection_name()
            self.store = VectorStore(collection_name=self.collection_name)
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

    def test_rerank_empty_documents(self):
        """Test re-ranking with no documents returns empty list."""
        from supportdesk_rag.supportdesk_rerank import ReRanker

        reranker = ReRanker()
        results = reranker.rerank(query="test", documents=[])
        assert results == []

    def test_rerank_with_none_documents(self):
        """Test re-ranking with None documents."""
        from supportdesk_rag.supportdesk_rerank import ReRanker

        reranker = ReRanker()
        results = reranker.rerank(query="test", documents=None)
        assert results == []

    def test_rerank_with_single_document(self):
        """Test re-ranking with single document."""
        if not hasattr(self, '_qdrant_available') or not self._qdrant_available:
            pytest.skip("Qdrant not available")

        from supportdesk_rag.supportdesk_rerank import ReRanker

        # Create a single document
        documents = [
            {"payload": {"content": "Single document about networking"}}
        ]

        reranker = ReRanker()
        results = reranker.rerank(query="network", documents=documents, top_k=10)

        assert len(results) == 1
        assert "rerank_score" in results[0]

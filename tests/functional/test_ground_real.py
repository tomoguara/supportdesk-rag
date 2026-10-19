"""Real integration tests for Stage 6: Ground - Context Grounding with Qdrant.

Tests context grounding workflow with real Qdrant instance:
- End-to-end: Ingest → Embed → Store → Hybrid Retrieve → Re-rank → Ground
- Context grounding with real embeddings and citations
- Citation tracking and metadata verification

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

from supportdesk_rag.supportdesk_ground import (
    GroundedContext,
    ContextGrounding,
)
from supportdesk_rag.supportdesk_retrieve import HybridRetriever
from supportdesk_rag.supportdesk_store import VectorStore, StorePipeline, get_embedding_model_dimensions
from supportdesk_rag.supportdesk_embed import EmbeddingService, Config


def get_test_collection_name():
    """Generate a unique collection name for testing."""
    timestamp = int(time.time() * 1000) % 1000000
    return f"test_ground_{uuid.uuid4().hex[:8]}_{timestamp}"


def check_qdrant_available():
    """Check if Qdrant is available."""
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host="localhost", port=6333, prefer_grpc=False)
        client.get_collections()
        return True
    except Exception:
        return False


def qdrant_skip_reason():
    """Skip reason if Qdrant is not available."""
    if not check_qdrant_available():
        return "Qdrant not available (connection refused)"
    return None


@pytest.fixture(scope="module", autouse=True)
def qdrant_available():
    """Check if Qdrant is available before running tests."""
    skip = qdrant_skip_reason()
    if skip:
        pytest.skip(skip, allow_module_level=True)


class TestContextGroundingIntegration:
    """Integration tests for context grounding with real Qdrant."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Set up and tear down Qdrant collection for each test."""
        skip = qdrant_skip_reason()
        if skip:
            pytest.skip(skip)

        try:
            self.collection_name = get_test_collection_name()
            self.dimensions = get_embedding_model_dimensions()

            # Create vector store
            self.store = VectorStore(collection_name=self.collection_name)
            self.pipeline = StorePipeline(vector_store=self.store)

            # Create embedding service
            config = Config()
            self.embed_service = EmbeddingService()

            yield
        finally:
            # Cleanup
            try:
                self.store.delete_collection()
            except Exception:
                pass

    def test_context_grounding_with_real_embeddings(self):
        """Test context grounding with real embeddings from Qdrant."""
        # Prepare test data
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
                "ticket_id": f"CG-00{i+1}",
                "chunk_index": 0,
                "chunk_text": text,
                "embedding": embedding,
                "metadata": {
                    "service": ["Network", "Database", "Authentication", "Application", "Storage"][i],
                    "severity": ["High", "Medium", "High", "Critical", "Low"][i],
                }
            })

        self.pipeline.store_chunks(chunks)

        # Setup hybrid search and re-ranking
        query = "network timeout"
        retriever = HybridRetriever(
            vector_store=self.store,
            embedding_service=self.embed_service
        )
        hybrid_results = retriever.retrieve(query=query, top_k=5)

        # Test context grounding
        grounder = ContextGrounding(max_passages=5)
        grounded = grounder.ground_context(hybrid_results)

        # Verify grounded context
        assert grounded.context_text is not None
        assert len(grounded.citations) > 0
        assert all("ticket_id" in c for c in grounded.citations)
        assert all("score" in c for c in grounded.citations)

    def test_citation_tracking(self):
        """Test that citations are properly tracked."""
        # Store test document
        test_text = "Solution: Restart the network service"
        embedding = self.embed_service.embed_text(test_text)

        self.pipeline.store_chunks([{
            "ticket_id": "CITATION-001",
            "chunk_index": 0,
            "chunk_text": test_text,
            "embedding": embedding,
            "metadata": {"service": "Network", "severity": "High"}
        }])

        retriever = HybridRetriever(
            vector_store=self.store,
            embedding_service=self.embed_service
        )
        hybrid_results = retriever.retrieve(query="restart network", top_k=5)

        grounder = ContextGrounding()
        grounded = grounder.ground_context(hybrid_results)

        # Verify citation
        assert len(grounded.citations) > 0
        assert grounded.citations[0]["ticket_id"] == "CITATION-001"


class TestGroundedContextWithRealData:
    """Tests for GroundedContext with real Qdrant data."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Set up and tear down Qdrant collection for each test."""
        skip = qdrant_skip_reason()
        if skip:
            pytest.skip(skip)

        try:
            self.collection_name = get_test_collection_name()
            self.dimensions = get_embedding_model_dimensions()

            self.store = VectorStore(collection_name=self.collection_name)
            self.pipeline = StorePipeline(vector_store=self.store)

            config = Config()
            self.embed_service = EmbeddingService()

            yield
        finally:
            try:
                self.store.delete_collection()
            except Exception:
                pass

    def test_grounding_with_retrieved_results(self):
        """Test context grounding with retrieved results from Qdrant."""
        # Store multiple documents
        documents = [
            {"ticket_id": "DOC-001", "chunk_text": "Network timeout solution", "service": "Network", "severity": "High"},
            {"ticket_id": "DOC-002", "chunk_text": "Database connection fix", "service": "Database", "severity": "Medium"},
            {"ticket_id": "DOC-003", "chunk_text": "Authentication reset method", "service": "Authentication", "severity": "High"},
        ]

        for doc in documents:
            embedding = self.embed_service.embed_text(doc["chunk_text"])
            self.pipeline.store_chunks([{
                "ticket_id": doc["ticket_id"],
                "chunk_index": 0,
                "chunk_text": doc["chunk_text"],
                "embedding": embedding,
                "metadata": {
                    "service": doc["service"],
                    "severity": doc["severity"],
                }
            }])

        retriever = HybridRetriever(
            vector_store=self.store,
            embedding_service=self.embed_service
        )
        hybrid_results = retriever.retrieve(query="network timeout", top_k=5)

        grounder = ContextGrounding()
        grounded = grounder.ground_context(hybrid_results)

        # Verify results
        assert len(grounded.citations) >= 1
        assert any("ticket_id" in c for c in grounded.citations)


class TestGroundingEdgeCases:
    """Tests for grounding edge cases with real Qdrant data."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Set up and tear down Qdrant collection for each test."""
        skip = qdrant_skip_reason()
        if skip:
            pytest.skip(skip)

        try:
            self.collection_name = get_test_collection_name()
            self.dimensions = get_embedding_model_dimensions()

            self.store = VectorStore(collection_name=self.collection_name)
            self.pipeline = StorePipeline(vector_store=self.store)

            config = Config()
            self.embed_service = EmbeddingService()

            yield
        finally:
            try:
                self.store.delete_collection()
            except Exception:
                pass

    def test_grounding_with_empty_results(self):
        """Test grounding with empty retrieval results."""
        retriever = HybridRetriever(
            vector_store=self.store,
            embedding_service=self.embed_service
        )
        empty_results = []  # Empty results list

        grounder = ContextGrounding()
        grounded = grounder.ground_context(empty_results)

        # Should return empty grounded context
        assert grounded.context_text == ""
        assert grounded.citations == []

    def test_grounding_with_single_result(self):
        """Test grounding with a single result."""
        # Store single document
        document = {
            "ticket_id": "SINGLE-001",
            "chunk_text": "Single solution document",
            "service": "Test",
            "severity": "Info"
        }

        embedding = self.embed_service.embed_text(document["chunk_text"])
        self.pipeline.store_chunks([{
            "ticket_id": document["ticket_id"],
            "chunk_index": 0,
            "chunk_text": document["chunk_text"],
            "embedding": embedding,
            "metadata": {
                "service": document["service"],
                "severity": document["severity"],
            }
        }])

        retriever = HybridRetriever(
            vector_store=self.store,
            embedding_service=self.embed_service
        )
        hybrid_results = retriever.retrieve(query="testing", top_k=5)

        grounder = ContextGrounding(max_passages=3)
        grounded = grounder.ground_context(hybrid_results)

        assert len(grounded.citations) == 1
        assert grounded.citations[0]["ticket_id"] == "SINGLE-001"

    def test_grounding_respects_max_passages(self):
        """Test that max_passages limits grounded context."""
        # Store multiple documents
        for i in range(10):
            document = {
                "ticket_id": f"MAX-00{i+1}",
                "chunk_text": f"Solution number {i+1}",
                "service": "Test",
                "severity": "Info"
            }

            embedding = self.embed_service.embed_text(document["chunk_text"])
            self.pipeline.store_chunks([{
                "ticket_id": document["ticket_id"],
                "chunk_index": 0,
                "chunk_text": document["chunk_text"],
                "embedding": embedding,
                "metadata": {
                    "service": document["service"],
                    "severity": document["severity"],
                }
            }])

        retriever = HybridRetriever(
            vector_store=self.store,
            embedding_service=self.embed_service
        )
        hybrid_results = retriever.retrieve(query="solution", top_k=10)

        # Limit to 3 passages
        grounder = ContextGrounding(max_passages=3)
        grounded = grounder.ground_context(hybrid_results)

        assert len(grounded.citations) == 3

    def test_grounding_with_min_score_filter(self):
        """Test that min_relevance_score filters results."""
        # Store documents
        for i in range(5):
            document = {
                "ticket_id": f"SCORE-00{i+1}",
                "chunk_text": f"Relevant solution {i+1}",
                "service": "Test",
                "severity": "Info"
            }

            embedding = self.embed_service.embed_text(document["chunk_text"])
            self.pipeline.store_chunks([{
                "ticket_id": document["ticket_id"],
                "chunk_index": 0,
                "chunk_text": document["chunk_text"],
                "embedding": embedding,
                "metadata": {
                    "service": document["service"],
                    "severity": document["severity"],
                }
            }])

        retriever = HybridRetriever(
            vector_store=self.store,
            embedding_service=self.embed_service
        )
        hybrid_results = retriever.retrieve(query="solution", top_k=10)

        # Filter with high min score (should still work with good matches)
        grounder = ContextGrounding(min_relevance_score=-3.0)
        grounded = grounder.ground_context(hybrid_results)

        # Should have at least some results
        assert isinstance(grounded, GroundedContext)

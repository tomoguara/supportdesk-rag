"""Simple functional tests for Stage 6: Ground - Context Grounding.

These tests verify the basic ground functionality without requiring Qdrant.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestGroundSimple:
    """Simple functional tests for context grounding."""

    def test_ground_context_basic_workflow(self):
        """Test basic context grounding workflow."""
        from supportdesk_rag.supportdesk_ground import ContextGrounding, GroundedContext

        grounder = ContextGrounding(max_passages=3)

        reranked_results = [
            {
                "rerank_score": 8.5,
                "payload": {
                    "ticket_id": "TICK-001",
                    "chunk_text": "Network timeout error when accessing API",
                },
                "retriever": "vector"
            },
            {
                "rerank_score": 7.2,
                "payload": {
                    "ticket_id": "TICK-002",
                    "chunk_text": "Database connection slow",
                },
                "retriever": "bm25"
            }
        ]

        grounded = grounder.ground_context(reranked_results)

        assert isinstance(grounded, GroundedContext)
        assert len(grounded.citations) == 2
        assert "[CITATION:TICK-001]" in grounded.context_text
        assert "[CITATION:TICK-002]" in grounded.context_text
        assert "Network timeout" in grounded.context_text

    def test_pipeline_integration_without_qdrant(self):
        """Test that pipeline stages can be used together without Qdrant."""
        from supportdesk_rag.supportdesk_ground import ContextGrounding

        # Simulate reranked results (simulating Stage 5 output)
        reranked_results = [
            {
                "rerank_score": 9.0,
                "payload": {"ticket_id": "TICK-1", "chunk_text": "Solution A"},
                "retriever": "vector"
            },
            {
                "rerank_score": 7.0,
                "payload": {"ticket_id": "TICK-2", "chunk_text": "Solution B"},
                "retriever": "bm25"
            }
        ]

        grounder = ContextGrounding(max_passages=5, max_tokens=3000)
        grounded = grounder.ground_context(reranked_results)

        # Verify context is built correctly
        assert grounded.total_tokens > 0
        assert len(grounded.citations) == 2
        assert grounded.relevance_scores == [9.0, 7.0]

    def test_ground_with_stats(self):
        """Test getting statistics from grounded context."""
        from supportdesk_rag.supportdesk_ground import ContextGrounding

        grounder = ContextGrounding()

        reranked_results = [
            {"rerank_score": 8.5, "payload": {"ticket_id": "T-1", "chunk_text": "Test"}},
            {"rerank_score": 7.0, "payload": {"ticket_id": "T-2", "chunk_text": "Test"}},
            {"rerank_score": 5.5, "payload": {"ticket_id": "T-3", "chunk_text": "Test"}},
        ]

        grounded = grounder.ground_context(reranked_results)
        stats = grounder.get_stats(grounded)

        assert stats["num_citations"] == 3
        assert stats["total_tokens"] > 0
        assert stats["average_score"] == pytest.approx(7.0)
        assert stats["min_score"] == 5.5
        assert stats["max_score"] == 8.5


class TestGroundCitationTracking:
    """Tests for citation tracking functionality."""

    def test_citation_id_format(self):
        """Test that citation IDs are formatted correctly."""
        from supportdesk_rag.supportdesk_ground import ContextGrounding

        grounder = ContextGrounding()

        reranked_results = [
            {"rerank_score": 8.5, "payload": {"ticket_id": "INC-123", "chunk_text": "Test"}}
        ]

        grounded = grounder.ground_context(reranked_results)

        assert grounded.citations[0]["citation_id"] == "[CITATION:INC-123]"
        assert grounded.citations[0]["ticket_id"] == "INC-123"

    def test_relevance_score_ordering(self):
        """Test that results are sorted by relevance score."""
        from supportdesk_rag.supportdesk_ground import ContextGrounding

        grounder = ContextGrounding()

        # Results in random order
        reranked_results = [
            {"rerank_score": 5.0, "payload": {"ticket_id": "LOW", "chunk_text": "Test"}},
            {"rerank_score": 9.0, "payload": {"ticket_id": "HIGH", "chunk_text": "Test"}},
            {"rerank_score": 7.0, "payload": {"ticket_id": "MED", "chunk_text": "Test"}},
        ]

        grounded = grounder.ground_context(reranked_results)

        # Verify scores are in descending order
        scores = [c["score"] for c in grounded.citations]
        assert scores == [9.0, 7.0, 5.0]

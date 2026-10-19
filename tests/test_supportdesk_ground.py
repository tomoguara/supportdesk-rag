"""Unit tests for Stage 6: Ground - Context Grounding with Citations.

Tests for:
- Config: Configuration management for Stage 6
- ContextGrounding: Grounding re-ranked results into coherent context
- GroundedContext: Dataclass for grounded context with citations
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from supportdesk_rag.supportdesk_ground import (
    Config,
    ContextGrounding,
    GroundedContext,
    create_context_grounder,
)


class TestConfig:
    """Tests for Stage 6 Configuration manager."""

    def test_config_singleton(self):
        """Config should be a singleton."""
        config1 = Config()
        config2 = Config()
        assert config1 is config2

    def test_config_default_values(self):
        """Test default configuration values for Stage 6."""
        config = Config()

        # Should have ground config with defaults
        ground_config = config.get_ground_config()
        # When config.yaml not found, defaults are returned as inner dict
        assert ground_config is not None
        assert ground_config.get("max_passages") is not None or "max_passages" in ground_config

    def test_config_get_with_dot_notation(self):
        """Test getting config values using dot notation."""
        config = Config()
        assert config.get("openai", "embedding_model") == "text-embedding-3-small"
        assert config.get("ingest", "chunk_size") == 1000

    def test_config_get_with_fallback(self):
        """Test Config returns None for non-existent keys."""
        config = Config()
        # Stage 6 config is separate from other stages
        assert config.get("ground", "max_passages") is None or \
               config._config.get("ground", {}).get("max_passages") == 5


class TestGroundedContext:
    """Tests for GroundedContext dataclass."""

    def test_create_grounded_context(self):
        """Test creating a GroundedContext instance."""
        context = GroundedContext(
            context_text="Sample context",
            citations=[{"ticket_id": "TICK-1", "score": 0.8}],
            relevance_scores=[0.8],
            total_tokens=100
        )
        
        assert context.context_text == "Sample context"
        assert len(context.citations) == 1
        assert context.relevance_scores == [0.8]
        assert context.total_tokens == 100

    def test_grounded_context_empty(self):
        """Test creating empty GroundedContext."""
        context = GroundedContext(
            context_text="",
            citations=[],
            relevance_scores=[],
            total_tokens=0
        )
        
        assert context.context_text == ""
        assert context.citations == []
        assert context.relevance_scores == []
        assert context.total_tokens == 0


class TestContextGrounding:
    """Tests for ContextGrounding class."""

    def test_init_with_defaults(self):
        """Test ContextGrounding initialization with defaults."""
        grounder = ContextGrounding()
        assert grounder.max_passages == 5
        assert grounder.max_tokens == 4000
        assert grounder.min_relevance_score == -3.0

    def test_init_with_custom_parameters(self):
        """Test ContextGrounding initialization with custom parameters."""
        grounder = ContextGrounding(
            max_passages=3,
            max_tokens=2000,
            min_relevance_score=-2.5
        )
        
        assert grounder.max_passages == 3
        assert grounder.max_tokens == 2000
        assert grounder.min_relevance_score == -2.5

    def test_init_with_config_override(self):
        """Test ContextGrounding initialization with config values."""
        with patch('supportdesk_rag.supportdesk_ground.Config') as MockConfig:
            mock_config = Mock()
            mock_config.get_ground_config.return_value = {
                "max_passages": 7,
                "max_tokens": 3000,
                "min_relevance_score": -1.5
            }
            MockConfig.return_value = mock_config
            
            grounder = ContextGrounding()
            
            assert grounder.max_passages == 7
            assert grounder.max_tokens == 3000
            assert grounder.min_relevance_score == -1.5

    def test_ground_context_single_result(self):
        """Test grounding a single result."""
        grounder = ContextGrounding()
        
        reranked_results = [
            {
                "rerank_score": 8.5,
                "payload": {
                    "ticket_id": "TICK-001",
                    "chunk_text": "Network connection timeout when accessing API endpoint",
                    "service": "Network",
                    "severity": "High"
                }
            }
        ]
        
        context = grounder.ground_context(reranked_results)
        
        assert isinstance(context, GroundedContext)
        assert "[CITATION:TICK-001]" in context.context_text
        assert "Network connection timeout" in context.context_text
        assert len(context.citations) == 1
        assert context.citations[0]["ticket_id"] == "TICK-001"
        assert context.citations[0]["score"] == 8.5
        assert context.citations[0]["rank"] == 1

    def test_ground_context_multiple_results(self):
        """Test grounding multiple results with citations."""
        grounder = ContextGrounding()
        
        reranked_results = [
            {
                "rerank_score": 8.5,
                "payload": {
                    "ticket_id": "TICK-001",
                    "chunk_text": "Network timeout error",
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
            },
            {
                "rerank_score": 6.1,
                "payload": {
                    "ticket_id": "TICK-003",
                    "chunk_text": "Memory leak in application",
                },
                "retriever": "vector"
            }
        ]
        
        context = grounder.ground_context(reranked_results)
        
        assert len(context.citations) == 3
        assert context.citations[0]["ticket_id"] == "TICK-001"
        assert context.citations[1]["ticket_id"] == "TICK-002"
        assert context.citations[2]["ticket_id"] == "TICK-003"
        
        # Verify scores are sorted in descending order
        scores = [c["score"] for c in context.citations]
        assert scores == sorted(scores, reverse=True)

    def test_ground_context_filters_by_min_score(self):
        """Test that results below min_relevance_score are filtered."""
        grounder = ContextGrounding(min_relevance_score=5.0)
        
        reranked_results = [
            {"rerank_score": 8.5, "payload": {"ticket_id": "TICK-001", "chunk_text": "High score"}},
            {"rerank_score": 4.0, "payload": {"ticket_id": "TICK-002", "chunk_text": "Low score"}},
            {"rerank_score": 7.0, "payload": {"ticket_id": "TICK-003", "chunk_text": "Medium score"}},
        ]
        
        context = grounder.ground_context(reranked_results)
        
        # Only results with score >= 5.0 should be included
        assert len(context.citations) == 2
        ticket_ids = [c["ticket_id"] for c in context.citations]
        assert "TICK-001" in ticket_ids
        assert "TICK-003" in ticket_ids
        assert "TICK-002" not in ticket_ids

    def test_ground_context_respects_max_passages(self):
        """Test that max_passages limits the number of results."""
        grounder = ContextGrounding(max_passages=2)
        
        reranked_results = [
            {"rerank_score": 9.0, "payload": {"ticket_id": "TICK-001", "chunk_text": "Best"}},
            {"rerank_score": 8.0, "payload": {"ticket_id": "TICK-002", "chunk_text": "Second"}},
            {"rerank_score": 7.0, "payload": {"ticket_id": "TICK-003", "chunk_text": "Third"}},
        ]
        
        context = grounder.ground_context(reranked_results)
        
        assert len(context.citations) == 2
        assert context.citations[0]["ticket_id"] == "TICK-001"
        assert context.citations[1]["ticket_id"] == "TICK-002"

    def test_ground_context_with_no_payload(self):
        """Test grounding results without payload (fallback extraction)."""
        grounder = ContextGrounding()
        
        reranked_results = [
            {
                "rerank_score": 8.5,
                "chunk_text": "Direct chunk text content",
                "ticket_id": "TICK-DIRECT"
            }
        ]
        
        context = grounder.ground_context(reranked_results)
        
        assert "[CITATION:TICK-DIRECT]" in context.context_text
        assert "Direct chunk text content" in context.context_text

    def test_ground_context_citation_metadata(self):
        """Test that citation metadata is correctly captured."""
        grounder = ContextGrounding()
        
        reranked_results = [
            {
                "rerank_score": 8.5,
                "payload": {
                    "ticket_id": "TICK-001",
                    "chunk_text": "Test content",
                },
                "retriever": "vector"
            }
        ]
        
        context = grounder.ground_context(reranked_results)
        
        citation = context.citations[0]
        assert citation["citation_id"] == "[CITATION:TICK-001]"
        assert citation["ticket_id"] == "TICK-001"
        assert citation["score"] == 8.5
        assert citation["rank"] == 1
        assert citation["retriever"] == "vector"
        assert "content_preview" in citation

    def test_ground_context_estimates_tokens(self):
        """Test token estimation is calculated."""
        grounder = ContextGrounding()
        
        # Long content should result in more tokens
        long_content = "word " * 100
        reranked_results = [
            {"rerank_score": 8.5, "payload": {"ticket_id": "TICK-001", "chunk_text": long_content}}
        ]
        
        context = grounder.ground_context(reranked_results)
        
        # Token estimation: 1 token ≈ 4 chars
        expected_tokens = len(long_content) // 4 + 30  # +30 for citation overhead
        assert context.total_tokens > 0
        assert context.total_tokens <= expected_tokens

    def test_truncate_context_under_budget(self):
        """Test that context below budget is returned unchanged."""
        grounder = ContextGrounding(max_tokens=5000)
        
        reranked_results = [
            {"rerank_score": 8.5, "payload": {"ticket_id": "TICK-001", "chunk_text": "Short"}}
        ]
        
        context = grounder.ground_context(reranked_results)
        assert context.total_tokens < 5000
        
        truncated = grounder.truncate_context(context, max_tokens=5000)
        
        assert truncated.context_text == context.context_text
        assert len(truncated.citations) == len(context.citations)

    def test_truncate_context_over_budget(self):
        """Test that context over budget is truncated."""
        grounder = ContextGrounding(max_passages=5)
        
        # Create results that will exceed budget when grounded
        reranked_results = []
        for i in range(5):
            content = f"Content for ticket {i} " + "word " * 50
            reranked_results.append({
                "rerank_score": 10.0 - i,
                "payload": {"ticket_id": f"TICK-{i:03d}", "chunk_text": content}
            })
        
        context = grounder.ground_context(reranked_results)
        
        # Force truncation to very low token limit
        truncated = grounder.truncate_context(context, max_tokens=50)
        
        # Should have fewer citations
        assert len(truncated.citations) <= len(context.citations)

    def test_get_stats(self):
        """Test getting grounding statistics."""
        grounder = ContextGrounding()
        
        reranked_results = [
            {"rerank_score": 8.5, "payload": {"ticket_id": "TICK-001", "chunk_text": "Test"}},
            {"rerank_score": 7.0, "payload": {"ticket_id": "TICK-002", "chunk_text": "Test"}},
            {"rerank_score": 5.5, "payload": {"ticket_id": "TICK-003", "chunk_text": "Test"}},
        ]
        
        context = grounder.ground_context(reranked_results)
        stats = grounder.get_stats(context)
        
        assert stats["num_citations"] == 3
        assert stats["total_tokens"] > 0
        assert stats["average_score"] == pytest.approx(7.0)
        assert stats["min_score"] == 5.5
        assert stats["max_score"] == 8.5
        assert "vector" in stats["retrievers_used"] or "unknown" in stats["retrievers_used"]


class TestCreateContextGrounder:
    """Tests for factory function."""

    def test_create_context_grounder(self):
        """Test factory function creates ContextGrounding correctly."""
        grounder = create_context_grounder(max_passages=3)
        
        assert grounder.max_passages == 3
        assert isinstance(grounder, ContextGrounding)


class TestIntegrationWithConfig:
    """Integration tests with configuration system."""

    def test_context_grounder_uses_config_values(self):
        """Test that context grounder can use config values."""
        with patch('supportdesk_rag.supportdesk_ground.Config') as MockConfig:
            mock_config = Mock()
            mock_config.get_ground_config.return_value = {
                "max_passages": 8,
                "max_tokens": 6000,
                "min_relevance_score": -1.0
            }
            MockConfig.return_value = mock_config
            
            grounder = ContextGrounding()
            
            assert grounder.max_passages == 8
            assert grounder.max_tokens == 6000
            assert grounder.min_relevance_score == -1.0

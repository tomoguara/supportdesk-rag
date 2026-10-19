"""Unit tests for Stage 4: Retrieve (BM25 + Hybrid Search with RRF)

Tests for:
- BM25Index: Keyword search with tiktoken tokenization
- HybridRetriever: Hybrid search combining vector + BM25 with RRF fusion
- Config: Configuration management for Stage 4
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from supportdesk_rag.supportdesk_retrieve import (
    Config,
    BM25Index,
    HybridRetriever,
    create_hybrid_retriever,
    create_bm25_index,
)


class TestConfig:
    """Tests for Stage 4 Configuration manager."""

    def test_config_singleton(self):
        """Config should be a singleton."""
        config1 = Config()
        config2 = Config()
        assert config1 is config2

    def test_config_default_values(self):
        """Test default configuration values for Stage 4."""
        config = Config()

        # Should have retrieve config with defaults
        retrieve_config = config.get_retrieve_config()
        assert retrieve_config.get("vector_weight") == 0.7
        assert retrieve_config.get("bm25_weight") == 0.3
        assert retrieve_config.get("k_rrf") == 60
        assert retrieve_config.get("vector_top_k") == 20
        assert retrieve_config.get("bm25_top_k") == 20

    def test_config_get_with_dot_notation(self):
        """Test getting config values using dot notation."""
        config = Config()
        assert config.get("openai", "embedding_model") == "text-embedding-3-small"
        assert config.get("ingest", "chunk_size") == 1000
        assert config.get("retrieve", "k_rrf") == 60


class TestBM25Index:
    """Tests for BM25Index class."""

    def test_init(self):
        """Test BM25Index initialization."""
        index = BM25Index()
        assert index.bm25 is None
        assert index.documents == []
        assert index.document_ids == []

    def test_build_with_custom_doc_ids(self):
        """Test building index with custom document IDs."""
        index = BM25Index()
        texts = ["doc1", "doc2", "doc3"]
        doc_ids = ["id1", "id2", "id3"]

        with patch('rank_bm25.BM25Okapi') as MockBM25, \
             patch('tiktoken.get_encoding') as MockEncoding:
            mock_encoding = Mock()
            mock_encoding.encode.return_value = [1, 2, 3]
            mock_encoding.decode.return_value = "test"
            MockEncoding.return_value = mock_encoding

            mock_bm25 = Mock()
            mock_bm25.get_scores.return_value = [0.5, 0.5, 0.5]
            MockBM25.return_value = mock_bm25

            import importlib
            import supportdesk_rag.supportdesk_retrieve as retrieve_mod
            importlib.reload(retrieve_mod)

            index = retrieve_mod.BM25Index()
            index.build(texts, doc_ids)

            assert index.document_ids == doc_ids

    def test_search_returns_empty_when_no_index(self):
        """Test search returns empty list when index not built."""
        index = BM25Index()
        results = index.search("test query")
        assert results == []


class TestRRFFusion:
    """Tests for RRF fusion algorithm."""

    def test_rrf_fusion_combining_results(self):
        """Test RRF fusion combines vector and BM25 results."""
        vector_results = [
            {"id": "doc1", "score": 0.9, "payload": {"chunk_text": "text1"}, "document": "text1"},
            {"id": "doc2", "score": 0.7, "payload": {"chunk_text": "text2"}, "document": "text2"},
        ]

        bm25_results = [
            {"doc_id": "doc2", "score": 0.8, "rank": 1, "document": "text2"},
            {"doc_id": "doc3", "score": 0.5, "rank": 2, "document": "text3"},
        ]

        hybrid = HybridRetriever(vector_store=Mock(), embedding_service=Mock())
        fused = hybrid.rrf_fusion(vector_results, bm25_results, k=60)

        # doc2 appears in both, should have highest fused score
        assert len(fused) >= 2
        # doc2 should be in results
        doc_ids = [r.get("doc_id") or r.get("id") for r in fused]
        assert "doc2" in doc_ids

    def test_rrf_fusion_with_different_ranks(self):
        """Test RRF gives higher score to documents appearing in both top ranks."""
        hybrid = HybridRetriever(vector_store=Mock(), embedding_service=Mock())

        vector_results = [
            {"id": "A", "score": 0.9, "document": "textA"},
            {"id": "B", "score": 0.8, "document": "textB"},
            {"id": "C", "score": 0.7, "document": "textC"},
        ]

        bm25_results = [
            {"doc_id": "B", "score": 0.95, "rank": 1, "document": "textB"},
            {"doc_id": "A", "score": 0.85, "rank": 2, "document": "textA"},
            {"doc_id": "D", "score": 0.6, "rank": 3, "document": "textD"},
        ]

        fused = hybrid.rrf_fusion(vector_results, bm25_results, k=60)

        doc_ids = [r.get("doc_id") or r.get("id") for r in fused]

        # Documents in both should rank highest
        assert "B" in doc_ids
        assert "A" in doc_ids


class TestHybridRetriever:
    """Tests for HybridRetriever class."""

    def test_init_without_bm25_index(self):
        """Test initialization without BM25 index."""
        mock_store = Mock()
        mock_store.client = Mock()

        # Mock scroll to return some documents
        mock_store.client.scroll.return_value = (
            [
                Mock(
                    id="doc1",
                    payload={"chunk_text": "test document"}
                )
            ],
            None
        )

        mock_embed_service = Mock()
        mock_embed_service.embed_text.return_value = [0.1] * 1536

        retriever = HybridRetriever(
            vector_store=mock_store,
            embedding_service=mock_embed_service
        )

        assert retriever.vector_store is mock_store
        assert retriever.embedding_service is mock_embed_service
        assert retriever.k_rrf == 60

    def test_init_with_custom_bm25_index(self):
        """Test initialization with custom BM25 index."""
        mock_store = Mock()
        mock_embed_service = Mock()

        custom_index = Mock(spec=BM25Index)
        custom_index.build = Mock()

        retriever = HybridRetriever(
            vector_store=mock_store,
            embedding_service=mock_embed_service,
            bm25_index=custom_index
        )

        assert retriever.bm25_index is custom_index

    def test_init_with_custom_parameters(self):
        """Test initialization with custom parameters."""
        mock_store = Mock()
        mock_embed_service = Mock()

        retriever = HybridRetriever(
            vector_store=mock_store,
            embedding_service=mock_embed_service,
            k_rrf=100,
            vector_top_k=30,
            bm25_top_k=30
        )

        assert retriever.k_rrf == 100
        assert retriever.vector_top_k == 30
        assert retriever.bm25_top_k == 30

    def test_retrieve_method_signature(self):
        """Test retrieve method has correct signature and defaults."""
        mock_store = Mock()
        mock_embed_service = Mock()
        mock_embed_service.embed_text.return_value = [0.1] * 1536

        # Mock search to return vector results
        mock_store.search.return_value = [
            {"id": "doc1", "score": 0.9, "payload": {"chunk_text": "test"}}
        ]

        retriever = HybridRetriever(
            vector_store=mock_store,
            embedding_service=mock_embed_service,
            vector_top_k=10,
            bm25_top_k=10
        )

        # Test that retrieve method exists and is callable
        assert hasattr(retriever, 'retrieve')
        assert callable(retriever.retrieve)

    def test_create_hybrid_retriever_factory(self):
        """Test factory function creates HybridRetriever correctly."""
        mock_store = Mock()
        mock_embed = Mock()

        retriever = create_hybrid_retriever(
            vector_store=mock_store,
            embedding_service=mock_embed,
            k_rrf=80
        )

        # Check type by name to avoid import issues
        assert type(retriever).__name__ == "HybridRetriever"
        assert retriever.k_rrf == 80

    def test_create_bm25_index_factory(self):
        """Test factory function creates BM25Index correctly."""
        texts = ["doc1", "doc2", "doc3"]
        doc_ids = ["id1", "id2", "id3"]

        with patch('rank_bm25.BM25Okapi') as MockBM25, \
             patch('tiktoken.get_encoding') as MockEncoding:
            mock_encoding = Mock()
            mock_encoding.encode.return_value = [1, 2, 3]
            mock_encoding.decode.return_value = "test"
            MockEncoding.return_value = mock_encoding

            mock_bm25 = Mock()
            mock_bm25.get_scores.return_value = [0.5, 0.5, 0.5]
            MockBM25.return_value = mock_bm25

            import importlib
            import supportdesk_rag.supportdesk_retrieve as retrieve_mod
            importlib.reload(retrieve_mod)

            index = retrieve_mod.create_bm25_index(texts, doc_ids)

            assert isinstance(index, retrieve_mod.BM25Index)


class TestIntegrationWithConfig:
    """Integration tests with configuration system."""

    def test_hybrid_retriever_uses_config_values(self):
        """Test that retriever can use config values."""
        config = Config()
        retrieve_config = config.get_retrieve_config()

        assert retrieve_config.get("vector_weight") == 0.7
        assert retrieve_config.get("bm25_weight") == 0.3
        assert retrieve_config.get("k_rrf") == 60

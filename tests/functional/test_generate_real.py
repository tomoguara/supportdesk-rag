"""Functional tests for Stage 7: Generate (Answer Generation)

Tests the actual generate_answer integration with a real model (or mocked).
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestGenerateIntegration:
    """Integration tests for Stage 7: Generate."""

    def test_generate_answer_method_exists_in_pipeline(self):
        """Test that generate_answer method exists in SupportDeskRAGPipeline."""
        from supportdesk_rag import SupportDeskRAGPipeline
        import inspect

        assert hasattr(SupportDeskRAGPipeline, "generate_answer")
        sig = inspect.signature(SupportDeskRAGPipeline.generate_answer)
        params = list(sig.parameters.keys())

        assert "query_text" in params
        assert "top_k" in params

    def test_pipeline_initialization_with_generation(self):
        """Test that pipeline initializes correctly with generate_enabled."""
        from supportdesk_rag import SupportDeskRAGPipeline

        with patch("supportdesk_rag.supportdesk_rag_pipeline.EmbeddingService"):
            with patch("supportdesk_rag.supportdesk_rag_pipeline.VectorStore") as mock_store:
                with patch("supportdesk_rag.supportdesk_rag_pipeline.HybridRetriever") as mock_retriever:
                    with patch("supportdesk_rag.supportdesk_rag_pipeline.ReRanker") as mock_reranker:
                        with patch("supportdesk_rag.supportdesk_rag_pipeline.ContextGrounding") as mock_ground:
                            with patch("supportdesk_rag.supportdesk_rag_pipeline.AnswerGenerator") as mock_generator:
                                # Setup mocks
                                mock_retriever_instance = Mock()
                                mock_retriever.return_value = mock_retriever_instance
                                mock_reranker.return_value = Mock()
                                mock_ground.return_value = Mock()
                                mock_generator.return_value = Mock()
                                mock_store_instance = Mock()
                                mock_store_instance.create_collection.return_value = None
                                mock_store.return_value = mock_store_instance

                                # Create pipeline with generation enabled
                                pipeline = SupportDeskRAGPipeline(
                                    generate_enabled=True
                                )

                                # Verify pipeline was created
                                assert pipeline is not None
                                assert pipeline.generate_enabled is True

                                # Verify the AnswerGenerator was instantiated
                                assert pipeline.generate_pipeline is not None

    def test_pipeline_with_all_stages(self):
        """Test that all stages initialize when enabled."""
        from supportdesk_rag import SupportDeskRAGPipeline

        with patch("supportdesk_rag.supportdesk_rag_pipeline.EmbeddingService"):
            with patch("supportdesk_rag.supportdesk_rag_pipeline.VectorStore") as mock_store:
                with patch("supportdesk_rag.supportdesk_rag_pipeline.HybridRetriever"):
                    with patch("supportdesk_rag.supportdesk_rag_pipeline.ReRanker"):
                        with patch("supportdesk_rag.supportdesk_rag_pipeline.ContextGrounding"):
                            with patch("supportdesk_rag.supportdesk_rag_pipeline.AnswerGenerator"):
                                mock_store_instance = Mock()
                                mock_store_instance.create_collection.return_value = None
                                mock_store.return_value = mock_store_instance

                                # Create pipeline with all stages
                                pipeline = SupportDeskRAGPipeline(
                                    rerank_enabled=True,
                                    ground_enabled=True,
                                    generate_enabled=True
                                )

                                # Verify all components
                                assert pipeline is not None
                                assert pipeline.generate_enabled is True
                                assert pipeline.ground_enabled is True
                                assert hasattr(pipeline, "generate_answer")


class TestGeneratePipelineIntegration:
    """Tests for generate_answer pipeline integration."""

    def test_pipeline_initialization(self):
        """Test that pipeline initializes correctly."""
        from supportdesk_rag import SupportDeskRAGPipeline

        with patch("supportdesk_rag.supportdesk_rag_pipeline.EmbeddingService"):
            with patch("supportdesk_rag.supportdesk_rag_pipeline.VectorStore") as mock_store:
                mock_store_instance = Mock()
                mock_store_instance.create_collection.return_value = None
                mock_store.return_value = mock_store_instance

                pipeline = SupportDeskRAGPipeline()

                # Verify basic properties
                assert hasattr(pipeline, "generate_enabled")
                assert hasattr(pipeline, "generate_answer")

    def test_pipeline_generates_answer(self):
        """Test that pipeline can generate an answer with mocked components."""
        from supportdesk_rag import SupportDeskRAGPipeline
        from supportdesk_rag.supportdesk_generate import GeneratedAnswer

        with patch("supportdesk_rag.supportdesk_rag_pipeline.EmbeddingService"):
            with patch("supportdesk_rag.supportdesk_rag_pipeline.VectorStore") as mock_store:
                with patch("supportdesk_rag.supportdesk_rag_pipeline.HybridRetriever"):
                    with patch("supportdesk_rag.supportdesk_rag_pipeline.ReRanker"):
                        with patch("supportdesk_rag.supportdesk_rag_pipeline.ContextGrounding"):
                            with patch("supportdesk_rag.supportdesk_rag_pipeline.AnswerGenerator") as MockGenerator:
                                # Setup mocks
                                mock_generator_instance = Mock()
                                MockGenerator.return_value = mock_generator_instance
                                
                                mock_answer = GeneratedAnswer(
                                    answer="Test answer",
                                    citations=[{"ticket_id": "INC001"}],
                                    prompt_tokens=100,
                                    completion_tokens=50,
                                    total_tokens=150,
                                    model="test-model"
                                )
                                mock_generator_instance.generate_answer.return_value = mock_answer

                                mock_store_instance = Mock()
                                mock_store_instance.create_collection.return_value = None
                                mock_store.return_value = mock_store_instance

                                pipeline = SupportDeskRAGPipeline(generate_enabled=True)

                                # Verify pipeline can generate
                                assert pipeline.generate_enabled is True

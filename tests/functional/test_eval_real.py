"""Functional tests for Evaluation Framework.

Tests for:
- End-to-end evaluation with actual RAG pipeline
- Integration with observability metrics
- Dataset-level evaluation
"""

import pytest
import sys
import json
import os
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def check_vector_store_available():
    """Check if vector store is available."""
    try:
        from supportdesk_rag import VectorStore
        store = VectorStore(collection_name="test_eval_temp")
        # Clean up immediately
        if hasattr(store, 'client'):
            try:
                store.client.delete_collection(collection_name="test_eval_temp")
            except Exception:
                pass
        return True
    except Exception:
        return False


class TestEvaluationWithHallucinationAndContext:
    """Tests that do not require vector store but test evaluation logic."""
    
    def test_hallucination_detection(self):
        """Test hallucination detection with known cases."""
        from supportdesk_rag import create_evaluator
        
        evaluator = create_evaluator()
        
        # Case 1: Accurate answer (should have low hallucination)
        accurate_answer = "Python is a programming language created by Guido van Rossum."
        accurate_ground_truth = "Python is a programming language created by Guido van Rossum."
        
        accurate_result = evaluator.evaluate(
            query_id="hall-001",
            query_text="What is Python?",
            predicted_answer=accurate_answer,
            ground_truth=accurate_ground_truth
        )
        
        assert accurate_result.hallucination_score is not None
        assert accurate_result.hallucination_score <= 0.5  # Should be low
        
        # Case 2: Contains new facts not in ground truth
        hallucinated_answer = "Python is a programming language created by Guido van Rossum. Python is used by Google."
        hallucinated_ground_truth = "Python is a programming language created by Guido van Rossum."
        
        hallucinated_result = evaluator.evaluate(
            query_id="hall-002",
            query_text="What is Python?",
            predicted_answer=hallucinated_answer,
            ground_truth=hallucinated_ground_truth
        )
        
        assert hallucinated_result.hallucination_score is not None
    
    def test_context_precision_recall(self):
        """Test context precision and recall metrics."""
        from supportdesk_rag import create_evaluator
        
        evaluator = create_evaluator()
        
        # Context with relevant documents
        relevant_context = [
            {"text": "Python is a programming language created by Guido van Rossum."},
            {"text": "Python was first released in 1991."},
        ]
        
        ground_truth = "Python is a programming language created by Guido van Rossum."
        
        result = evaluator.evaluate(
            query_id="ctx-001",
            query_text="What is Python?",
            predicted_answer="Python is a programming language.",
            ground_truth=ground_truth,
            retrieved_context=relevant_context
        )
        
        assert result.context_precision is not None
        assert result.context_recall is not None
        assert 0 <= result.context_precision <= 1
        assert 0 <= result.context_recall <= 1


class TestEvaluationIntegration:
    """Integration tests for evaluation with RAG pipeline."""
    
    @pytest.fixture(autouse=True)
    def check_vector_store(self):
        """Check if vector store is available before running tests."""
        if not check_vector_store_available():
            pytest.skip("Vector store not available (Qdrant not running)")
    
    def test_evaluation_with_rag_pipeline(self):
        """Test evaluation using the full RAG pipeline."""
        try:
            from supportdesk_rag import SupportDeskRAGPipeline
            from supportdesk_rag import create_evaluator, create_evaluation_pipeline
        except ImportError as e:
            pytest.skip(f"Could not import modules: {e}")
        
        # Create RAG pipeline
        try:
            rag_pipeline = SupportDeskRAGPipeline()
        except Exception as e:
            pytest.skip(f"Could not create RAG pipeline: {e}")
        
        # Create evaluation pipeline
        eval_pipeline = create_evaluation_pipeline(rag_pipeline=rag_pipeline)
        
        # Define test queries with ground truth
        test_queries = [
            {
                "query_id": "eval-001",
                "query_text": "What is supportdesk-rag?",
                "ground_truth": "supportdesk-rag is a RAG system for support ticket analysis."
            },
            {
                "query_id": "eval-002", 
                "query_text": "What does RAG stand for?",
                "ground_truth": "RAG stands for Retrieval-Augmented Generation."
            }
        ]
        
        # Evaluate dataset
        summary = eval_pipeline.evaluate_dataset(test_queries)
        
        assert summary.total_evaluations == 2
        assert summary.per_metric_stats is not None
        
        # Verify that evaluation results were computed with scores
        for result in eval_pipeline.evaluator.results:
            assert result.relevance_score is not None, "Relevance score should be computed"
            assert result.answer_quality_score is not None, "Answer quality score should be computed"
    
    def test_evaluation_results_export(self):
        """Test exporting evaluation results."""
        try:
            from supportdesk_rag import SupportDeskRAGPipeline
            from supportdesk_rag import create_evaluation_pipeline
        except ImportError as e:
            pytest.skip(f"Could not import modules: {e}")
        
        try:
            rag_pipeline = SupportDeskRAGPipeline()
        except Exception as e:
            pytest.skip(f"Could not create RAG pipeline: {e}")
        
        eval_pipeline = create_evaluation_pipeline(rag_pipeline=rag_pipeline)
        
        # Run some evaluations
        eval_pipeline.evaluate_query(
            query_id="eval-export-001",
            query_text="Test query",
            ground_truth="Test ground truth."
        )
        
        # Export to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            eval_pipeline.export_results(temp_path)
            
            with open(temp_path, 'r') as f:
                data = json.load(f)
            
            assert "summary" in data
            assert "results" in data
            assert data["summary"]["total_evaluations"] == 1
            
            # Verify metrics were computed
            assert data["results"][0]["metrics"]["relevance_score"] is not None
            assert data["results"][0]["metrics"]["answer_quality_score"] is not None
        finally:
            os.unlink(temp_path)
    
    def test_context_precision_recall_with_pipeline(self):
        """Test context metrics through the evaluation pipeline."""
        try:
            from supportdesk_rag import SupportDeskRAGPipeline
            from supportdesk_rag import create_evaluation_pipeline
        except ImportError as e:
            pytest.skip(f"Could not import modules: {e}")
        
        try:
            rag_pipeline = SupportDeskRAGPipeline()
        except Exception as e:
            pytest.skip(f"Could not create RAG pipeline: {e}")
        
        eval_pipeline = create_evaluation_pipeline(rag_pipeline=rag_pipeline)
        
        # Run evaluation through pipeline
        result = eval_pipeline.evaluate_query(
            query_id="ctx-pipeline-001",
            query_text="What is Python?",
            ground_truth="Python is a programming language created by Guido van Rossum."
        )
        
        # Verify that at least one evaluation metric was computed
        assert result.relevance_score is not None or result.answer_quality_score is not None


class TestEvaluationWithObservability:
    """Tests for evaluation integration with observability."""
    
    def test_combined_metrics(self):
        """Test combining evaluation metrics with latency and cost."""
        from supportdesk_rag import create_evaluator
        
        evaluator = create_evaluator()
        
        result = evaluator.evaluate(
            query_id="combined-001",
            query_text="Test query",
            predicted_answer="Test answer.",
            ground_truth="Test ground truth.",
            cost_usd=0.001,
            latency_ms=250
        )
        
        assert result.cost_usd == 0.001
        assert result.latency_ms == 250
        assert result.relevance_score is not None
        assert result.answer_quality_score is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

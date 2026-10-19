"""Unit tests for Stage 9: Evaluation Framework.

Tests for:
- Evaluator: Core evaluation metrics and logic
- EvaluationPipeline: Integration with RAG pipeline
- EvaluationResult: Data structure validation
"""

import pytest
import sys
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from supportdesk_rag.supportdesk_eval import (
    EvaluationResult,
    EvaluationSummary,
    Evaluator,
    EvaluationPipeline,
    EvaluationMetric,
    create_evaluator,
    compute_basic_metrics,
)


class TestEvaluationMetric:
    """Tests for EvaluationMetric enum."""
    
    def test_metric_enum_values(self):
        """Test that all expected metrics are defined."""
        assert EvaluationMetric.RELEVANCE.value == "relevance"
        assert EvaluationMetric.ANSWER_QUALITY.value == "answer_quality"
        assert EvaluationMetric.HALLUCINATION.value == "hallucination"
        assert EvaluationMetric.CONTEXT_PRECISION.value == "context_precision"
        assert EvaluationMetric.CONTEXT_RECALL.value == "context_recall"
        assert EvaluationMetric.COST.value == "cost"
        assert EvaluationMetric.LATENCY.value == "latency"


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""
    
    def test_result_creation(self):
        """Test basic result creation."""
        result = EvaluationResult(
            query_id="test-001",
            query_text="What is Python?",
            predicted_answer="Python is a programming language."
        )
        assert result.query_id == "test-001"
        assert result.query_text == "What is Python?"
        assert result.predicted_answer == "Python is a programming language."
        assert result.relevance_score is None
        assert result.ground_truth is None
    
    def test_result_with_all_fields(self):
        """Test result with all fields populated."""
        result = EvaluationResult(
            query_id="test-002",
            query_text="What is Java?",
            predicted_answer="Java is a programming language.",
            ground_truth="Java is a programming language Developed by Sun Microsystems.",
            retrieved_context=[{"text": "Java..."}],
            relevance_score=0.85,
            answer_quality_score=0.9,
            hallucination_score=0.1,
            context_precision=0.9,
            context_recall=0.8,
            cost_usd=0.001,
            latency_ms=150,
            model="gpt-4"
        )
        assert result.relevance_score == 0.85
        assert result.model == "gpt-4"
        assert result.timestamp is not None
    
    def test_result_with_metadata(self):
        """Test result with custom metadata."""
        result = EvaluationResult(
            query_id="test-003",
            query_text="Test",
            predicted_answer="Answer",
            metadata={"user_id": "user-123", "session_id": "sess-456"}
        )
        assert result.metadata == {"user_id": "user-123", "session_id": "sess-456"}


class TestEvaluator:
    """Tests for the Evaluator class."""
    
    def test_evaluator_creation(self):
        """Test evaluator initialization."""
        evaluator = create_evaluator()
        assert evaluator.results == []
        assert evaluator.threshold_relevance == 0.5
        assert evaluator.threshold_answer_quality == 0.6
        assert evaluator.threshold_hallucination == 0.3
    
    def test_evaluator_with_custom_thresholds(self):
        """Test evaluator with custom thresholds."""
        evaluator = create_evaluator({
            "threshold_relevance": 0.7,
            "threshold_answer_quality": 0.8,
            "threshold_hallucination": 0.2
        })
        assert evaluator.threshold_relevance == 0.7
        assert evaluator.threshold_answer_quality == 0.8
        assert evaluator.threshold_hallucination == 0.2
    
    def test_evaluate_basic(self):
        """Test basic evaluation without ground truth."""
        evaluator = create_evaluator()
        result = evaluator.evaluate(
            query_id="test-001",
            query_text="What is Python?",
            predicted_answer="Python is a programming language."
        )
        assert len(evaluator.results) == 1
        assert result.query_id == "test-001"
        # Without ground truth, most scores should be None
        assert result.relevance_score is None
        assert result.hallucination_score is None
    
    def test_evaluate_with_ground_truth(self):
        """Test evaluation with ground truth."""
        evaluator = create_evaluator()
        result = evaluator.evaluate(
            query_id="test-002",
            query_text="What is Python?",
            predicted_answer="Python is a programming language developed by Guido van Rossum.",
            ground_truth="Python is a programming language created by Guido van Rossum.",
            latency_ms=120
        )
        assert result.relevance_score is not None
        assert result.answer_quality_score is not None
        assert result.hallucination_score is not None
        assert result.latency_ms == 120
    
    def test_evaluate_with_context(self):
        """Test evaluation with retrieved context."""
        evaluator = create_evaluator()
        context = [
            {"text": "Python is a programming language created by Guido van Rossum in 1991."},
            {"text": "Python is known for its simplicity and readability."}
        ]
        result = evaluator.evaluate(
            query_id="test-003",
            query_text="What is Python?",
            predicted_answer="Python is a programming language.",
            ground_truth="Python is a programming language created by Guido van Rossum.",
            retrieved_context=context
        )
        assert result.context_precision is not None
        assert result.context_recall is not None
    
    def test_evaluator_summary_empty(self):
        """Test summary with no results."""
        evaluator = create_evaluator()
        summary = evaluator.get_summary()
        assert summary.total_evaluations == 0
        assert summary.average_relevance is None
    
    def test_evaluator_summary(self):
        """Test summary with results - ensure scores are computed by evaluator."""
        evaluator = create_evaluator()
        evaluator.evaluate(
            query_id="test-001",
            query_text="What is Python?",
            predicted_answer="Python is a programming language created by Guido van Rossum.",
            ground_truth="Python is a programming language created by Guido van Rossum."
        )
        evaluator.evaluate(
            query_id="test-002",
            query_text="What is Java?",
            predicted_answer="Java is a programming language created by James Gosling.",
            ground_truth="Java is a programming language created by James Gosling."
        )
        summary = evaluator.get_summary()
        assert summary.total_evaluations == 2
        # Both queries are identical to ground truth, so high scores expected
        assert summary.average_relevance is not None
        assert summary.average_relevance >= 0.5
        assert summary.average_answer_quality is not None
        assert summary.average_answer_quality >= 0.9
    
    def test_export_results(self):
        """Test exporting results to JSON."""
        evaluator = create_evaluator()
        evaluator.evaluate(
            query_id="test-001",
            query_text="What is Python?",
            predicted_answer="Python is a programming language.",
            ground_truth="Python is a programming language."
        )
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            evaluator.export_results(temp_path)
            
            with open(temp_path, 'r') as f:
                data = json.load(f)
            
            assert "summary" in data
            assert "results" in data
            assert data["summary"]["total_evaluations"] == 1
            assert len(data["results"]) == 1
            assert data["results"][0]["query_id"] == "test-001"
        finally:
            import os
            os.unlink(temp_path)
    
    def test_compute_basic_metrics(self):
        """Test basic metrics function."""
        answer = "Python is a programming language developed by Guido van Rossum."
        ground_truth = "Python is a programming language created by Guido van Rossum."
        
        metrics = compute_basic_metrics(answer, ground_truth)
        
        assert "relevance" in metrics
        assert "answer_quality" in metrics
        assert "hallucination" in metrics
        assert 0 <= metrics["relevance"] <= 1
        assert 0 <= metrics["answer_quality"] <= 1
        assert 0 <= metrics["hallucination"] <= 1


class TestEvaluationSummary:
    """Tests for EvaluationSummary dataclass."""
    
    def test_summary_with_none_values(self):
        """Test summary with None values."""
        summary = EvaluationSummary(total_evaluations=0)
        assert summary.total_evaluations == 0
        assert summary.average_relevance is None
    
    def test_summary_with_values(self):
        """Test summary with actual values."""
        summary = EvaluationSummary(
            total_evaluations=3,
            average_relevance=0.85,
            average_answer_quality=0.9,
            average_hallucination=0.1,
            average_context_precision=0.8,
            average_context_recall=0.75,
            total_cost_usd=0.003,
            total_latency_ms=450
        )
        assert summary.total_evaluations == 3
        assert summary.average_relevance == 0.85
        assert summary.total_cost_usd == 0.003


class TestIntegration:
    """Integration tests for evaluation with mock pipeline."""
    
    def test_evaluation_pipeline_with_mock(self):
        """Test EvaluationPipeline with a mock RAG pipeline."""
        class MockPipeline:
            def generate_answer(self, query, **kwargs):
                return {
                    "answer": {"answer": "Python is a programming language."},
                    "grounded_context": {"citations": [{"text": "Python..."}]}
                }
        
        mock_pipeline = MockPipeline()
        pipeline = EvaluationPipeline(rag_pipeline=mock_pipeline)
        
        result = pipeline.evaluate_query(
            query_id="test-001",
            query_text="What is Python?",
            ground_truth="Python is a programming language."
        )
        
        assert result.query_id == "test-001"
        assert result.relevance_score is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

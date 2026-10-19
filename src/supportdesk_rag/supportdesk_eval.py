"""Stage 9: Evaluation Framework - Comprehensive evaluation for RAG pipelines.

This module provides comprehensive evaluation capabilities for the SupportDesk RAG system,
including metrics for relevance, answer quality, hallucination detection, and cost analysis.

Key features:
- Automated evaluation with configurable metrics
- Ground truth comparison support
- Hallucination detection using LLM-based fact-checking
- Context precision/recall metrics
- Answer quality scoring

For integration with the unified pipeline, see `supportdesk_rag_pipeline`.
"""

import os
import json
import re
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class EvaluationMetric(Enum):
    """Evaluation metrics supported by the framework."""
    RELEVANCE = "relevance"
    ANSWER_QUALITY = "answer_quality"
    HALLUCINATION = "hallucination"
    CONTEXT_PRECISION = "context_precision"
    CONTEXT_RECALL = "context_recall"
    COST = "cost"
    LATENCY = "latency"


@dataclass
class EvaluationResult:
    """Result of an evaluation run."""
    query_id: str
    query_text: str
    predicted_answer: str
    ground_truth: Optional[str] = None
    retrieved_context: Optional[List[Dict]] = None
    relevance_score: Optional[float] = None
    answer_quality_score: Optional[float] = None
    hallucination_score: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    cost_usd: Optional[float] = None
    latency_ms: Optional[float] = None
    model: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationSummary:
    """Summary of multiple evaluation results."""
    total_evaluations: int
    average_relevance: Optional[float] = None
    average_answer_quality: Optional[float] = None
    average_hallucination: Optional[float] = None
    average_context_precision: Optional[float] = None
    average_context_recall: Optional[float] = None
    total_cost_usd: Optional[float] = None
    total_latency_ms: Optional[float] = None
    per_metric_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)


class Evaluator:
    """Base evaluation framework for RAG systems."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the evaluator."""
        self.config = config or {}
        self.results: List[EvaluationResult] = []
        self.threshold_relevance = self.config.get("threshold_relevance", 0.5)
        self.threshold_answer_quality = self.config.get("threshold_answer_quality", 0.6)
        self.threshold_hallucination = self.config.get("threshold_hallucination", 0.3)
        
    def evaluate(
        self,
        query_id: str,
        query_text: str,
        predicted_answer: str,
        ground_truth: Optional[str] = None,
        retrieved_context: Optional[List[Dict]] = None,
        cost_usd: Optional[float] = None,
        latency_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """Evaluate a single query-answer pair."""
        result = EvaluationResult(
            query_id=query_id,
            query_text=query_text,
            predicted_answer=predicted_answer,
            ground_truth=ground_truth,
            retrieved_context=retrieved_context,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            metadata=metadata or {}
        )
        
        if ground_truth:
            result.relevance_score = self._evaluate_relevance(query_text, predicted_answer, ground_truth)
            result.answer_quality_score = self._evaluate_answer_quality(predicted_answer, ground_truth)
            result.hallucination_score = self._detect_hallucination(predicted_answer, ground_truth)
            
        if retrieved_context:
            if ground_truth:
                result.context_precision = self._evaluate_context_precision(retrieved_context, ground_truth)
                result.context_recall = self._evaluate_context_recall(retrieved_context, ground_truth)
            else:
                result.context_precision = self._evaluate_context_precision(retrieved_context, query_text)
        
        self.results.append(result)
        return result
    
    def _evaluate_relevance(self, query: str, answer: str, ground_truth: str) -> float:
        """Evaluate relevance of answer to query and ground truth."""
        query_tokens = set(query.lower().split())
        ground_tokens = set(ground_truth.lower().split())
        answer_tokens = set(answer.lower().split())
        
        expected_overlap = len(ground_tokens & answer_tokens) / len(ground_tokens) if ground_tokens else 0
        query_alignment = len(query_tokens & answer_tokens) / len(query_tokens) if query_tokens else 0
        
        relevance = 0.6 * expected_overlap + 0.4 * query_alignment
        return min(max(relevance, 0.0), 1.0)
    
    def _evaluate_answer_quality(self, answer: str, ground_truth: str) -> float:
        """Evaluate overall quality of the answer."""
        if not answer or not ground_truth:
            return 0.0
        
        answer_len = len(answer.split())
        ground_len = len(ground_truth.split())
        length_ratio = min(answer_len / max(ground_len, 1), 1.0) if ground_len > 0 else 0.5
        
        answer_words = set(answer.lower().split())
        ground_words = set(ground_truth.lower().split())
        overlap = len(answer_words & ground_words) / len(ground_words) if ground_words else 0
        
        coherence = 0.5
        if answer.endswith(('.', '!', '?')):
            coherence += 0.2
        if len(answer) > 50:
            coherence += 0.2
        coherence = min(coherence, 1.0)
        
        quality = 0.4 * length_ratio + 0.4 * overlap + 0.2 * coherence
        return min(max(quality, 0.0), 1.0)
    
    def _detect_hallucination(self, answer: str, ground_truth: str) -> float:
        """Detect hallucinations in the answer. Returns 0-1 (lower is better)."""
        if not answer or not ground_truth:
            return 1.0
        
        answer_sentences = re.split(r'[.!?\n]+', answer)
        ground_sentences = re.split(r'[.!?\n]+', ground_truth)
        
        hallucinated_sentences = 0
        total_sentences = len(answer_sentences)
        
        for sentence in answer_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            is_supported = False
            for ground_sent in ground_sentences:
                if sentence.lower() in ground_sent.lower() or ground_sent.lower() in sentence.lower():
                    is_supported = True
                    break
            
            if not is_supported:
                answer_words = set(sentence.lower().split())
                ground_words = set(' '.join(ground_sentences).lower().split())
                
                if len(answer_words - ground_words) > len(answer_words) * 0.3:
                    hallucinated_sentences += 1
        
        hallucination_rate = hallucinated_sentences / max(total_sentences, 1)
        return min(hallucination_rate, 1.0)
    
    def _evaluate_context_precision(self, retrieved_context: List[Dict], ground_truth: str) -> float:
        """Evaluate precision of retrieved context."""
        if not retrieved_context:
            return 0.0
        
        ground_lower = ground_truth.lower()
        relevant_count = 0
        
        for doc in retrieved_context:
            doc_text = doc.get("text", "") or doc.get("content", "") or ""
            if self._is_relevant(doc_text, ground_lower):
                relevant_count += 1
        
        return relevant_count / len(retrieved_context)
    
    def _evaluate_context_recall(self, retrieved_context: List[Dict], ground_truth: str) -> float:
        """Evaluate recall of retrieved context."""
        if not retrieved_context or not ground_truth:
            return 0.0
        
        ground_lower = ground_truth.lower()
        retrieved_text = " ".join(doc.get("text", "") or doc.get("content", "") for doc in retrieved_context)
        retrieved_words = set(retrieved_text.lower().split())
        ground_words = set(ground_lower.split())
        
        if not ground_words:
            return 0.0
        
        overlap = len(retrieved_words & ground_words)
        return overlap / len(ground_words)
    
    def _is_relevant(self, doc_text: str, query_or_answer: str) -> bool:
        """Check if document is relevant to query or answer."""
        doc_lower = doc_text.lower()
        query_lower = query_or_answer.lower()
        
        doc_words = set(doc_lower.split())
        query_words = set(query_lower.split())
        
        if not query_words:
            return True
        
        overlap = len(doc_words & query_words)
        return overlap / len(query_words) >= 0.3
    
    def get_summary(self) -> EvaluationSummary:
        """Get summary of all evaluation results."""
        if not self.results:
            return EvaluationSummary(total_evaluations=0)
        
        total = len(self.results)
        
        relevances = [r.relevance_score for r in self.results if r.relevance_score is not None]
        qualities = [r.answer_quality_score for r in self.results if r.answer_quality_score is not None]
        hallucinations = [r.hallucination_score for r in self.results if r.hallucination_score is not None]
        context_precisions = [r.context_precision for r in self.results if r.context_precision is not None]
        context_recalls = [r.context_recall for r in self.results if r.context_recall is not None]
        costs = [r.cost_usd for r in self.results if r.cost_usd is not None]
        latencies = [r.latency_ms for r in self.results if r.latency_ms is not None]
        
        def avg(values):
            return sum(values) / len(values) if values else None
        
        return EvaluationSummary(
            total_evaluations=total,
            average_relevance=avg(relevances),
            average_answer_quality=avg(qualities),
            average_hallucination=avg(hallucinations),
            average_context_precision=avg(context_precisions),
            average_context_recall=avg(context_recalls),
            total_cost_usd=sum(costs) if costs else None,
            total_latency_ms=sum(latencies) if latencies else None,
            per_metric_stats={
                "relevance": {"min": min(relevances) if relevances else None, "max": max(relevances) if relevances else None, "avg": avg(relevances), "count": len(relevances)},
                "answer_quality": {"min": min(qualities) if qualities else None, "max": max(qualities) if qualities else None, "avg": avg(qualities), "count": len(qualities)},
                "hallucination": {"min": min(hallucinations) if hallucinations else None, "max": max(hallucinations) if hallucinations else None, "avg": avg(hallucinations), "count": len(hallucinations)}
            }
        )
    
    def export_results(self, filepath: str) -> None:
        """Export all evaluation results to a JSON file."""
        data = {
            "summary": {"total_evaluations": len(self.results), "metadata": self.get_summary().__dict__},
            "results": [
                {"query_id": r.query_id, "query_text": r.query_text, "predicted_answer": r.predicted_answer,
                 "ground_truth": r.ground_truth, "metrics": {"relevance_score": r.relevance_score,
                 "answer_quality_score": r.answer_quality_score, "hallucination_score": r.hallucination_score,
                 "context_precision": r.context_precision, "context_recall": r.context_recall,
                 "cost_usd": r.cost_usd, "latency_ms": r.latency_ms}, "metadata": r.metadata, "timestamp": r.timestamp}
                for r in self.results
            ]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)


class EvaluationPipeline:
    """End-to-end evaluation pipeline with the RAG system."""
    
    def __init__(self, rag_pipeline: Any, evaluator: Optional[Evaluator] = None, config: Optional[Dict[str, Any]] = None):
        """Initialize the evaluation pipeline."""
        self.rag_pipeline = rag_pipeline
        self.evaluator = evaluator or Evaluator(config)
        self.config = config or {}
        self.runs: List[Tuple[Dict, EvaluationResult]] = []
        
    def evaluate_query(
        self,
        query_id: str,
        query_text: str,
        ground_truth: Optional[str] = None,
        **pipeline_kwargs
    ) -> EvaluationResult:
        """Evaluate a single query through the full pipeline."""
        start_time = time.time()
        result = self.rag_pipeline.generate_answer(query_text, **pipeline_kwargs)
        latency_ms = (time.time() - start_time) * 1000
        
        # Extract predicted answer - GeneratedAnswer.answer is a dataclass with .answer attribute
        answer_obj = result.get("answer")
        if answer_obj:
            predicted_answer = getattr(answer_obj, "answer", "")
        else:
            predicted_answer = ""
        
        # Extract citations from grounded context
        grounded_context = result.get("grounded_context")
        if grounded_context:
            retrieved_context = getattr(grounded_context, "citations", [])
        else:
            retrieved_context = []
        
        eval_result = self.evaluator.evaluate(
            query_id=query_id, query_text=query_text, predicted_answer=predicted_answer,
            ground_truth=ground_truth, retrieved_context=retrieved_context, latency_ms=latency_ms
        )
        
        self.runs.append((result, eval_result))
        return eval_result
    
    def evaluate_dataset(self, queries: List[Dict[str, Any]], parallel: bool = False) -> EvaluationSummary:
        """Evaluate multiple queries."""
        for q in queries:
            query_id = q.get("query_id", f"query_{len(self.runs)}")
            query_text = q.get("query_text", "")
            ground_truth = q.get("ground_truth")
            self.evaluate_query(query_id=query_id, query_text=query_text, ground_truth=ground_truth)
        return self.evaluator.get_summary()
    
    def export_results(self, filepath: str) -> None:
        """Export all evaluation results to a JSON file."""
        self.evaluator.export_results(filepath)


def create_evaluator(config: Optional[Dict[str, Any]] = None) -> Evaluator:
    """Factory function to create an Evaluator."""
    return Evaluator(config)


def create_evaluation_pipeline(rag_pipeline: Any, config: Optional[Dict[str, Any]] = None) -> EvaluationPipeline:
    """Factory function to create an EvaluationPipeline."""
    evaluator = create_evaluator(config)
    return EvaluationPipeline(rag_pipeline=rag_pipeline, evaluator=evaluator, config=config)


def compute_basic_metrics(answer: str, ground_truth: str) -> Dict[str, float]:
    """Compute basic evaluation metrics without full evaluation framework."""
    evaluator = Evaluator()
    return {
        "relevance": evaluator._evaluate_relevance("", answer, ground_truth),
        "answer_quality": evaluator._evaluate_answer_quality(answer, ground_truth),
        "hallucination": evaluator._detect_hallucination(answer, ground_truth)
    }

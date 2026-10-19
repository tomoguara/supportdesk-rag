"""Stage 8: Observability - Comprehensive monitoring for RAG pipelines.

This module provides comprehensive observability for the SupportDesk RAG system,
including tracing, metrics collection, and performance monitoring.

Key features:
- Distributed tracing using OpenTelemetry
- Per-stage metrics (latency, throughput, accuracy)
- Custom logging with structured JSON format
- Metrics collection and summary reporting

For integration with the unified pipeline, see `supportdesk_rag_pipeline`.
"""

import os
import json
import time
import logging
import uuid
from contextlib import contextmanager
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import yaml
from threading import Lock
from datetime import datetime


# Try to import OpenTelemetry
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
    from opentelemetry.exporter.jaeger.proto.grpc import JaegerExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.semconv.resource import ResourceAttributes
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False


try:
    import prometheus_client
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
logger = logging.getLogger(__name__)


class Config:
    """Configuration manager for Observability module."""

    _instance = None
    _config = None

    def __new__(cls, config_path: str = "config/config.yaml"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._config = cls._load_config(config_path)
        return cls._instance

    @staticmethod
    def _load_config(config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {
                "observability": {
                    "enabled": True,
                    "tracing": {
                        "enabled": True,
                        "Jaeger": {
                            "host": "localhost",
                            "port": 14317,
                            "service_name": "supportdesk-rag"
                        }
                    },
                    "metrics": {
                        "enabled": True,
                        "prometheus_port": 9090,
                        "metric_prefix": "supportdesk_rag"
                    },
                    "logging": {
                        "level": "INFO",
                        "json_format": True
                    }
                }
            }

    def get(self, *keys, default=None):
        """Get config value using dot notation."""
        value = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def get_observability_config(self) -> Dict:
        """Get observability-specific configuration."""
        return self._config.get("observability", {})

    def is_tracing_enabled(self) -> bool:
        """Check if tracing is enabled."""
        return self.get("observability", "tracing", "enabled", default=True)

    def is_metrics_enabled(self) -> bool:
        """Check if metrics collection is enabled."""
        return self.get("observability", "metrics", "enabled", default=True)

    def is_logging_enabled(self) -> bool:
        """Check if logging is enabled."""
        return self.get("observability", "logging", "enabled", default=True)


@contextmanager
def trace_stage(
    stage_name: str,
    extra: Optional[Dict] = None,
    tracer: Optional[Any] = None
):
    """Context manager for tracing stages.

    Args:
        stage_name: Name of the stage being traced
        extra: Optional dictionary of extra attributes to include
        tracer: Optional OpenTelemetry tracer instance

    Yields:
        str: Trace ID for the operation

    Example:
        with trace_stage("retrieve") as trace_id:
            results = retriever.retrieve(query, embedding)
    """
    trace_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    # Log structured message
    logger.info(
        json.dumps({
            "stage": stage_name,
            "trace_id": trace_id,
            "event": "start",
            "timestamp": datetime.now().isoformat()
        })
    )

    # OpenTelemetry tracing if available
    span = None
    if OTEL_AVAILABLE and tracer:
        try:
            span = tracer.start_span(f"rag.{stage_name}")
            span.set_attribute("stage.name", stage_name)
            span.set_attribute("trace.id", trace_id)
        except Exception:
            span = None

    try:
        yield trace_id
    except Exception as e:
        # Log error
        logger.error(
            json.dumps({
                "stage": stage_name,
                "trace_id": trace_id,
                "event": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
        )

        # Record exception in span
        if span:
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(e))
            span.end()

        raise
    finally:
        elapsed = time.time() - start_time

        # Log completion
        logger.info(
            json.dumps({
                "stage": stage_name,
                "trace_id": trace_id,
                "event": "end",
                "duration_ms": elapsed * 1000,
                "timestamp": datetime.now().isoformat()
            })
        )

        # Record duration in span
        if span:
            span.set_attribute("stage.duration_ms", elapsed * 1000)
            span.set_attribute("extra", str(extra) if extra else "")
            span.end()


@dataclass
class StageMetrics:
    """Metrics for a single stage."""
    stage_name: str
    durations_ms: List[float] = field(default_factory=list)
    metadata_list: List[Dict] = field(default_factory=list)
    error_count: int = 0
    success_count: int = 0

    def record(self, duration_ms: float, metadata: Optional[Dict] = None, error: bool = False):
        """Record a stage execution."""
        self.durations_ms.append(duration_ms)
        self.metadata_list.append(metadata or {})
        if error:
            self.error_count += 1
        else:
            self.success_count += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for this stage."""
        if not self.durations_ms:
            return {
                "count": 0,
                "mean_ms": 0,
                "min_ms": 0,
                "max_ms": 0,
                "p95_ms": 0,
                "error_count": self.error_count,
                "success_count": self.success_count
            }

        sorted_durations = sorted(self.durations_ms)
        p95_idx = int(len(sorted_durations) * 0.95)

        return {
            "count": len(self.durations_ms),
            "mean_ms": sum(self.durations_ms) / len(self.durations_ms),
            "min_ms": min(self.durations_ms),
            "max_ms": max(self.durations_ms),
            "p95_ms": sorted_durations[p95_idx] if sorted_durations else 0,
            "error_count": self.error_count,
            "success_count": self.success_count
        }


class RAGPipelineObserver:
    """Observer for RAG pipeline metrics.

    Collects and aggregates metrics across all stages of the RAG pipeline,
    including latency, throughput, and error rates.

    Example:
        observer = RAGPipelineObserver()

        with trace_stage("retrieve") as trace_id:
            results = retriever.retrieve(query, embedding)
            observer.record("retrieve", 45.2, {"top_k": 10})

        with trace_stage("generate") as trace_id:
            answer = generator.generate_answer(...)
            observer.record("generate", 1200.5, {"model": "gpt-4o-mini"})

        print(observer.get_summary())
    """

    def __init__(self):
        """Initialize the observer with empty metrics."""
        self.metrics: Dict[str, StageMetrics] = {}
        self.total_start_time: Optional[float] = None
        self.query_count: int = 0
        self._lock = Lock()

    def start_pipeline(self):
        """Mark the start of a pipeline execution."""
        with self._lock:
            self.total_start_time = time.time()
            self.query_count += 1

    def record(
        self,
        stage: str,
        duration_ms: float,
        metadata: Optional[Dict] = None,
        error: bool = False
    ):
        """Record stage metrics.

        Args:
            stage: Name of the stage (e.g., 'embed', 'retrieve', 'generate')
            duration_ms: Duration in milliseconds
            metadata: Optional dictionary of metadata to store
            error: Whether this execution resulted in an error
        """
        with self._lock:
            if stage not in self.metrics:
                self.metrics[stage] = StageMetrics(stage_name=stage)

            self.metrics[stage].record(duration_ms, metadata, error)

    def record_pipeline(self, stage: str, duration_ms: float, metadata: Optional[Dict] = None):
        """Record a successful pipeline stage execution."""
        self.record(stage, duration_ms, metadata, error=False)

    def record_pipeline_error(self, stage: str, duration_ms: float, metadata: Optional[Dict] = None):
        """Record a failed pipeline stage execution."""
        self.record(stage, duration_ms, metadata, error=True)

    def get_summary(self) -> Dict[str, Dict]:
        """Get latency summary per stage.

        Returns:
            Dictionary mapping stage names to their statistics.
        """
        with self._lock:
            summary = {}
            for stage, stage_metrics in self.metrics.items():
                summary[stage] = stage_metrics.get_summary()
            return summary

    def get_total_time(self) -> Optional[float]:
        """Get total pipeline execution time if still running."""
        if self.total_start_time:
            return time.time() - self.total_start_time
        return None

    def get_throughput(self) -> float:
        """Get queries per second (QPS)."""
        total_time = self.get_total_time()
        if total_time and total_time > 0:
            return self.query_count / total_time
        return 0.0

    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self.metrics = {}
            self.total_start_time = None
            self.query_count = 0


class MetricsCollector:
    """Collects and reports various RAG metrics.

    Tracks:
    - Latency (end-to-end and per-stage)
    - Throughput (queries per second)
    - Accuracy (via eval sets)
    - Hallucination rate
    - Cost (API costs)
    - Drift (embedding distribution shifts)
    - User satisfaction
    """

    def __init__(self, config: Optional[Config] = None):
        """Initialize the metrics collector.

        Args:
            config: Optional Config instance (uses default if not provided)
        """
        self.config = config or Config()
        self.observer = RAGPipelineObserver()

        # Eval sets for accuracy tracking
        self.eval_sets: Dict[str, List[Dict]] = {}

        # Cost tracking
        self.costs: Dict[str, float] = {}
        self.cost_breakdown: List[Dict] = []

        # Drift tracking (embedding distribution)
        self.embedding_stats: Dict[str, Dict] = {}

        # User satisfaction tracking
        self.satisfaction_scores: List[float] = []

    def record_latency(self, stage: str, duration_ms: float):
        """Record latency for a stage.

        Args:
            stage: Stage name
            duration_ms: Duration in milliseconds
        """
        self.observer.record_pipeline(stage, duration_ms)

    def record_throughput(self, count: int = 1):
        """Record query count for throughput calculation.

        Args:
            count: Number of queries (default: 1)
        """
        for _ in range(count):
            self.observer.start_pipeline()

    def record_accuracy(self, eval_set_name: str, correct: bool):
        """Record accuracy metric from eval set.

        Args:
            eval_set_name: Name of the eval set
            correct: Whether the answer was correct
        """
        if eval_set_name not in self.eval_sets:
            self.eval_sets[eval_set_name] = []

        self.eval_sets[eval_set_name].append({
            "correct": correct,
            "timestamp": time.time()
        })

    def get_accuracy(self, eval_set_name: str) -> float:
        """Get accuracy for an eval set.

        Args:
            eval_set_name: Name of the eval set

        Returns:
            Accuracy as a percentage (0-100)
        """
        if eval_set_name not in self.eval_sets or not self.eval_sets[eval_set_name]:
            return 0.0

        correct_count = sum(
            1 for r in self.eval_sets[eval_set_name] if r["correct"]
        )
        return (correct_count / len(self.eval_sets[eval_set_name])) * 100

    def record_cost(self, stage: str, cost: float, metadata: Optional[Dict] = None):
        """Record API cost for a stage.

        Args:
            stage: Stage name
            cost: Cost in dollars
            metadata: Optional metadata (e.g., model name)
        """
        if stage not in self.costs:
            self.costs[stage] = 0.0
        self.costs[stage] += cost

        self.cost_breakdown.append({
            "stage": stage,
            "cost": cost,
            "metadata": metadata or {},
            "timestamp": time.time()
        })

    def get_total_cost(self) -> float:
        """Get total cost across all stages."""
        return sum(self.costs.values())

    def get_cost_by_stage(self) -> Dict[str, float]:
        """Get cost breakdown by stage."""
        return dict(self.costs)

    def record_embedding_stats(self, stage: str, embeddings: List[List[float]]):
        """Record embedding statistics for drift detection.

        Args:
            stage: Stage name
            embeddings: List of embeddings to analyze
        """
        import numpy as np

        if stage not in self.embedding_stats:
            self.embedding_stats[stage] = {
                "count": 0,
                "mean_norm": 0.0,
                "embedding_dimensions": 0
            }

        if embeddings:
            norms = [np.linalg.norm(e) for e in embeddings]
            current_count = self.embedding_stats[stage]["count"]

            # Update running mean
            new_mean_norm = (
                (current_count * self.embedding_stats[stage]["mean_norm"] +
                 sum(norms)) / (current_count + len(embeddings))
            )

            self.embedding_stats[stage] = {
                "count": current_count + len(embeddings),
                "mean_norm": new_mean_norm,
                "embedding_dimensions": len(embeddings[0]) if embeddings else 0
            }

    def get_embedding_drift(self, stage: str) -> Optional[Dict]:
        """Get embedding drift statistics.

        Args:
            stage: Stage name

        Returns:
            Dictionary with drift statistics or None if no data
        """
        return self.embedding_stats.get(stage)

    def record_satisfaction(self, score: float):
        """Record user satisfaction score.

        Args:
            score: Satisfaction score (0-10 scale)
        """
        self.satisfaction_scores.append(score)

    def get_average_satisfaction(self) -> float:
        """Get average user satisfaction score."""
        if not self.satisfaction_scores:
            return 0.0
        return sum(self.satisfaction_scores) / len(self.satisfaction_scores)

    def record_hallucination(self, detected: bool, metadata: Optional[Dict] = None):
        """Record hallucination detection result.

        Args:
            detected: Whether hallucination was detected
            metadata: Optional metadata
        """
        if not hasattr(self, '_hallucinations'):
            self._hallucinations = []

        self._hallucinations.append({
            "detected": detected,
            "metadata": metadata or {},
            "timestamp": time.time()
        })

    def get_hallucination_rate(self) -> float:
        """Get hallucination detection rate.

        Returns:
            Percentage of hallucinations detected (0-100)
        """
        if not hasattr(self, '_hallucinations') or not self._hallucinations:
            return 0.0

        hallucination_count = sum(
            1 for h in self._hallucinations if h["detected"]
        )
        return (hallucination_count / len(self._hallucinations)) * 100

    def get_full_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics.

        Returns:
            Dictionary containing all metrics
        """
        return {
            "latency": self.observer.get_summary(),
            "throughput_qps": self.observer.get_throughput(),
            "accuracy": {
                eval_set: self.get_accuracy(eval_set)
                for eval_set in self.eval_sets
            },
            "cost": {
                "by_stage": self.get_cost_by_stage(),
                "total": self.get_total_cost()
            },
            "hallucination_rate": self.get_hallucination_rate(),
            "embedding_drift": {
                stage: self.get_embedding_drift(stage)
                for stage in self.embedding_stats
            },
            "user_satisfaction": {
                "average": self.get_average_satisfaction(),
                "count": len(self.satisfaction_scores)
            }
        }

    def reset(self):
        """Reset all metrics."""
        self.observer.reset()
        self.eval_sets = {}
        self.costs = {}
        self.cost_breakdown = []
        self.embedding_stats = {}
        self.satisfaction_scores = []
        if hasattr(self, '_hallucinations'):
            delattr(self, '_hallucinations')


class ObservabilityPipeline:
    """Pipeline wrapper for observability.

    Wraps the RAG pipeline to automatically collect metrics and trace
    all stage executions.

    Example:
        # Create base pipeline
        base_pipeline = SupportDeskRAGPipeline()

        # Wrap with observability
        observability = ObservabilityPipeline(base_pipeline)

        # Run with automatic tracing
        results = observability.generate_answer("query", top_k=5)
        print(observability.observer.get_summary())
    """

    def __init__(
        self,
        pipeline: Any,
        config: Optional[Config] = None,
        collector: Optional[MetricsCollector] = None,
        enable_tracing: Optional[bool] = None,
        enable_metrics: Optional[bool] = None
    ):
        """Initialize the observability pipeline wrapper.

        Args:
            pipeline: The RAG pipeline to wrap
            config: Optional configuration instance
            collector: Optional metrics collector
            enable_tracing: Override for tracing (default: from config)
            enable_metrics: Override for metrics (default: from config)
        """
        self.pipeline = pipeline
        self.config = config or Config()
        self.collector = collector or MetricsCollector(self.config)

        # Get enabled status from config or override
        self.enable_tracing = (
            enable_tracing
            if enable_tracing is not None
            else self.config.is_tracing_enabled()
        )
        self.enable_metrics = (
            enable_metrics
            if enable_metrics is not None
            else self.config.is_metrics_enabled()
        )

        # Initialize OpenTelemetry tracer if available
        self.tracer = None
        if OTEL_AVAILABLE and self.enable_tracing:
            self._init_tracer()

        # Store trace IDs for cross-stage correlation
        self.current_trace_id: Optional[str] = None

    def _init_tracer(self):
        """Initialize OpenTelemetry tracer."""
        service_name = self.config.get("observability", "tracing", "Jaeger", "service_name", default="supportdesk-rag")

        try:
            # Setup resource
            resource = Resource.create({
                ResourceAttributes.SERVICE_NAME: service_name
            })

            # Create tracer provider
            provider = TracerProvider(resource=resource)

            # Add console exporter for development
            console_exporter = SimpleSpanProcessor(ConsoleSpanExporter())
            provider.add_span_processor(console_exporter)

            # Add Jaeger exporter for production
            jaeger_host = self.config.get("observability", "tracing", "Jaeger", "host", default="localhost")
            jaeger_port = self.config.get("observability", "tracing", "Jaeger", "port", default=14317)

            jaeger_exporter = JaegerExporter(
                agent_host_name=jaeger_host,
                agent_port=jaeger_port
            )
            jaeger_processor = SimpleSpanProcessor(jaeger_exporter)
            provider.add_span_processor(jaeger_processor)

            # Set global tracer
            trace.set_tracer_provider(provider)
            self.tracer = trace.get_tracer(__name__)

        except Exception as e:
            logger.warning(
                json.dumps({
                    "event": "otel_init_failed",
                    "error": str(e),
                    "message": "OpenTelemetry initialization failed, falling back to logging"
                })
            )
            self.tracer = None

    def _trace_stage(self, stage_name: str, extra: Optional[Dict] = None):
        """Context manager for tracing a stage."""
        return trace_stage(stage_name, extra, self.tracer)

    def generate_answer(
        self,
        query_text: str,
        top_k: int = 5,
        vector_top_k: Optional[int] = None,
        bm25_top_k: Optional[int] = None,
        rerank_enabled: Optional[bool] = None,
        ground_enabled: Optional[bool] = None,
        max_passages: Optional[int] = None,
        max_tokens: Optional[int] = None,
        min_relevance_score: Optional[float] = None
    ) -> Dict:
        """Run the pipeline with observability.

        Wraps the generate_answer method with automatic tracing and metrics.

        Args:
            query_text: User's incident question
            top_k: Number of final results to return
            vector_top_k: Candidates from vector search
            bm25_top_k: Candidates from BM25 search
            rerank_enabled: Override for re-ranking
            ground_enabled: Override for grounding
            max_passages: Override for max passages
            max_tokens: Override for max tokens
            min_relevance_score: Override for min relevance

        Returns:
            Results dictionary from pipeline
        """
        if not self.enable_metrics:
            return self.pipeline.generate_answer(
                query_text=query_text,
                top_k=top_k,
                vector_top_k=vector_top_k,
                bm25_top_k=bm25_top_k,
                rerank_enabled=rerank_enabled,
                ground_enabled=ground_enabled,
                max_passages=max_passages,
                max_tokens=max_tokens,
                min_relevance_score=min_relevance_score
            )

        # Record pipeline start
        self.collector.observer.start_pipeline()

        results = {}
        start_time = time.time()

        try:
            with self._trace_stage("generate_answer") as trace_id:
                self.current_trace_id = trace_id

                # Execute the actual pipeline
                results = self.pipeline.generate_answer(
                    query_text=query_text,
                    top_k=top_k,
                    vector_top_k=vector_top_k,
                    bm25_top_k=bm25_top_k,
                    rerank_enabled=rerank_enabled,
                    ground_enabled=ground_enabled,
                    max_passages=max_passages,
                    max_tokens=max_tokens,
                    min_relevance_score=min_relevance_score
                )

                # Record success
                total_duration = (time.time() - start_time) * 1000
                self.collector.record_pipeline("generate_answer", total_duration)

        except Exception as e:
            # Record error
            total_duration = (time.time() - start_time) * 1000
            self.collector.record_pipeline_error("generate_answer", total_duration)

            # Re-raise exception
            raise

        return results

    def hybrid_search(
        self,
        query_text: str,
        top_k: int = 5,
        vector_top_k: Optional[int] = None,
        bm25_top_k: Optional[int] = None
    ) -> Dict:
        """Run hybrid search with observability.

        Args:
            query_text: Search query
            top_k: Number of results to return
            vector_top_k: Vector search candidates
            bm25_top_k: BM25 search candidates

        Returns:
            Search results dictionary
        """
        if not self.enable_metrics:
            return self.pipeline.hybrid_search(
                query_text=query_text,
                top_k=top_k,
                vector_top_k=vector_top_k,
                bm25_top_k=bm25_top_k
            )

        self.collector.observer.start_pipeline()

        try:
            with self._trace_stage("hybrid_search") as trace_id:
                self.current_trace_id = trace_id
                return self.pipeline.hybrid_search(
                    query_text=query_text,
                    top_k=top_k,
                    vector_top_k=vector_top_k,
                    bm25_top_k=bm25_top_k
                )
        except Exception:
            raise

    def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics.

        Returns:
            Dictionary containing all metrics
        """
        return self.collector.get_full_metrics()

    def get_observer(self) -> RAGPipelineObserver:
        """Get the underlying observer instance."""
        return self.collector.observer

    def reset(self):
        """Reset all metrics."""
        self.collector.reset()
        self.current_trace_id = None


# Backward compatibility aliases
SupportDeskObservability = ObservabilityPipeline
SupportDeskMetricsCollector = MetricsCollector
SupportDeskPipelineObserver = RAGPipelineObserver


def create_observability_pipeline(
    pipeline: Any,
    enable_tracing: Optional[bool] = None,
    enable_metrics: Optional[bool] = None
) -> ObservabilityPipeline:
    """Factory function to create an ObservabilityPipeline.

    Args:
        pipeline: The RAG pipeline to wrap
        enable_tracing: Override for tracing (default: from config)
        enable_metrics: Override for metrics (default: from config)

    Returns:
        ObservabilityPipeline instance
    """
    return ObservabilityPipeline(
        pipeline=pipeline,
        enable_tracing=enable_tracing,
        enable_metrics=enable_metrics
    )

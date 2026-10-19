from supportdesk_rag.supportdesk_ingest import SupportDeskIngest
from supportdesk_rag.supportdesk_embed import (
    Config,
    EmbeddingService,
    EmbeddingPipeline,
    create_embedding_pipeline,
    SupportDeskRAGPipeline as SupportDeskRAGPipelineEmbed,
)
from supportdesk_rag.supportdesk_rag_pipeline import SupportDeskRAGPipeline
from supportdesk_rag.supportdesk_store import (
    VectorStore,
    StorePipeline,
    create_vector_store,
    create_store_pipeline,
    get_embedding_model_dimensions,
)
from supportdesk_rag.supportdesk_retrieve import (
    Config as RetrieveConfig,
    BM25Index,
    HybridRetriever,
    create_hybrid_retriever,
    create_bm25_index,
    SupportDeskRetriever,
)
from supportdesk_rag.supportdesk_rerank import (
    Config as ReRankConfig,
    ReRanker,
    ReRankerPipeline,
    create_reranker,
    SupportDeskReRanker,
    SupportDeskReRankerPipeline,
)
from supportdesk_rag.supportdesk_ground import (
    GroundedContext,
    Config as GroundConfig,
    ContextGrounding,
    create_context_grounder,
    SupportDeskGrounding,
    SupportDeskGroundedContext,
)
from supportdesk_rag.model_selector import (
    ModelSelector,
    create_model_selector,
    get_model_selector,
    use_ollama,
    get_embedding_model_name,
    get_llm_model_name,
    get_base_url,
    print_model_config
)

# Backward compatibility import
from supportdesk_rag.supportdesk_embed import SupportDeskRAGPipelineWithStore

# Stage 7 imports
from supportdesk_rag.supportdesk_generate import (
    GeneratedAnswer,
    Config as GenerateConfig,
    AnswerGenerator,
    create_answer_generator,
    SupportDeskGenerator,
    SupportDeskGeneratedAnswer,
)

# Guard imports
from supportdesk_rag.supportdesk_guard import (
    Guardrails,
    Config as GuardConfig,
    create_guardrails,
    SupportDeskGuardrails,
    SupportDeskGuard,
)

# Observability imports
from supportdesk_rag.supportdesk_observability import (
    Config as ObservabilityConfig,
    trace_stage,
    RAGPipelineObserver,
    MetricsCollector,
    ObservabilityPipeline,
    create_observability_pipeline,
    SupportDeskObservability,
    SupportDeskMetricsCollector,
    SupportDeskPipelineObserver,
)

# Evaluation imports
from supportdesk_rag.supportdesk_eval import (
    EvaluationResult,
    EvaluationSummary,
    Evaluator,
    EvaluationPipeline,
    EvaluationMetric,
    create_evaluator,
    create_evaluation_pipeline,
    compute_basic_metrics,
)

__all__ = [
    "SupportDeskIngest",
    "Config",
    "SupportDeskRAGPipeline",
    "SupportDeskRAGPipelineWithStore",
    "VectorStore",
    "StorePipeline",
    "create_vector_store",
    "create_store_pipeline",
    "get_embedding_model_dimensions",
    "BM25Index",
    "HybridRetriever",
    "create_hybrid_retriever",
    "create_bm25_index",
    "SupportDeskRetriever",
    "RetrieveConfig",
    "ReRanker",
    "ReRankerPipeline",
    "create_reranker",
    "SupportDeskReRanker",
    "SupportDeskReRankerPipeline",
    "ReRankConfig",
    "GroundedContext",
    "GroundConfig",
    "ContextGrounding",
    "create_context_grounder",
    "SupportDeskGrounding",
    "SupportDeskGroundedContext",
    "ModelSelector",
    "create_model_selector",
    "get_model_selector",
    "use_ollama",
    "get_embedding_model_name",
    "get_llm_model_name",
    "get_base_url",
    "print_model_config",
    # Stage 7 exports
    "GeneratedAnswer",
    "GenerateConfig",
    "AnswerGenerator",
    "create_answer_generator",
    "SupportDeskGenerator",
    "SupportDeskGeneratedAnswer",
    # Guard exports
    "Guardrails",
    "GuardConfig",
    "create_guardrails",
    "SupportDeskGuardrails",
    "SupportDeskGuard",
    # Observability exports
    "ObservabilityConfig",
    "trace_stage",
    "RAGPipelineObserver",
    "MetricsCollector",
    "ObservabilityPipeline",
    "create_observability_pipeline",
    "SupportDeskObservability",
    "SupportDeskMetricsCollector",
    "SupportDeskPipelineObserver",
    # Evaluation exports
    "EvaluationResult",
    "EvaluationSummary",
    "Evaluator",
    "EvaluationPipeline",
    "EvaluationMetric",
    "create_evaluator",
    "create_evaluation_pipeline",
    "compute_basic_metrics",
]


def main() -> None:
    print("Hello from supportdesk-rag!")

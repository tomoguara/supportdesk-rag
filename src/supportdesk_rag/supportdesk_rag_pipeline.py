"""Unified RAG Pipeline: Stage 1 (Ingest) + Stage 2 (Embed) + Stage 3 (Store) + Stage 4 (Retrieve) + Stage 5 (Re-rank)

This module provides the main entry point for the SupportDesk RAG system,
combining all stages into a unified pipeline.
"""
import os
import json
import uuid
from typing import List, Dict, Optional, Any
from pathlib import Path

# Stage 1 imports
from supportdesk_rag.supportdesk_ingest import SupportDeskIngest

# Stage 2 imports
from supportdesk_rag.supportdesk_embed import (
    Config,
    EmbeddingService,
    EmbeddingPipeline,
    create_embedding_pipeline
)

# Stage 3 imports
from supportdesk_rag.supportdesk_store import VectorStore, StorePipeline

# Stage 4 imports
from supportdesk_rag.supportdesk_retrieve import HybridRetriever, create_hybrid_retriever

# Stage 5 imports
from supportdesk_rag.supportdesk_rerank import ReRanker, ReRankerPipeline

# Stage 6 imports
from supportdesk_rag.supportdesk_ground import (
    GroundedContext,
    Config as GroundConfig,
    ContextGrounding,
    create_context_grounder
)

# Stage 7 imports
from supportdesk_rag.supportdesk_generate import (
    GeneratedAnswer,
    Config as GenerateConfig,
    AnswerGenerator,
    create_answer_generator
)

# Guard imports
from supportdesk_rag.supportdesk_guard import (
    Guardrails,
    Config as GuardConfig,
    create_guardrails,
    SupportDeskGuardrails,
    SupportDeskGuard
)

# Observability imports
from supportdesk_rag.supportdesk_observability import (
    ObservabilityPipeline,
    RAGPipelineObserver,
    MetricsCollector,
    Config as ObservabilityConfig,
)


class SupportDeskRAGPipeline:
    """Unified pipeline for Stage 1 (Ingest) + Stage 2 (Embed) + Stage 3 (Store) + Stage 4 (Retrieve) + Stage 5 (Re-rank) + Stage 6 (Ground) + Stage 7 (Generate) + Stage 8 (Observability).

    This is the main entry point for the SupportDesk RAG system. It integrates
    all eight stages:

    - Stage 1: Ingest - Load and chunk ticket data
    - Stage 2: Embed - Generate embeddings using OpenAI
    - Stage 3: Store - Persist embeddings in Qdrant vector database
    - Stage 4: Retrieve - Hybrid search (vector + BM25 with RRF fusion)
    - Stage 5: Re-rank - Cross-encoder re-ranking for precise relevance
    - Stage 6: Ground - Combine re-ranked documents into grounded context with citations
    - Stage 7: Generate - Generate answers from grounded context using LLM
    - Stage 8: Observability - OpenTelemetry tracing and metrics collection

    Usage:
        ```python
        from supportdesk_rag import SupportDeskRAGPipeline

        # Create pipeline (auto-initializes all stages including observability)
        pipeline = SupportDeskRAGPipeline()

        # Run complete pipeline on ticket data
        result = pipeline.run("data/tickets.json")

        # Search stored embeddings using hybrid retrieval
        results = pipeline.hybrid_search("network timeout", top_k=5)

        # Search with re-ranking, grounding, and generation enabled
        generated_results = pipeline.generate_answer("network timeout", top_k=5)
        
        # Access generated answer
        print(f"Answer: {generated_results['answer'].answer}")
        print(f"Citations: {generated_results['answer'].citations}")
        print(f"Tokens used: {generated_results['answer'].total_tokens}")

        # Access observability metrics
        metrics = pipeline.get_observability_metrics()
        print(f"Pipeline duration: {metrics.get('pipeline_duration_ms', 0)}ms")
        ```

    The pipeline automatically handles:
    - Ticket ingestion and chunking
    - PII cleaning
    - Embedding generation
    - Vector storage in Qdrant
    - Hybrid search with BM25 + RRF fusion
    - Cross-encoder re-ranking for precise relevance (optional)
    - Context grounding with citation tracking (optional)
    - LLM-based answer generation with citations (optional)
    - OpenTelemetry tracing and metrics collection (optional)

    Observability:
        - Tracing: Track pipeline execution across all stages
        - Metrics: Collect latency, throughput, accuracy, and more
        - JSON logging: Structured observability logs
        ```python
        # Create pipeline with observability enabled (default)
        pipeline = SupportDeskRAGPipeline(observability_enabled=True)

        # Access metrics after pipeline execution
        metrics = pipeline.get_observability_metrics()

        # Reset metrics for a fresh collection
        pipeline.reset_observability()
        ```
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        batch_size: int = 100,
        rerank_enabled: Optional[bool] = None,  # None = use config
        ground_enabled: bool = True,
        generate_enabled: bool = True,
        max_passages: Optional[int] = None,
        max_tokens: Optional[int] = None,
        min_relevance_score: Optional[float] = None,
        enable_citation_check: bool = True,
        enable_relevance_gate: bool = True,
        enable_pii_redact: bool = True,
        enable_safety_filter: bool = False,
        min_citations: int = 1,
        relevance_threshold: float = 0.5,
        guard_min_tokens: int = 10,
        observability_enabled: bool = True,
        enable_tracing: Optional[bool] = None,
        enable_metrics: Optional[bool] = None
    ):
        """Initialize the unified RAG pipeline.

        Args:
            chunk_size: Size of text chunks for Stage 1 (default: 1000)
            chunk_overlap: Overlap between chunks for Stage 1 (default: 150)
            batch_size: Batch size for embedding generation (default: 100)
            rerank_enabled: Whether re-ranking is enabled (Stage 5, default: True)
            ground_enabled: Whether context grounding is enabled (Stage 6, default: True)
            generate_enabled: Whether answer generation is enabled (Stage 7, default: True)
            max_passages: Maximum number of passages to include in grounded context (default: 5)
            max_tokens: Maximum token count for grounded context (default: 4000)
            min_relevance_score: Minimum rerank score to include (default: -3.0)
            enable_citation_check: Whether to verify citations in answers (default: True)
            enable_relevance_gate: Whether to check answer relevance (default: True)
            enable_pii_redact: Whether to redact PII from answers (default: True)
            enable_safety_filter: Whether to filter harmful content (default: False)
            min_citations: Minimum citations required in answers (default: 1)
            relevance_threshold: Minimum similarity for relevance check (default: 0.5)
            guard_min_tokens: Minimum answer length in tokens (default: 10)
            observability_enabled: Whether to enable observability (default: True)
            enable_tracing: Override tracing setting (default: from config)
            enable_metrics: Override metrics setting (default: from config)
        """
        self.config = Config()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        
        # Use config values if not explicitly provided
        rerank_config = self.config.get_rerank_config()
        self.rerank_enabled = (
            rerank_enabled 
            if rerank_enabled is not None 
            else rerank_config.get("enabled", True)
        )
        self.generate_enabled = generate_enabled

        # Guard configuration
        self.enable_citation_check = enable_citation_check
        self.enable_relevance_gate = enable_relevance_gate
        self.enable_pii_redact = enable_pii_redact
        self.enable_safety_filter = enable_safety_filter
        self.min_citations = min_citations
        self.relevance_threshold = relevance_threshold
        self.max_tokens = max_tokens if max_tokens is not None else 4000
        self.min_tokens = guard_min_tokens

        # Observability configuration
        self.observability_enabled = observability_enabled
        self.enable_tracing = enable_tracing
        self.enable_metrics = enable_metrics

        # Initialize Stage 1: Ingest
        self.ingest = SupportDeskIngest(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        # Initialize Stage 2: Embed
        self.embed_service = EmbeddingService(batch_size=self.batch_size)
        self.embed_pipeline = EmbeddingPipeline(
            embedding_service=self.embed_service,
            batch_size=self.batch_size
        )

        # Initialize Stage 3: Store
        try:
            self.vector_store = VectorStore()
            self.store_enabled = True
        except Exception as e:
            print(f"Warning: Could not initialize vector store: {e}")
            print("Continuing without Stage 3 storage (embeddings will not be persisted)")
            self.vector_store = None
            self.store_enabled = False

        # Initialize Stage 4: Retrieve
        self.hybrid_retriever = create_hybrid_retriever(
            vector_store=self.vector_store,
            embedding_service=self.embed_service
        ) if self.store_enabled else None

        # Initialize Stage 5: Re-rank
        self.rerank_pipeline = ReRankerPipeline(
            enabled=self.rerank_enabled
        ) if self.store_enabled else None

        # Initialize Stage 6: Ground
        self.ground_enabled = ground_enabled
        self.max_passages = max_passages
        # self.max_tokens already set on line 193
        self.min_relevance_score = min_relevance_score

        # Initialize Stage 6: Ground
        if self.store_enabled and self.ground_enabled:
            self.ground_pipeline = ContextGrounding(
                max_passages=self.max_passages,
                max_tokens=self.max_tokens,
                min_relevance_score=self.min_relevance_score
            )
        else:
            self.ground_pipeline = None

        # Initialize Stage 7: Generate
        if self.store_enabled and self.ground_enabled and self.generate_enabled:
            self.generate_pipeline = AnswerGenerator()
        else:
            self.generate_pipeline = None

        # Initialize Guard: Validate answers
        if self.store_enabled and self.generate_enabled:
            self.guard_pipeline = Guardrails(
                enable_citation_check=self.enable_citation_check,
                enable_relevance_gate=self.enable_relevance_gate,
                enable_pii_redact=self.enable_pii_redact,
                enable_safety_filter=self.enable_safety_filter,
                min_citations_per_answer=self.min_citations,
                relevance_threshold=self.relevance_threshold,
                max_tokens=self.max_tokens,
                min_tokens=self.min_tokens,
                embedding_service=self.embed_service
            )
        else:
            self.guard_pipeline = None

        # Initialize Stage 8: Observability
        if self.observability_enabled:
            self.observability = ObservabilityPipeline(
                pipeline=self,
                enable_tracing=self.enable_tracing,
                enable_metrics=self.enable_metrics
            )
        else:
            self.observability = None

    def run(self, data_path: str, save: bool = False,
            output_path: Optional[str] = None, clean: bool = True,
            store: bool = True) -> Dict:
        """Run the complete Stage 1 + Stage 2 + Stage 3 pipeline.

        Args:
            data_path: Path to ticket data file
            save: Whether to save embeddings to file (Stage 2)
            output_path: Output path for saved embeddings
            clean: Whether to clean PII from text (Stage 1)
            store: Whether to store embeddings in Qdrant (Stage 3)

        Returns:
            Dict with results from all stages
        """
        result = {}

        # Stage 1: Ingest
        print(f"Stage 1: Ingesting tickets from {data_path}...")
        chunks = self.ingest.get_chunks(data_path, clean=clean)
        result["ingest"] = {"chunks_generated": len(chunks)}
        print(f"  Generated {len(chunks)} chunks")

        # Stage 2: Embed
        print(f"Stage 2: Embedding {len(chunks)} chunks...")
        embedded_chunks = self.embed_pipeline.process_chunks(chunks)
        result["embed"] = {
            "embeddings_created": len(embedded_chunks),
            "embedded_chunks": embedded_chunks
        }
        print(f"  Created {len(embedded_chunks)} embeddings")

        # Stage 3: Store
        if store and self.store_enabled:
            print(f"Stage 3: Storing {len(embedded_chunks)} embeddings in Qdrant...")
            store_pipeline = StorePipeline(vector_store=self.vector_store)
            store_result = store_pipeline.store_chunks(embedded_chunks)
            result["store"] = store_result
            print(f"  Stored {store_result['stored']} points")
        elif store and not self.store_enabled:
            print("Stage 3: Skipped - vector store not available")
            result["store"] = {"skipped": True, "reason": "vector_store_not_available"}

        if save:
            self.embed_pipeline.process_and_save(embedded_chunks.copy(), output_path)
            result["save"] = {"output_path": output_path}

        return result

    def run_from_chunks(self, chunks: List[Dict], save: bool = False,
                       output_path: Optional[str] = None,
                       store: bool = True) -> Dict:
        """Run Stage 2 + Stage 3 on already-ingested chunks.

        Args:
            chunks: Pre-ingested chunks
            save: Whether to save embeddings to file
            output_path: Output path for saved embeddings
            store: Whether to store embeddings in Qdrant

        Returns:
            Dict with results from all stages, including:
            - "embed": {"embeddings_created": int, "embedded_chunks": List[Dict]}
            - "store": {"stored": int, ...} (if store=True and store enabled)
            - "save": {"output_path": str} (if save=True)
        """
        result = {}

        # Stage 2: Embed
        print(f"Stage 2: Embedding {len(chunks)} chunks...")
        embedded_chunks = self.embed_pipeline.process_chunks(chunks)
        result["embed"] = {
            "embeddings_created": len(embedded_chunks),
            "embedded_chunks": embedded_chunks
        }
        print(f"  Created {len(embedded_chunks)} embeddings")

        # Stage 3: Store
        if store and self.store_enabled:
            print(f"Stage 3: Storing {len(embedded_chunks)} embeddings in Qdrant...")
            store_pipeline = StorePipeline(vector_store=self.vector_store)
            store_result = store_pipeline.store_chunks(embedded_chunks)
            result["store"] = store_result
            print(f"  Stored {store_result['stored']} points")
        elif store and not self.store_enabled:
            print("Stage 3: Skipped - vector store not available")
            result["store"] = {"skipped": True, "reason": "vector_store_not_available"}

        if save:
            self.embed_pipeline.process_and_save(embedded_chunks.copy(), output_path)
            result["save"] = {"output_path": output_path}

        return result

    def search(self, query_text: str, top_k: int = 10,
               service: Optional[str] = None,
               severity: Optional[str] = None) -> List[Dict]:
        """Search stored embeddings for similar content.

        Args:
            query_text: Text query to search for
            top_k: Number of results to return
            service: Optional filter by service
            severity: Optional filter by severity

        Returns:
            List of search results with scores and payloads
        """
        if not self.store_enabled:
            raise RuntimeError("Vector store is not enabled")

        # Embed the query
        query_embedding = self.embed_service.embed_text(query_text)

        # Search with optional filters
        return self.vector_store.filtered_search(
            query_vector=query_embedding,
            service=service,
            severity=severity,
            top_k=top_k
        )

    def hybrid_search(self, query_text: str, top_k: int = 10,
                      vector_top_k: Optional[int] = None,
                      bm25_top_k: Optional[int] = None) -> List[Dict]:
        """Hybrid search using vector + BM25 with RRF fusion.

        Args:
            query_text: Text query to search for
            top_k: Number of final results to return
            vector_top_k: Number of candidates from vector search
            bm25_top_k: Number of candidates from BM25 search

        Returns:
            List of hybrid search results with fused scores and metadata
        """
        if not self.store_enabled:
            raise RuntimeError("Vector store is not enabled")

        if not self.hybrid_retriever:
            raise RuntimeError("Hybrid retriever not initialized")

        return self.hybrid_retriever.retrieve(
            query=query_text,
            top_k=top_k,
            vector_top_k=vector_top_k,
            bm25_top_k=bm25_top_k
        )

    def get_storage_stats(self) -> Dict:
        """Get statistics about the stored embeddings."""
        if not self.store_enabled:
            return {"error": "Vector store not available"}

        return {
            "collection": self.vector_store.collection_name,
            "total_points": self.vector_store.count_points(),
            "host": self.vector_store.host,
            "port": self.vector_store.port,
            "dimensions": self.vector_store.dimensions
        }

    def hybrid_search_with_rerank(
        self,
        query_text: str,
        top_k: int = 10,
        vector_top_k: Optional[int] = None,
        bm25_top_k: Optional[int] = None,
        rerank_enabled: Optional[bool] = None
    ) -> List[Dict]:
        """Hybrid search with re-ranking using cross-encoder for precise relevance.

        This method combines Stage 4 (Hybrid Retrieve) with Stage 5 (Re-rank):
        1. Hybrid search retrieves ALL candidates (vector + BM25 with RRF fusion)
        2. Re-ranker applies cross-encoder to ALL candidates for optimal ordering
        3. Returns top_k results from the re-ranked list

        This approach ensures the most accurate results by using the cross-encoder
        to refine rankings for all retrieved documents before selecting the final top_k.

        Args:
            query_text: Text query to search for
            top_k: Number of final results to return after re-ranking
            vector_top_k: Number of candidates from vector search
            bm25_top_k: Number of candidates from BM25 search
            rerank_enabled: Override for re-ranking (default: class config)

        Returns:
            List of re-ranked hybrid search results with fused scores and rerank_score
        """
        if not self.store_enabled:
            raise RuntimeError("Vector store is not enabled")

        if not self.hybrid_retriever:
            raise RuntimeError("Hybrid retriever not initialized")

        if not self.rerank_pipeline:
            raise RuntimeError("Re-rank pipeline not initialized")

        # Get hybrid search results - retrieve all candidates (no top_k limit)
        # This ensures we re-rank ALL retrieved documents for optimal ordering
        hybrid_results = self.hybrid_retriever.retrieve(
            query=query_text,
            vector_top_k=vector_top_k,
            bm25_top_k=bm25_top_k
        )

        # Apply re-ranking to all candidates, then select top_k
        reranked_results = self.rerank_pipeline.run(
            query=query_text,
            documents=hybrid_results,
            top_k=top_k
        )

        return reranked_results


    def ground_context(
        self,
        reranked_results: List[Dict],
        query: Optional[str] = None
    ) -> GroundedContext:
        """Ground re-ranked results into a coherent context prompt with citations.
        
        This method combines Stage 5 (Re-rank) with Stage 6 (Ground):
        1. Takes re-ranked results from hybrid search
        2. Filters by minimum relevance score
        3. Selects top-n passages
        4. Builds context with citation markers
        5. Returns grounded context with citations and metadata
        
        Args:
            reranked_results: Re-ranked retrieval results from hybrid_search_with_rerank
            query: Original user query (reserved for future query-adaptive weighting)
            
        Returns:
            GroundedContext with context_text, citations, relevance_scores, and total_tokens
            
        Raises:
            RuntimeError: If ground pipeline is not initialized
        """
        if not self.ground_enabled:
            raise RuntimeError("Context grounding is disabled")
            
        if not self.ground_pipeline:
            raise RuntimeError("Ground pipeline not initialized")
            
        # Use ground pipeline to create grounded context
        grounded_context = self.ground_pipeline.ground_context(
            reranked_results=reranked_results,
            query=query
        )
        
        return grounded_context
    
    def hybrid_search_with_ground(
        self,
        query_text: str,
        top_k: int = 10,
        vector_top_k: Optional[int] = None,
        bm25_top_k: Optional[int] = None,
        rerank_enabled: Optional[bool] = None,
        ground_enabled: Optional[bool] = None,
        max_passages: Optional[int] = None,
        max_tokens: Optional[int] = None,
        min_relevance_score: Optional[float] = None
    ) -> Dict:
        """Hybrid search with re-ranking and context grounding.
        
        This method combines all stages 1-6:
        1. Hybrid search retrieves candidates (vector + BM25 with RRF fusion)
        2. Re-ranker applies cross-encoder for precise ordering
        3. Context grounder combines results into grounded prompt with citations
        
        This approach ensures the most accurate and traceable results by using
        the full pipeline from retrieval to grounded context generation.
        
        Args:
            query_text: Text query to search for
            top_k: Number of final results to return after re-ranking
            vector_top_k: Number of candidates from vector search
            bm25_top_k: Number of candidates from BM25 search
            rerank_enabled: Override for re-ranking (default: class config)
            ground_enabled: Override for context grounding (default: class config)
            max_passages: Override for max passages in grounded context (default: class config)
            max_tokens: Override for max tokens in grounded context (default: class config)
            min_relevance_score: Override for minimum relevance score (default: class config)
            
        Returns:
            Dict with:
                - hybrid_results: Original hybrid search results
                - reranked_results: Re-ranked results with rerank_score
                - grounded_context: GroundedContext with citations and context
        """
        if not self.store_enabled:
            raise RuntimeError("Vector store is not enabled")
            
        if not self.hybrid_retriever:
            raise RuntimeError("Hybrid retriever not initialized")
            
        # Get hybrid search results - retrieve all candidates (no top_k limit)
        hybrid_results = self.hybrid_retriever.retrieve(
            query=query_text,
            vector_top_k=vector_top_k,
            bm25_top_k=bm25_top_k
        )
        
        # Apply re-ranking
        if self.rerank_pipeline and (rerank_enabled if rerank_enabled is not None else self.rerank_enabled):
            reranked_results = self.rerank_pipeline.run(
                query=query_text,
                documents=hybrid_results,
                top_k=top_k
            )
        else:
            # Use raw hybrid results if re-ranking is disabled
            reranked_results = hybrid_results[:top_k]
            
        # Apply context grounding
        grounded_context = None
        if self.ground_enabled and self.ground_pipeline and (ground_enabled if ground_enabled is not None else self.ground_enabled):
            # Create context grounder with any overrides
            if max_passages is not None or max_tokens is not None or min_relevance_score is not None:
                grounder = ContextGrounding(
                    max_passages=max_passages or self.max_passages,
                    max_tokens=max_tokens or self.max_tokens,
                    min_relevance_score=min_relevance_score or self.min_relevance_score
                )
                grounded_context = grounder.ground_context(
                    reranked_results=reranked_results,
                    query=query_text
                )
            else:
                grounded_context = self.ground_pipeline.ground_context(
                    reranked_results=reranked_results,
                    query=query_text
                )
        
        return {
            "hybrid_results": hybrid_results,
            "reranked_results": reranked_results,
            "grounded_context": grounded_context
        }
    
    def get_grounding_stats(self, grounded_context: GroundedContext) -> Dict:
        """Get statistics about a grounded context.
        
        Args:
            grounded_context: GroundedContext to analyze
            
        Returns:
            Dict with statistics about the grounded context
        """
        if not self.ground_pipeline:
            raise RuntimeError("Ground pipeline not initialized")
            
        return self.ground_pipeline.get_stats(grounded_context)
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
        """Run the complete pipeline and generate an answer.
        
        This method executes all stages up to Stage 7 (Generate):
        - Stage 1: Ingest (if store not enabled)
        - Stage 2: Embed
        - Stage 3: Store (if store not enabled)
        - Stage 4: Retrieve (hybrid search)
        - Stage 5: Re-rank (optional)
        - Stage 6: Ground (optional)
        - Stage 7: Generate (answer generation)

        Args:
            query_text: User's incident question
            top_k: Number of final results to return after re-ranking
            vector_top_k: Number of candidates from vector search
            bm25_top_k: Number of candidates from BM25 search
            rerank_enabled: Override for re-ranking (default: class config)
            ground_enabled: Override for context grounding (default: class config)
            max_passages: Override for max passages in grounded context (default: class config)
            max_tokens: Override for max tokens in grounded context (default: class config)
            min_relevance_score: Override for minimum relevance score (default: class config)
            
        Returns:
            Dict with:
                - hybrid_results: Original hybrid search results
                - reranked_results: Re-ranked results with rerank_score
                - grounded_context: GroundedContext with citations and context
                - answer: GeneratedAnswer with answer text and metadata
        """
        if not self.store_enabled:
            raise RuntimeError("Vector store is not enabled")
            
        if not self.hybrid_retriever:
            raise RuntimeError("Hybrid retriever not initialized")
        
        # Get grounded results
        grounded_results = self.hybrid_search_with_ground(
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
        
        # Generate answer if grounded context was created
        answer = None
        grounded_context = grounded_results.get("grounded_context")
        if grounded_context and self.generate_enabled and self.generate_pipeline:
            answer = self.generate_pipeline.generate_answer(
                question=query_text,
                context=grounded_context.context_text,
                citations=grounded_context.citations
            )

            # Apply guardrails to answer if guard pipeline is enabled
            if self.guard_pipeline and answer:
                guard_result = self.guard_pipeline.process_answer(
                    answer=answer.answer,
                    question=query_text,
                    citations=grounded_context.citations,
                    embedding_service=self.embed_service
                )
                
                # Update answer with guarded results
                answer.answer = guard_result["modified_answer"]
                # Store guard result in grounded_results for inspection
                grounded_results["guard_result"] = guard_result
        
        # Add answer to results
        grounded_results["answer"] = answer
        # Record observability metrics after answer generation
        if self.observability and self.observability.enable_metrics:
            total_duration = self.observability.collector.observer.get_total_time()
            if total_duration:
                self.observability.collector.record_pipeline("answer", total_duration * 1000)

        return grounded_results
    def get_observability_metrics(self) -> Dict[str, Any]:
        """Get observability metrics if enabled.

        Returns:
            Dictionary containing all collected metrics or empty dict if disabled
        """
        if self.observability:
            return self.observability.get_metrics()
        return {}

    def reset_observability(self):
        """Reset observability metrics."""
        if self.observability:
            self.observability.reset()

# Backward compatibility aliases
SupportDeskRAGPipelineWithStore = SupportDeskRAGPipeline


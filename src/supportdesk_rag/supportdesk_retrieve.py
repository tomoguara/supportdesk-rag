"""Stage 4: Retrieve - Hybrid Search with BM25 + RRF Fusion

Provides hybrid search combining vector similarity with keyword (BM25) matching.
Uses Reciprocal Rank Fusion (RRF) to blend results from both retrieval methods.

For SupportDesk RAG, this stage:
- Combines semantic (vector) and lexical (BM25) search
- Uses RRF fusion for robust results without tuning
- Returns ranked results with fused scores

For the unified 4-stage pipeline (Ingest + Embed + Store + Retrieve),
see `supportdesk_rag_pipeline`.
"""
import os
import json
from typing import List, Dict, Optional
from pathlib import Path

import yaml
from tqdm import tqdm

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from supportdesk_rag.model_selector import (
    get_embedding_model_name,
)


class Config:
    """Configuration manager for SupportDesk RAG system."""

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
                "openai": {
                    "api_key_env": "OPENAI_API_KEY",
                    "embedding_model": "text-embedding-3-small",
                    "embedding_dimensions": 1536,
                    "batch_size": 100
                },
                "ingest": {
                    "chunk_size": 1000,
                    "chunk_overlap": 150,
                    "chunk_strategy": "recursive",
                    "clean_pii": True
                },
                "embed": {
                    "model": "text-embedding-3-small",
                    "batch_size": 100,
                    "output_directory": "data/embedded"
                },
                "store": {
                    "qdrant": {
                        "host": "localhost",
                        "port": 6333,
                        "collection_name": "tickets",
                        "metrics": False
                    }
                },
                "retrieve": {
                    "vector_weight": 0.7,
                    "bm25_weight": 0.3,
                    "k_rrf": 60,
                    "vector_top_k": 20,
                    "bm25_top_k": 20
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

    def get_openai_config(self) -> Dict:
        return self._config.get("openai", {})

    def get_ingest_config(self) -> Dict:
        return self._config.get("ingest", {})

    def get_embed_config(self) -> Dict:
        return self._config.get("embed", {})

    def get_store_config(self) -> Dict:
        return self._config.get("store", {})

    def get_retrieve_config(self) -> Dict:
        return self._config.get("retrieve", {})


class BM25Index:
    """Simple BM25 index for keyword search using tiktoken for tokenization.

    This is a demo implementation suitable for small to medium ticket sets.
    For production, consider using Elasticsearch or OpenSearch with BM25.
    """

    def __init__(self):
        self.bm25 = None
        self.documents = []
        self.document_ids = []

    def build(self, texts: List[str], doc_ids: Optional[List[str]] = None):
        """Build BM25 index from texts.

        Args:
            texts: List of text documents to index
            doc_ids: Optional list of document IDs. If not provided, generates UUIDs.
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "rank-bm25 package is required. Install with: pip install rank-bm25"
            )

        # Use tiktoken for tokenization (already in codebase)
        import tiktoken

        # Use cl100k_base encoding which is used by text-embedding models
        encoding = tiktoken.get_encoding("cl100k_base")

        # Tokenize documents
        tokenized_docs = []
        for text in texts:
            # Decode tokens back to words for BM25
            tokens = encoding.encode(text)
            # Simple lowercasing and word-based tokenization
            decoded = encoding.decode(tokens).lower()
            words = decoded.split()
            tokenized_docs.append(words)

        self.bm25 = BM25Okapi(tokenized_docs)
        self.documents = texts
        self.document_ids = doc_ids or [str(i) for i in range(len(texts))]

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Search BM25 index and return top-k results with scores.

        Args:
            query: Search query string
            top_k: Number of top results to return

        Returns:
            List of results with document, score, and rank
        """
        if self.bm25 is None:
            return []

        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "rank-bm25 package is required. Install with: pip install rank-bm25"
            )

        # Tokenize query using the same encoding
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")

        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices and scores
        import numpy as np
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            if scores[idx] > 0:
                doc_id = self.document_ids[idx] if idx < len(self.document_ids) else str(idx)
                results.append({
                    "document": self.documents[idx],
                    "score": float(scores[idx]),
                    "rank": rank,
                    "doc_id": doc_id
                })

        return results


class HybridRetriever:
    """Hybrid retriever combining vector search + BM25 with RRF fusion.

    For SupportDesk RAG, this provides:
    - Semantic search via vector similarity (vector_weight)
    - Lexical search via BM25 keyword matching (bm25_weight)
    - RRF fusion for robust, tuned results
    """

    def __init__(
        self,
        vector_store,
        embedding_service,
        bm25_index: Optional[BM25Index] = None,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        k_rrf: int = 60,
        vector_top_k: int = 20,
        bm25_top_k: int = 20
    ):
        """
        Initialize the hybrid retriever.

        Args:
            vector_store: VectorStore instance (Stage 3)
            embedding_service: EmbeddingService instance (Stage 2)
            bm25_index: Optional BM25Index. If None, creates on-the-fly index
            vector_weight: Weight for vector search scores in RRF fusion
            bm25_weight: Weight for BM25 scores in RRF fusion
            k_rrf: RRF constant (default 60)
            vector_top_k: Number of candidates from vector search
            bm25_top_k: Number of candidates from BM25 search
        """
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.bm25_index = bm25_index
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.k_rrf = k_rrf
        self.vector_top_k = vector_top_k
        self.bm25_top_k = bm25_top_k

        # Build BM25 index from vector store if not provided
        if self.bm25_index is None:
            self._build_bm25_from_vector_store()

    def _build_bm25_from_vector_store(self):
        """Build BM25 index from documents in vector store."""
        # Get all documents from vector store
        try:
            from qdrant_client.models import Filter

            # Scan all points (with limit for demo purposes)
            points = []
            offset = None

            while True:
                result = self.vector_store.client.scroll(
                    collection_name=self.vector_store.collection_name,
                    limit=100,
                    scroll_filter=Filter(),
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
                points.extend(result[0])
                offset = result[1]

                if offset is None:
                    break

            texts = []
            doc_ids = []
            for point in points:
                payload = point.payload
                text = payload.get("chunk_text", "")
                if text:
                    texts.append(text)
                    doc_ids.append(str(point.id))

            if texts:
                self.bm25_index = BM25Index()
                self.bm25_index.build(texts, doc_ids)

        except Exception as e:
            print(f"Warning: Could not build BM25 index from vector store: {e}")
            print("BM25 search will not be available. Consider providing a BM25 index.")

    def rrf_fusion(
        self,
        vector_results: List[Dict],
        bm25_results: List[Dict],
        k: int = 60
    ) -> List[Dict]:
        """Reciprocal Rank Fusion (RRF) to combine retrieval results.

        RRF Formula: fused_score(doc) = sum(1 / (k + rank_i(doc))) for each retriever i

        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search
            k: RRF constant (default 60)

        Returns:
            Fused results sorted by fused score
        """
        fused_scores = {}

        # Process vector results (rank starts at 1)
        for rank, result in enumerate(vector_results, start=1):
            doc_id = result.get("id") or result.get("doc_id")
            if doc_id:
                fused_scores[doc_id] = {
                    "score": 1 / (k + rank),
                    "vector_rank": rank,
                    "bm25_rank": None
                }

        # Process BM25 results (rank starts at 1)
        for rank, result in enumerate(bm25_results, start=1):
            doc_id = result.get("doc_id")
            if doc_id:
                if doc_id in fused_scores:
                    fused_scores[doc_id]["score"] += 1 / (k + rank)
                    fused_scores[doc_id]["bm25_rank"] = rank
                else:
                    fused_scores[doc_id] = {
                        "score": 1 / (k + rank),
                        "vector_rank": None,
                        "bm25_rank": rank
                    }

        # Sort by fused score (descending)
        sorted_docs = sorted(
            fused_scores.items(),
            key=lambda x: x[1]["score"],
            reverse=True
        )

        # Build final results
        results = []
        for doc_id, score_data in sorted_docs:
            result = {}

            # Find the original result with payload
            for r in vector_results:
                if r.get("id") == doc_id or str(r.get("doc_id")) == doc_id:
                    result = r.copy()
                    break

            if not result:
                for r in bm25_results:
                    if r.get("doc_id") == doc_id:
                        result = {
                            "id": doc_id,
                            "document": r["document"],
                            "score": r["score"],
                            "rank": r["rank"]
                        }
                        break

            # Normalize result: extract document from payload if needed
            payload = result.get("payload")
            if payload and "chunk_text" in payload and "document" not in result:
                result["document"] = payload["chunk_text"]
            elif payload and "chunk_text" not in payload and "document" not in result:
                # Fallback: use other payload fields to build document
                result["document"] = ""

            result["fused_score"] = score_data["score"]
            result["retriever"] = "vector" if score_data.get("vector_rank") else "bm25"
            result["vector_rank"] = score_data.get("vector_rank")
            result["bm25_rank"] = score_data.get("bm25_rank")
            results.append(result)

        return results

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        query_embedding: Optional[List[float]] = None,
        vector_top_k: Optional[int] = None,
        bm25_top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Retrieve using hybrid search.

        Args:
            query: Search query text
            top_k: Number of final results to return
            query_embedding: Optional pre-computed embedding. If None, computed from query
            vector_top_k: Override for vector search candidates
            bm25_top_k: Override for BM25 search candidates

        Returns:
            List of fused results with scores and metadata
        """
        # Use provided values or defaults
        vector_k = vector_top_k or self.vector_top_k
        bm25_k = bm25_top_k or self.bm25_top_k

        # 1. Compute query embedding if not provided
        if query_embedding is None:
            query_embedding = self.embedding_service.embed_text(query)

        # 2. Vector search
        vector_results = self.vector_store.search(
            query_vector=query_embedding,
            top_k=vector_k
        )

        # 3. BM25 search (if index available)
        bm25_results = []
        if self.bm25_index:
            bm25_results = self.bm25_index.search(query, top_k=bm25_k)

        # 4. Fuse results
        fused_results = self.rrf_fusion(vector_results, bm25_results, k=self.k_rrf)

        # 5. Return top-k fused results
        return fused_results[:top_k]


def create_hybrid_retriever(
    vector_store,
    embedding_service,
    bm25_index: Optional[BM25Index] = None,
    **kwargs
) -> HybridRetriever:
    """Factory function to create a HybridRetriever instance.

    Args:
        vector_store: VectorStore instance (Stage 3)
        embedding_service: EmbeddingService instance (Stage 2)
        bm25_index: Optional BM25Index
        **kwargs: Additional arguments for HybridRetriever

    Returns:
        HybridRetriever instance
    """
    return HybridRetriever(
        vector_store=vector_store,
        embedding_service=embedding_service,
        bm25_index=bm25_index,
        **kwargs
    )


def create_bm25_index(texts: List[str], doc_ids: Optional[List[str]] = None) -> BM25Index:
    """Factory function to create and populate a BM25Index.

    Args:
        texts: List of document texts
        doc_ids: Optional list of document IDs

    Returns:
        BM25Index instance populated with documents
    """
    index = BM25Index()
    index.build(texts, doc_ids)
    return index


# Backward compatibility alias
SupportDeskRetriever = HybridRetriever
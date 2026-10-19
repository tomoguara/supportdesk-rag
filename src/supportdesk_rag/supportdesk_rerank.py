"""Stage 5: Re-rank - Cross-Encoder Re-ranking for Hybrid Search Results

Provides re-ranking capabilities using cross-encoder models to get finer-grained
relevance scores for hybrid search candidates. This stage improves precision by
using a cross-encoder to re-rank documents retrieved by the hybrid search (vector + BM25).

For SupportDesk RAG, this stage:
- Re-ranks top candidates from hybrid search using cross-encoder
- Provides more accurate relevance scoring
- Optional configuration to enable/disable re-ranking
"""

import os
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


class Config:
    """Configuration manager for Stage 5: Re-rank."""

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
                "rerank": {
                    "enabled": True,
                    "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                    "top_k": 10,
                    "min_score": -5.0,
                    "batch_size": 32
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

    def get_rerank_config(self) -> Dict:
        """Get re-rank specific configuration."""
        return self._config.get("rerank", {})


class ReRanker:
    """Cross-encoder re-ranker for retrieval results.

    Uses a cross-encoder model (e.g., cross-encoder/ms-marco-MiniLM-L-6-v2)
    to re-rank documents based on query relevance. This provides more accurate
    scoring than simple vector similarity.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        min_score: float = -5.0,
        batch_size: int = 32
    ):
        """
        Initialize re-ranker model.

        Args:
            model_name: Cross-encoder model name (default from config)
            min_score: Minimum rerank score to include (default: -5.0)
            batch_size: Batch size for processing (default: 32)
        """
        from sentence_transformers import CrossEncoder

        config = Config()
        self.model_name = model_name or config.get_rerank_config().get(
            "model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        self.min_score = min_score
        self.batch_size = batch_size

        try:
            self.model = CrossEncoder(self.model_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize cross-encoder model '{self.model_name}'. "
                f"Make sure sentence-transformers is installed: pip install sentence-transformers"
            ) from e

    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Re-rank documents based on query relevance.
        """
        if not documents:
            return []

        config = Config()
        if top_k is None:
            top_k = config.get_rerank_config().get("top_k", 10)

        # Prepare query-document pairs for cross-encoder
        pairs = [
            (query, _get_document_content(doc))
            for doc in documents
        ]

        # Get relevance scores
        scores = self.model.predict(pairs, batch_size=self.batch_size)

        # Attach scores and sort
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)

        # Sort by rerank score (higher = more relevant)
        documents.sort(key=lambda x: x["rerank_score"], reverse=True)

        return documents[:top_k]

    def rerank_with_metadata(
        self,
        query: str,
        documents: List[Dict],
        top_k: Optional[int] = None,
        min_score: Optional[float] = None
    ) -> List[Dict]:
        """Re-rank with metadata filtering."""
        if not documents:
            return []

        config = Config()
        if top_k is None:
            top_k = config.get_rerank_config().get("top_k", 10)
        if min_score is None:
            min_score = self.min_score

        # First re-rank all documents
        reranked = self.rerank(query, documents, top_k=len(documents))

        # Filter by minimum score
        filtered = [
            doc for doc in reranked
            if doc.get("rerank_score", float('-inf')) >= min_score
        ]

        return filtered[:top_k]

    def rerank_batch(
        self,
        query_doc_pairs: List[Dict],
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """Re-rank multiple query-document sets in batch."""
        results = []
        for item in query_doc_pairs:
            query = item.get("query", "")
            docs = item.get("documents", [])
            reranked = self.rerank(query, docs, top_k=top_k)
            results.append(reranked)
        return results


def _get_document_content(doc: Dict) -> str:
    """Extract content from a document dictionary."""
    if "payload" in doc and isinstance(doc["payload"], dict):
        return doc["payload"].get("content", "")
    if "chunk_text" in doc:
        return doc["chunk_text"]
    if "document" in doc:
        return doc["document"]
    return ""


def create_reranker(**kwargs) -> ReRanker:
    """Factory function to create a ReRanker instance."""
    return ReRanker(**kwargs)


class ReRankerPipeline:
    """Pipeline for re-ranking hybrid search results in the SupportDesk RAG system."""

    def __init__(
        self,
        reranker: Optional[ReRanker] = None,
        enabled: Optional[bool] = None,
        top_k: Optional[int] = None
    ):
        """
        Initialize re-rank pipeline.
        """
        config = Config()

        self.reranker = reranker or ReRanker()
        self.enabled = enabled if enabled is not None else config.get_rerank_config().get("enabled", True)
        self.top_k = top_k or config.get_rerank_config().get("top_k", 10)

    def run(
        self,
        query: str,
        documents: List[Dict],
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Run re-ranking pipeline on hybrid search results.
        """
        if not self.enabled:
            return documents

        if not documents:
            return []

        return self.reranker.rerank(query, documents, top_k=top_k or self.top_k)

    def run_batch(
        self,
        query_doc_pairs: List[Dict]
    ) -> List[List[Dict]]:
        """
        Run re-ranking on multiple queries in batch.
        """
        if not self.enabled:
            return [item.get("documents", []) for item in query_doc_pairs]

        return self.reranker.rerank_batch(query_doc_pairs)


# Backward compatibility aliases
SupportDeskReRanker = ReRanker
SupportDeskReRankerPipeline = ReRankerPipeline

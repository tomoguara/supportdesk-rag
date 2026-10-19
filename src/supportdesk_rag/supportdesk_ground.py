"""Stage 6: Ground - Combine re-ranked documents into grounded context prompt.

This module provides functionality to combine re-ranked documents from Stage 5
into a coherent context prompt with citation tracking for the SupportDesk RAG system.

Key features:
- Passage scoring and filtering based on relevance threshold
- Citation tracking for traceability
- Token budget management for LLM context windows
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Optional
from pathlib import Path
import yaml

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class GroundedContext:
    """Context assembled for the LLM with citation metadata.
    
    Args:
        context_text: Combined text of selected passages with citation markers
        citations: List of citation dictionaries with ticket IDs, scores, and ranks
        relevance_scores: List of rerank scores for each citation
        total_tokens: Estimated token count for the context
    """
    context_text: str
    citations: List[Dict]
    relevance_scores: List[float]
    total_tokens: int


class Config:
    """Configuration manager for Stage 6: Ground."""
    
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
                "ground": {
                    "max_passages": 5,
                    "max_tokens": 4000,
                    "min_relevance_score": -3.0
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
    
    def get_ground_config(self) -> Dict:
        """Get ground-specific configuration."""
        return self._config.get("ground", {})


class ContextGrounding:
    """Ground retrieval results into a coherent context prompt.
    
    This class combines re-ranked documents from Stage 5 into a grounded
    context prompt with citation tracking. It implements the recommended
    approach for SupportDesk RAG: simple passage concatenation with
    relevance filtering.
    
    Usage:
        ```python
        from supportdesk_rag import ContextGrounding
        
        grounder = ContextGrounding(max_passages=5, max_tokens=4000)
        
        grounded = grounder.ground_context(
            reranked_results=reranked_results,
            query=query
        )
        
        print(f"Context has {len(grounded.citations)} citations")
        print(f"Estimated tokens: {grounded.total_tokens}")
        print(grounded.context_text)
        ```
    """
    
    def __init__(
        self,
        max_passages: Optional[int] = None,
        max_tokens: Optional[int] = None,
        min_relevance_score: Optional[float] = None
    ):
        """
        Initialize context grounder.
        
        Args:
            max_passages: Maximum number of passages to include (default: 5)
            max_tokens: Maximum token count for context (default: 4000)
            min_relevance_score: Minimum rerank score to include (default: -3.0)
        """
        config = Config()
        
        self.max_passages = (
            max_passages 
            or config.get_ground_config().get("max_passages", 5)
        )
        self.max_tokens = (
            max_tokens 
            or config.get_ground_config().get("max_tokens", 4000)
        )
        self.min_relevance_score = (
            min_relevance_score
            or config.get_ground_config().get("min_relevance_score", -3.0)
        )
    
    def ground_context(
        self,
        reranked_results: List[Dict],
        query: Optional[str] = None,
        embedding_service: Optional[object] = None
    ) -> GroundedContext:
        """
        Assemble grounded context from reranked results.
        
        Args:
            reranked_results: Re-ranked retrieval results from Stage 5
            query: Original user question (reserved for future query-adaptive weighting)
            embedding_service: Optional for future query-adaptive weighting
            
        Returns:
            GroundedContext with citations and metadata
        """
        # 1. Filter by minimum relevance score
        # Support both 'rerank_score' (Stage 5 re-rank) and 'score' (vector search score)
        relevant_results = [
            r for r in reranked_results
            if r.get("rerank_score", r.get("score", float('-inf'))) >= self.min_relevance_score
        ]
        
        # 2. Sort by rerank score (should already be sorted, but ensure)
        relevant_results.sort(
            key=lambda x: x.get("rerank_score", x.get("score", float('-inf'))), 
            reverse=True
        )
        
        # 3. Take top-n passages
        selected = relevant_results[:self.max_passages]
        
        # 4. Build context text with citation markers
        context_parts = []
        citations = []
        
        for i, result in enumerate(selected):
            # Extract passage content from various possible locations
            passage = ""
            if "payload" in result and isinstance(result["payload"], dict):
                passage = result["payload"].get("chunk_text", "")
                if not passage:
                    passage = result["payload"].get("content", "")
            elif "chunk_text" in result:
                passage = result["chunk_text"]
            elif "document" in result:
                passage = result["document"]
            
            # Extract ticket_id or generate fallback ID
            ticket_id = ""
            if "payload" in result and isinstance(result["payload"], dict):
                ticket_id = result["payload"].get("ticket_id", "")
            if not ticket_id:
                ticket_id = result.get("ticket_id", "")
            if not ticket_id:
                ticket_id = result.get("id", f"passage_{i}")
            
            # Add citation marker with passage
            context_parts.append(f"[CITATION:{ticket_id}]\n{passage}\n")
            
            # Track citation metadata
            citations.append({
                "citation_id": f"[CITATION:{ticket_id}]",
                "ticket_id": ticket_id,
                "score": result.get("rerank_score", result.get("score", 0)),
                "rank": i + 1,
                "retriever": result.get("retriever", "unknown"),
                "content_preview": passage[:100] if passage else ""
            })
        
        # Join context parts with separator
        context_text = "\n---\n".join(context_parts)
        
        # 5. Estimate token count (rough heuristic: 1 token ≈ 4 chars)
        estimated_tokens = len(context_text) // 4
        
        return GroundedContext(
            context_text=context_text,
            citations=citations,
            relevance_scores=[c["score"] for c in citations],
            total_tokens=estimated_tokens
        )
    
    def truncate_context(
        self,
        context: GroundedContext,
        max_tokens: Optional[int] = None
    ) -> GroundedContext:
        """
        Truncate context to stay within token budget.
        
        Strategy: Remove lowest-scoring passages first.
        
        Args:
            context: GroundedContext to truncate
            max_tokens: Maximum token count (default: instance max_tokens)
            
        Returns:
            Truncated GroundedContext
        """
        target_tokens = max_tokens or self.max_tokens
        
        if context.total_tokens <= target_tokens:
            return context
        
        # Build mapping from ticket_id to content for reconstruction
        id_to_content = {}
        for citation in context.citations:
            ticket_id = citation["ticket_id"]
            # Extract the content for this citation from context_text
            # Find content between this citation marker and the next
            marker = f"[CITATION:{ticket_id}]"
            idx = context.context_text.find(marker)
            if idx != -1:
                # Find end (either next separator or end of string)
                next_sep = context.context_text.find("\n---\n", idx)
                if next_sep != -1:
                    content_end = next_sep
                else:
                    content_end = len(context.context_text)
                id_to_content[ticket_id] = context.context_text[idx:content_end]
        
        # Sort citations by score (ascending) to remove lowest first
        sorted_citations = sorted(
            context.citations, 
            key=lambda x: x["score"]
        )
        
        # Remove lowest-scoring citations until under budget
        kept_citations = []
        total = 0
        
        for citation in sorted_citations:
            ticket_id = citation["ticket_id"]
            if ticket_id in id_to_content:
                content_length = len(id_to_content[ticket_id])
                estimated_tokens = content_length // 4
                
                if total + estimated_tokens <= target_tokens:
                    kept_citations.append(citation)
                    total += estimated_tokens
        
        # Rebuild context text from kept citations
        # Sort kept citations by rank for proper order
        kept_citations.sort(key=lambda x: x["rank"])
        
        kept_ids = {c["ticket_id"] for c in kept_citations}
        context_parts = []
        
        for citation in kept_citations:
            ticket_id = citation["ticket_id"]
            if ticket_id in id_to_content:
                context_parts.append(id_to_content[ticket_id])
        
        context_text = "\n---\n".join(context_parts)
        
        # Recalculate token count
        estimated_tokens = len(context_text) // 4
        
        return GroundedContext(
            context_text=context_text,
            citations=kept_citations,
            relevance_scores=[c["score"] for c in kept_citations],
            total_tokens=estimated_tokens
        )
    
    def get_stats(self, context: GroundedContext) -> Dict:
        """
        Get statistics about a grounded context.
        
        Args:
            context: GroundedContext to analyze
            
        Returns:
            Dict with statistics
        """
        return {
            "num_citations": len(context.citations),
            "total_tokens": context.total_tokens,
            "average_score": (
                sum(context.relevance_scores) / len(context.relevance_scores)
                if context.relevance_scores else 0
            ),
            "min_score": min(context.relevance_scores) if context.relevance_scores else 0,
            "max_score": max(context.relevance_scores) if context.relevance_scores else 0,
            "retrievers_used": list(set(c.get("retriever", "unknown") for c in context.citations))
        }


# Factory function
def create_context_grounder(**kwargs) -> ContextGrounding:
    """Create a ContextGrounding instance with optional parameters.
    
    Args:
        **kwargs: Arguments for ContextGrounding initialization
        
    Returns:
        ContextGrounding instance
    """
    return ContextGrounding(**kwargs)


# Backward compatibility aliases
SupportDeskGrounding = ContextGrounding
SupportDeskGroundedContext = GroundedContext

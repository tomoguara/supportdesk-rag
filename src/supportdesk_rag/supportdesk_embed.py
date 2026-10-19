"""Stage 2: Embedding Pipeline for SupportDesk RAG system.

This module provides the embedding pipeline components:

- Config: Configuration manager
- EmbeddingService: OpenAI embedding service
- EmbeddingPipeline: Pipeline to embed chunks after ingestion
- create_embedding_pipeline: Factory function

For the unified 3-stage pipeline (Ingest + Embed + Store), see `supportdesk_rag_pipeline`.
"""
import os
import json
import uuid
from typing import List, Dict, Optional
from pathlib import Path
from tqdm import tqdm
import yaml
import openai

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from supportdesk_rag.model_selector import (
    get_embedding_model_name,
    get_base_url,
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
                    "vector_db_path": "data/vector_db",
                    "collection_name": "tickets"
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

    def get_rerank_config(self) -> Dict:
        return self._config.get("rerank", {})


class EmbeddingService:
    """Embedding service using OpenAI's embedding models."""

    def __init__(self, model: Optional[str] = None, batch_size: Optional[int] = None):
        from supportdesk_rag.model_selector import get_client_kwargs

        self.config = Config()
        self.model = model or get_embedding_model_name()
        self.batch_size = batch_size or self.config.get_embed_config().get("batch_size", 100)
        self.dimensions = self.config.get_openai_config().get("embedding_dimensions", 1536)

        # Use model_selector to get OpenAI client kwargs
        kwargs = get_client_kwargs()
        self.client = openai.OpenAI(**kwargs)

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string."""
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model,
                encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            raise RuntimeError(f"Embedding failed for text: {e}")

    def embed_batch(self, texts: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        """Embed multiple texts in batches."""
        bs = batch_size if batch_size is not None else self.batch_size
        embeddings = []

        for i in tqdm(range(0, len(texts), bs), desc="Embedding batches"):
            batch = texts[i:i + bs]
            try:
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model,
                    encoding_format="float"
                )
                embeddings.extend([d.embedding for d in response.data])
            except Exception as e:
                print(f"Batch {i//bs} failed: {e}")
                embeddings.extend([[0.0] * self.dimensions for _ in batch])

        return embeddings


class EmbeddingPipeline:
    """Pipeline to embed chunks after ingestion."""

    def __init__(self, embedding_service: Optional[EmbeddingService] = None,
                 batch_size: Optional[int] = None):
        self.config = Config()
        self.embedding_service = embedding_service or EmbeddingService()
        self.batch_size = batch_size or self.config.get_embed_config().get("batch_size", 100)

    def process_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Embed all chunks and attach embeddings to metadata."""
        texts = [chunk["chunk_text"] for chunk in chunks]
        embeddings = self.embedding_service.embed_batch(texts, batch_size=self.batch_size)

        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding
            if "id" not in chunk:
                chunk["id"] = str(uuid.uuid4())

        return chunks

    def process_and_save(self, chunks: List[Dict], output_path: Optional[str] = None) -> List[Dict]:
        """Embed all chunks, attach embeddings, and save to file."""
        embedded_chunks = self.process_chunks(chunks)

        config_output_dir = self.config.get_embed_config().get("output_directory", "data/embedded")
        if output_path is None:
            output_path = os.path.join(config_output_dir, "embedded_tickets.json")

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({
                "chunks": embedded_chunks,
                "metadata": {
                    "total_chunks": len(embedded_chunks),
                    "embedding_model": self.embedding_service.model,
                    "batch_size": self.batch_size,
                    "generated_at": str(uuid.uuid4()),
                    "output_path": str(output_file)
                }
            }, f, indent=2, default=str)

        print(f"Embedded chunks saved to: {output_path}")
        return embedded_chunks


def create_embedding_pipeline(
    model: Optional[str] = None,
    batch_size: Optional[int] = None
) -> EmbeddingPipeline:
    """Factory function to create an embedding pipeline."""
    embedding_service = EmbeddingService(model=model, batch_size=batch_size)
    return EmbeddingPipeline(embedding_service=embedding_service, batch_size=batch_size)


# Stage 1 + 2 only pipeline (no Stage 3 integration)
class SupportDeskRAGPipeline:
    """Unified pipeline for Stage 1 (Ingest) + Stage 2 (Embed).

    Provides end-to-end processing from raw tickets to embedded chunks
    ready for vector storage.

    For the unified 3-stage pipeline including storage, use `supportdesk_rag_pipeline.SupportDeskRAGPipeline`.
    """

    def __init__(self, chunk_size: Optional[int] = None,
                 chunk_overlap: Optional[int] = None,
                 batch_size: Optional[int] = None):
        from supportdesk_rag.supportdesk_ingest import SupportDeskIngest

        self.config = Config()
        self.chunk_size = chunk_size or self.config.get_ingest_config().get("chunk_size", 1000)
        self.chunk_overlap = chunk_overlap or self.config.get_ingest_config().get("chunk_overlap", 150)
        self.batch_size = batch_size or self.config.get_embed_config().get("batch_size", 100)

        self.ingest = SupportDeskIngest(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        self.embed_service = EmbeddingService(batch_size=self.batch_size)
        self.embed_pipeline = EmbeddingPipeline(
            embedding_service=self.embed_service,
            batch_size=self.batch_size
        )

    def run(self, data_path: str, save: bool = False,
            output_path: Optional[str] = None, clean: bool = True) -> List[Dict]:
        """Run the complete Stage 1 + Stage 2 pipeline."""
        print(f"Stage 1: Ingesting tickets from {data_path}...")
        chunks = self.ingest.get_chunks(data_path, clean=clean)
        print(f"  Generated {len(chunks)} chunks")

        print(f"Stage 2: Embedding {len(chunks)} chunks...")
        embedded_chunks = self.embed_pipeline.process_chunks(chunks)
        print(f"  Created {len(embedded_chunks)} embeddings")

        if save:
            if output_path is None:
                config_output_dir = self.config.get_embed_config().get("output_directory", "data/embedded")
                output_path = os.path.join(config_output_dir, "embedded_tickets.json")

            print(f"Saving embedded chunks to {output_path}...")
            self.embed_pipeline.process_and_save(embedded_chunks.copy(), output_path)

        return embedded_chunks

    def run_from_chunks(self, chunks: List[Dict], save: bool = False,
                       output_path: Optional[str] = None) -> List[Dict]:
        """Run Stage 2 embedding on already-ingested chunks."""
        print(f"Stage 2: Embedding {len(chunks)} chunks...")
        embedded_chunks = self.embed_pipeline.process_chunks(chunks)
        print(f"  Created {len(embedded_chunks)} embeddings")

        if save:
            self.embed_pipeline.process_and_save(embedded_chunks.copy(), output_path)

        return embedded_chunks


# Backward compatibility alias for the unified 3-stage pipeline
# Import the unified 3-stage pipeline class
from supportdesk_rag.supportdesk_rag_pipeline import SupportDeskRAGPipeline as _UnifiedPipeline


class SupportDeskRAGPipelineWithStore(_UnifiedPipeline):
    """Alias for the unified 3-stage pipeline (Stage 1 + Stage 2 + Stage 3).

    This class provides backward compatibility. It is now an alias for
    `SupportDeskRAGPipeline` defined in `supportdesk_rag_pipeline`.

    Usage:
        ```python
        from supportdesk_rag import SupportDeskRAGPipelineWithStore

        pipeline = SupportDeskRAGPipelineWithStore()
        result = pipeline.run("data/tickets.json", store=True)
        ```
    """

    pass

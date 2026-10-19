"""Stage 3: Storage - Vector Database Integration

Persists vectors + metadata in Qdrant for fast top-k similarity search.
Supports filtered queries by metadata (e.g., specific service or date range).
"""
import os
import uuid
from typing import List, Dict, Optional
from pathlib import Path

import yaml
from tqdm import tqdm

from supportdesk_rag.model_selector import get_embedding_model_name


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

    def get_qdrant_config(self) -> Dict:
        """Get Qdrant-specific configuration."""
        return self._config.get("store", {}).get("qdrant", {})


class VectorStore:
    """Vector store using Qdrant for storage and similarity search.

    Provides:
    - Persistent vector storage
    - Fast top-k similarity search
    - Metadata filtering (service, severity, etc.)
    - Upsert operations for updates
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        collection_name: Optional[str] = None,
        dimensions: Optional[int] = None,
        skip_init: bool = False
    ):
        """Initialize Qdrant client and create/retrieve collection.

        Args:
            host: Qdrant host (default: localhost from config)
            port: Qdrant port (default: 6333 from config)
            collection_name: Name of the vector collection (default: tickets)
            dimensions: Embedding dimensions (default: 1536 from model)
            skip_init: If True, skip collection creation (for testing)
        """
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import VectorParams, Distance
        except ImportError:
            raise ImportError(
                "qdrant-client is required. Install with: pip install qdrant-client"
            )

        self.config = Config()
        qdrant_config = self.config.get_qdrant_config()

        self.host = host or qdrant_config.get("host", "localhost")
        self.port = port or qdrant_config.get("port", 6333)
        self.collection_name = collection_name or qdrant_config.get("collection_name", "tickets")
        self.dimensions = dimensions or get_embedding_model_dimensions()

        # Initialize Qdrant client
        self.client = QdrantClient(host=self.host, port=self.port)

        # Create collection if it doesn't exist (unless skip_init is True)
        if not skip_init:
            if not self.collection_exists():
                self._create_collection()

    def _create_collection(self):
        """Create a new vector collection in Qdrant."""
        from qdrant_client.models import VectorParams, Distance

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.dimensions,
                distance=Distance.COSINE
            )
        )
        print(f"Created Qdrant collection: {self.collection_name}")

    def collection_exists(self) -> bool:
        """Check if the collection exists."""
        try:
            self.client.get_collection(self.collection_name)
            return True
        except Exception:
            return False

    def upsert_points(self, points: List[Dict], batch_size: int = 100) -> None:
        """Upsert (insert/update) vectors into the store.

        Args:
            points: List of points, each with:
                - id: Unique identifier
                - vector: Embedding vector
                - payload: Metadata dictionary
            batch_size: Number of points to upsert in each batch
        """
        from qdrant_client.models import PointStruct

        if not points:
            print("No points to upsert")
            return

        print(f"Upserting {len(points)} points to Qdrant...")

        for i in tqdm(range(0, len(points), batch_size), desc="Upserting batches"):
            batch = points[i:i + batch_size]

            # Build payload: use existing payload if present, otherwise extract from chunk fields
            qdrant_points = [
                PointStruct(
                    id=point.get("id") or str(uuid.uuid4()),
                    vector=point.get("vector") or point.get("embedding", []),
                    payload=point.get("payload", {
                        "ticket_id": point.get("ticket_id", ""),
                        "chunk_index": point.get("chunk_index", 0),
                        "chunk_text": point.get("chunk_text", "")
                    })
                )
                for point in batch
            ]

            self.client.upsert(
                collection_name=self.collection_name,
                points=qdrant_points
            )

        print(f"Successfully upserted {len(points)} points")
        return len(points)

    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filter_conditions: Optional[List] = None,
        with_payload: bool = True,
        with_vector: bool = False,
        score_threshold: Optional[float] = None
    ) -> List[Dict]:
        """Search for top-k most similar vectors.

        Args:
            query_vector: The embedding vector to search with
            top_k: Number of results to return
            filter_conditions: Optional list of Qdrant Filter conditions
            with_payload: Whether to include metadata in results
            with_vector: Whether to include the vector in results
            score_threshold: Optional minimum score threshold for results

        Returns:
            List of results with id, score, and payload (if requested)
        """
        from qdrant_client.models import Filter

        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=Filter(must=filter_conditions) if filter_conditions else None,
            limit=top_k,
            with_payload=with_payload,
            score_threshold=score_threshold
        )

        # query_points returns QueryResponse object with points attribute
        points_list = search_result.points if hasattr(search_result, 'points') else search_result
        return [
            {
                "id": point.id,
                "score": point.score,
                "payload": point.payload if with_payload else None
            }
            for point in points_list
        ]

    def filtered_search(
        self,
        query_vector: List[float],
        service: Optional[str] = None,
        severity: Optional[str] = None,
        ticket_type: Optional[str] = None,
        top_k: int = 10
    ) -> List[Dict]:
        """Search with common metadata filters.

        Args:
            query_vector: The embedding vector to search with
            service: Filter by service name
            severity: Filter by severity level
            ticket_type: Filter by ticket type
            top_k: Number of results to return

        Returns:
            List of filtered search results
        """
        from qdrant_client.models import FieldCondition, MatchValue

        conditions = []

        if service:
            conditions.append(
                FieldCondition(
                    key="service",
                    match=MatchValue(value=service)
                )
            )

        if severity:
            conditions.append(
                FieldCondition(
                    key="severity",
                    match=MatchValue(value=severity)
                )
            )

        if ticket_type:
            conditions.append(
                FieldCondition(
                    key="ticket_type",
                    match=MatchValue(value=ticket_type)
                )
            )

        return self.search(query_vector, top_k=top_k, filter_conditions=conditions)

    def get_all_points(self) -> List[Dict]:
        """Retrieve all points from the collection.
        
        Returns:
            List of all points with id, score, and payload
        """
        points = []
        offset = None
        
        while True:
            result = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                offset=offset,
                with_payload=True
            )
            batch, offset = result
            
            for point in batch:
                points.append({
                    "id": point.id,
                    "score": 0.0,  # Score is not available from scroll
                    "payload": point.payload
                })
            
            if offset is None:
                break
        
        return points

    def get_point(self, point_id: str) -> Optional[Dict]:
        """Retrieve a single point by ID.

        Args:
            point_id: The ID of the point to retrieve

        Returns:
            Point with id, vector, and payload, or None if not found
        """
        try:
            point = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id],
                with_payload=True,
                with_vector=True
            )
            if point:
                return {
                    "id": point[0].id,
                    "vector": point[0].vector,
                    "payload": point[0].payload
                }
            return None
        except Exception:
            return None

    def delete_point(self, point_id: str) -> bool:
        """Delete a point by ID.

        Args:
            point_id: The ID of the point to delete

        Returns:
            True if deletion was successful
        """
        from qdrant_client.models import PointId

        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=[PointId(id=point_id)]
            )
            return True
        except Exception:
            return False

    def count_points(self) -> int:
        """Count total number of points in the collection."""
        return self.client.count(
            collection_name=self.collection_name,
            exact=True
        ).count

    def health_check(self) -> bool:
        """Check if Qdrant is reachable."""
        try:
            self.client.get_collection(self.collection_name)
            return True
        except Exception:
            return False



    def wipe(self) -> Dict[str, Any]:
        """Delete all points from the collection.
        
        This clears all data while preserving the collection schema.
        Returns statistics about the operation.
        """
        try:
            from qdrant_client.models import PointIdsList
            
            # Get current count before wipe
            before_count = self.count_points()
            
            # Delete all points if any exist
            if before_count > 0:
                batch_size = 1000
                deleted_count = 0
                
                while True:
                    # Scroll through all points and collect IDs
                    points, next_page = self.client.scroll(
                        collection_name=self.collection_name,
                        limit=batch_size
                    )
                    
                    if not points:
                        break
                    
                    point_ids = [p.id for p in points]
                    
                    # Delete these points using the delete method
                    self.client.delete(
                        collection_name=self.collection_name,
                        points_selector=point_ids,
                        wait=True
                    )
                    
                    deleted_count += len(point_ids)
                    
                    if next_page is None:
                        break
                
                print(f"Deleted {deleted_count} points from collection '{self.collection_name}'")
            
            after_count = self.count_points()
            
            return {
                "success": True,
                "before_count": before_count,
                "deleted_count": before_count - after_count,
                "after_count": after_count,
                "collection": self.collection_name
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "collection": self.collection_name
            }

def get_embedding_model_dimensions() -> int:
    """Get embedding dimensions based on model configuration."""
    config = Config()
    model = config.get("openai", "embedding_model", default="text-embedding-3-small")

    # OpenAI embedding models dimensions
    dimensions_map = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    # Check if explicit dimensions are configured
    explicit_dims = config.get("openai", "embedding_dimensions", default=None)
    if explicit_dims is not None:
        return explicit_dims

    return dimensions_map.get(model, 4096)


class StorePipeline:
    """Pipeline to process and store embeddings in Qdrant.

    Integrates Stage 2 (Embed) with Stage 3 (Store).
    """

    def __init__(self, vector_store: Optional[VectorStore] = None, skip_init: bool = False):
        """Initialize the storage pipeline.

        Args:
            vector_store: Optional VectorStore instance (creates default if not provided)
            skip_init: If True, skip VectorStore initialization (for testing)
        """
        self.vector_store = vector_store or VectorStore(skip_init=skip_init)

    def store_chunks(
        self,
        chunks: List[Dict],
        batch_size: int = 100,
        id_prefix: str = "chunk"
    ) -> Dict:
        """Process and store embedded chunks in Qdrant.

        Args:
            chunks: List of chunks with 'embedding' field
            batch_size: Number of points to store per batch
            id_prefix: Prefix for auto-generated IDs

        Returns:
            Dict with storage statistics
        """
        if not chunks:
            return {"stored": 0, "failed": 0, "skipped": 0}

        # Extract embedding dimension to validate
        sample_embedding = chunks[0].get("embedding", [])
        if sample_embedding:
            actual_dims = len(sample_embedding)
            expected_dims = self.vector_store.dimensions
            if actual_dims != expected_dims:
                print(f"Warning: Embedding dimensions mismatch. "
                      f"Expected {expected_dims}, got {actual_dims}")

        # Build points for upsert
        points = []
        for i, chunk in enumerate(chunks):
            point_id = chunk.get("chunk_id") or chunk.get("id")
            if not point_id:
                # Generate a proper UUID v4 for Qdrant
                point_id = str(uuid.uuid4())
            
            point = {
                "id": point_id,
                "vector": chunk.get("embedding", []),
                "payload": {
                    "ticket_id": chunk.get("ticket_id", ""),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "chunk_text": chunk.get("chunk_text", ""),
                    "service": chunk.get("metadata", {}).get("service", ""),
                    "severity": chunk.get("metadata", {}).get("severity", ""),
                    "ticket_type": chunk.get("metadata", {}).get("ticket_type", ""),
                    "created_at": chunk.get("metadata", {}).get("created_at", ""),
                    "user_name": chunk.get("metadata", {}).get("user_name", ""),
                    "user_email": chunk.get("metadata", {}).get("user_email", "")
                }
            }
            points.append(point)

        # Upsert to Qdrant with error handling
        try:
            self.vector_store.upsert_points(points, batch_size=batch_size)
            stored_count = len(points)
        except Exception as e:
            # Handle dimension mismatch or other Qdrant errors
            if "dimension" in str(e).lower() or "dim" in str(e).lower():
                print(f"Dimension mismatch error: {e}")
                return {"stored": 0, "failed": len(points), "skipped": 0}
            raise

        return {
            "stored": stored_count,
            "failed": 0,
            "skipped": 0,
            "collection": self.vector_store.collection_name,
            "total_points": self.vector_store.count_points()
        }


def create_vector_store(**kwargs) -> VectorStore:
    """Factory function to create a VectorStore instance."""
    return VectorStore(**kwargs)


def create_store_pipeline(**kwargs) -> StorePipeline:
    """Factory function to create a StorePipeline instance.

    Args:
        **kwargs: Arguments for VectorStore creation or pre-created vector_store

    Returns:
        StorePipeline instance
    """
    skip_init = kwargs.get("skip_init", False)
    
    # If vector_store is provided, use it; otherwise create a new one
    if "vector_store" in kwargs:
        vector_store = kwargs["vector_store"]
    else:
        vector_store = VectorStore(**{k: v for k, v in kwargs.items() if k != "skip_init"})
    
    return StorePipeline(vector_store=vector_store, skip_init=skip_init)

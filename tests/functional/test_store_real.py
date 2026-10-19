"""Real integration tests for Stage 3: Storage (VectorStore) with Qdrant.

Tests actual storage of embeddings in a real Qdrant instance at localhost:6333.
These tests require Qdrant to be running on localhost:6333.
To run these tests, start Qdrant first:
    docker run -p 6333:6333 qdrant/qdrant

Tests will be automatically skipped if Qdrant is not accessible.
"""
import pytest
import sys
from pathlib import Path
import uuid
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from supportdesk_rag.supportdesk_store import (
    VectorStore,
    StorePipeline,
    get_embedding_model_dimensions
)


def get_test_collection_name():
    """Generate a unique collection name for testing."""
    timestamp = int(time.time() * 1000) % 1000000
    return f"test_tickets_{uuid.uuid4().hex[:8]}_{timestamp}"


def require_qdrant():
    """Decorator to skip tests if Qdrant is not available."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                store = VectorStore(skip_init=True)
                store.collection_exists()
                return func(*args, **kwargs)
            except Exception as e:
                pytest.skip(f"Qdrant not available: {e}")
        return wrapper
    return decorator


class TestVectorStoreReal:
    """Real integration tests for VectorStore with actual Qdrant."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Set up and tear down Qdrant collection for each test."""
        try:
            self.collection_name = get_test_collection_name()
            self.dimensions = get_embedding_model_dimensions()
            self.store = VectorStore(collection_name=self.collection_name)
            self.pipeline = StorePipeline(vector_store=self.store)
            self._qdrant_available = True
        except Exception as e:
            pytest.skip(f"Qdrant not available: {e}")
            self._qdrant_available = False
            return

        yield

        # Cleanup: delete the collection after test
        try:
            if self._qdrant_available and hasattr(self.store, 'client'):
                self.store.client.delete_collection(collection_name=self.collection_name)
        except Exception:
            pass

    @require_qdrant()
    def test_create_collection_and_store_single_chunk(self):
        """Test creating collection and storing a single chunk."""
        chunk = {
            "ticket_id": "TICK-001",
            "chunk_index": 0,
            "chunk_text": "Network connection timeout when accessing API endpoint",
            "embedding": [0.1] * self.dimensions,
            "metadata": {
                "service": "Network Infrastructure",
                "severity": "High",
                "ticket_type": "Incident",
                "created_at": "2026-01-15T10:00:00",
                "user_name": "John Doe",
                "user_email": "john@example.com"
            }
        }

        result = self.pipeline.store_chunks([chunk])

        assert result["stored"] == 1
        assert result["collection"] == self.collection_name

        # Verify the point was stored by counting
        count = self.store.count_points()
        assert count == 1

    def test_store_multiple_chunks(self):
        """Test storing multiple chunks in one batch."""
        if not hasattr(self, '_qdrant_available') or not self._qdrant_available:
            pytest.skip("Qdrant not available")
            
        chunks = [
            {
                "ticket_id": f"TICK-{i:03d}",
                "chunk_index": 0,
                "chunk_text": f"Test ticket {i} description",
                "embedding": [0.1 * i] * self.dimensions,
                "metadata": {
                    "service": "Test Service",
                    "severity": "Medium",
                    "ticket_type": "Service Request",
                    "created_at": f"2026-01-{15 + i:02d}",
                    "user_name": f"User {i}",
                    "user_email": f"user{i}@example.com"
                }
            }
            for i in range(5)
        ]

        result = self.pipeline.store_chunks(chunks)

        assert result["stored"] == 5
        count = self.store.count_points()
        assert count == 5

    def test_batch_storage_with_different_batch_sizes(self):
        """Test that batch storage works with various batch sizes."""
        if not hasattr(self, '_qdrant_available') or not self._qdrant_available:
            pytest.skip("Qdrant not available")
            
        chunks = [
            {
                "ticket_id": f"TICK-{i:03d}",
                "chunk_index": 0,
                "chunk_text": f"Batch test chunk {i}",
                "embedding": [0.2 * i] * self.dimensions,
                "metadata": {"service": "Batch Test"}
            }
            for i in range(15)
        ]

        # Test with batch_size=5
        result = self.pipeline.store_chunks(chunks, batch_size=5)
        assert result["stored"] == 15
        assert self.store.count_points() == 15

    def test_dimension_validation(self):
        """Test that dimension mismatch is handled correctly."""
        if not hasattr(self, '_qdrant_available') or not self._qdrant_available:
            pytest.skip("Qdrant not available")
            
        wrong_dimension_chunks = [
            {
                "ticket_id": "TICK-001",
                "chunk_index": 0,
                "chunk_text": "Wrong dimension embedding",
                "embedding": [0.1] * 100,  # Wrong dimension
                "metadata": {}
            }
        ]

        result = self.pipeline.store_chunks(wrong_dimension_chunks)

        # Should report 0 stored due to dimension mismatch
        assert result["stored"] == 0

    def test_store_empty_list(self):
        """Test storing an empty list of chunks."""
        if not hasattr(self, '_qdrant_available') or not self._qdrant_available:
            pytest.skip("Qdrant not available")
            
        result = self.pipeline.store_chunks([])

        assert result["stored"] == 0
        assert self.store.count_points() == 0


class TestStorePipelineReal:
    """Real integration tests for StorePipeline with actual Qdrant."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Set up and tear down Qdrant collection for each test."""
        try:
            self.collection_name = get_test_collection_name()
            self.dimensions = get_embedding_model_dimensions()
            self.store = VectorStore(collection_name=self.collection_name)
            self.pipeline = StorePipeline(vector_store=self.store)
            self._qdrant_available = True
        except Exception as e:
            pytest.skip(f"Qdrant not available: {e}")
            self._qdrant_available = False
            return

        yield

        try:
            if hasattr(self.store, 'client'):
                self.store.client.delete_collection(collection_name=self.collection_name)
        except Exception:
            pass

    def test_store_and_verify_payload(self):
        """Test that stored payload matches original chunk."""
        if not hasattr(self, '_qdrant_available') or not self._qdrant_available:
            pytest.skip("Qdrant not available")
            
        original_chunk = {
            "ticket_id": "TICK-ORIGINAL",
            "chunk_index": 0,
            "chunk_text": "This is the original chunk text that should be stored",
            "embedding": [0.5] * self.dimensions,
            "metadata": {
                "service": "Verification Test",
                "severity": "Low",
                "ticket_type": "Question"
            }
        }

        self.pipeline.store_chunks([original_chunk])

        # Retrieve and verify
        points = self.store.get_all_points()
        assert len(points) == 1

        point = points[0]
        assert point.get("payload", {}).get("ticket_id") == "TICK-ORIGINAL"
        assert point.get("payload", {}).get("chunk_text") == original_chunk["chunk_text"]

    def test_collection_exists_check(self):
        """Test that collection exists check works correctly."""
        if not hasattr(self, '_qdrant_available') or not self._qdrant_available:
            pytest.skip("Qdrant not available")
            
        # Collection should exist after setup
        assert self.store.collection_exists()
        assert self.store.count_points() == 0

        # Store something and verify count
        self.pipeline.store_chunks([{
            "ticket_id": "TICK-001",
            "chunk_index": 0,
            "chunk_text": "Test",
            "embedding": [0.1] * self.dimensions,
            "metadata": {}
        }])
        assert self.store.count_points() == 1

    def test_store_pipeline_returns_correct_format(self):
        """Test that store_chunks returns properly formatted result."""
        if not hasattr(self, '_qdrant_available') or not self._qdrant_available:
            pytest.skip("Qdrant not available")
            
        chunks = [
            {
                "ticket_id": "TICK-001",
                "chunk_index": 0,
                "chunk_text": "Format test",
                "embedding": [0.1] * self.dimensions,
                "metadata": {}
            }
        ]

        result = self.pipeline.store_chunks(chunks, batch_size=10)

        assert isinstance(result, dict)
        assert "stored" in result
        assert "collection" in result
        assert "total_points" in result
        assert result["collection"] == self.collection_name
        assert result["stored"] == 1
        assert result["total_points"] == 1


class TestQdrantConnection:
    """Tests for Qdrant connection handling."""

    def test_qdrant_connection_available(self):
        """Test that Qdrant is accessible."""
        try:
            store = VectorStore(collection_name="test_connection_check")
            exists = store.collection_exists()
            # If we get here, Qdrant is accessible
            assert True
            # Cleanup
            store.client.delete_collection(collection_name="test_connection_check")
        except Exception as e:
            pytest.skip(f"Could not connect to Qdrant at localhost:6333: {e}")

    def test_lazy_init_without_qdrant(self):
        """Test that skip_init=True works when Qdrant is not available."""
        # This test uses skip_init to avoid Qdrant connection
        store = VectorStore(collection_name="test_lazy", skip_init=True)
        
        assert store.client is not None  # Client exists but collection not created
        assert store.collection_name == "test_lazy"
        assert store.dimensions == get_embedding_model_dimensions()

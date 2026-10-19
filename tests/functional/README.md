# Real Integration Tests for Stage 3: Storage

This directory contains real integration tests that connect to a live Qdrant instance.

## Prerequisites

These tests require a running Qdrant instance at `localhost:6333`.

### Starting Qdrant with Docker

```bash
docker run -p 6333:6333 qdrant/qdrant
```

Or run Qdrant locally:
```bash
# Download Qdrant from https://github.com/qdrant/qdrant
# Run:
qdrant
```

## Test Files

- `test_store_real.py` - Tests for storing and managing vectors
- `test_search_real.py` - Tests for semantic search and filtering
- `test_pipeline_real.py` - End-to-end pipeline tests

## Running Tests

### With Qdrant Available
```bash
pytest tests/functional/ -v --tb=short
```

### With Qdrant Available (only qdrant tests)
```bash
pytest tests/functional/ -v --tb=short -m qdrant_test
```

### Skip Qdrant Tests (if not available)
```bash
pytest tests/functional/ -v --tb=short --deselect tests/functional/test_store_real.py::TestQdrantConnection
```

## Test Categories

### Store Tests (`test_store_real.py`)
- `TestVectorStoreReal` - Basic storage operations
  - `test_create_collection_and_store_single_chunk` - Store single vector
  - `test_store_multiple_chunks` - Batch storage
  - `test_batch_storage_with_different_batch_sizes` - Different batch sizes
  - `test_dimension_validation` - Validate embedding dimensions
  - `test_store_empty_list` - Handle empty input

- `TestStorePipelineReal` - StorePipeline operations
  - `test_store_and_verify_payload` - Verify stored data
  - `test_collection_exists_check` - Collection management
  - `test_store_pipeline_returns_correct_format` - API contract

### Search Tests (`test_search_real.py`)
- `TestSemanticSearchReal` - Basic search functionality
  - `test_semantic_search_with_related_query` - Similar query search
  - `test_search_with_metadata_filter` - Filtered search
  - `test_search_returns_top_k_results` - Top-k precision
  - `test_search_with_zero_results` - Edge cases

- `TestSearchWithRealEmbeddings` - Real embedding workflow
  - `test_search_with_real_embeddings_workflow` - Full workflow

### Pipeline Tests (`test_pipeline_real.py`)
- `TestFullPipelineIntegration` - End-to-end pipeline
  - `test_full_pipeline_ingest_embed_store` - Full pipeline run
  - `test_search_after_pipeline_storage` - Search after storage
  - `test_pipeline_with_multiple_severities` - Complex filtering

## Test Notes

- Tests use unique collection names (with UUID + timestamp) to avoid conflicts
- Collections are automatically deleted after each test
- Tests will be skipped automatically if Qdrant is not available

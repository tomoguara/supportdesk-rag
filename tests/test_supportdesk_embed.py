"""
Unit tests for Stage 2: Embedding
"""
import pytest
import os
import json
from pathlib import Path
from unittest.mock import Mock, patch
import sys
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from supportdesk_rag.supportdesk_embed import (
    Config,
    EmbeddingService,
    EmbeddingPipeline,
    create_embedding_pipeline,
)


class TestConfig:
    """Tests for Config class"""
    
    def test_config_loads_from_yaml(self):
        """Test that config loads from YAML file"""
        config = Config("config/config.yaml")
        assert config is not None
        assert config.get("openai", "embedding_model") == "text-embedding-3-small"
        assert config.get("ingest", "chunk_size") == 1000
    
    def test_config_get_with_defaults(self):
        """Test that config returns defaults when keys don't exist"""
        config = Config("config/config.yaml")
        assert config.get("nonexistent", default="default_value") == "default_value"
    
    def test_config_get_nested(self):
        """Test nested config access"""
        config = Config("config/config.yaml")
        openai_config = config.get_openai_config()
        assert openai_config is not None
        assert "api_key_env" in openai_config


class TestEmbeddingService:
    """Tests for EmbeddingService class"""
    
    def test_init_with_config_defaults(self):
        """Test initialization uses config defaults"""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "USE_OLLAMA_MODELS": "false"}):
            service = EmbeddingService()
            config = Config()
            assert service.batch_size == config.get_embed_config().get("batch_size", 100)
    
    def test_init_with_override(self):
        """Test initialization with custom values"""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "USE_OLLAMA_MODELS": "false"}):
            service = EmbeddingService(model="text-embedding-3-large", batch_size=50)
            assert service.model == "text-embedding-3-large"
            assert service.batch_size == 50
    
    def test_embed_text_success(self):
        """Test embedding a single text"""
        mock_data = Mock()
        mock_data.embedding = [0.1] * 1536
        mock_response = Mock()
        mock_response.data = [mock_data]
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "USE_OLLAMA_MODELS": "false"}):
            service = EmbeddingService()
            service.client = Mock()
            service.client.embeddings.create.return_value = mock_response
            
            embedding = service.embed_text("test text")
            assert len(embedding) == 1536
            assert all(isinstance(v, float) for v in embedding)
    
    def test_embed_text_failure_raises_error(self):
        """Test that embedding failure raises RuntimeError"""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "USE_OLLAMA_MODELS": "false"}):
            service = EmbeddingService()
            service.client = Mock()
            service.client.embeddings.create.side_effect = Exception("API Error")
            
            with pytest.raises(RuntimeError, match="Embedding failed"):
                service.embed_text("test text")
    
    def test_embed_batch(self):
        """Test embedding a batch of texts"""
        mock_data1 = Mock()
        mock_data1.embedding = [0.1] * 1536
        mock_data2 = Mock()
        mock_data2.embedding = [0.2] * 1536
        mock_response = Mock()
        mock_response.data = [mock_data1, mock_data2]
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "USE_OLLAMA_MODELS": "false"}):
            service = EmbeddingService(batch_size=100)
            service.client = Mock()
            service.client.embeddings.create.return_value = mock_response
            
            embeddings = service.embed_batch(["text1", "text2"])
            assert len(embeddings) == 2
            assert len(embeddings[0]) == 1536


class TestEmbeddingPipeline:
    """Tests for EmbeddingPipeline class"""
    
    def test_init_with_service(self):
        """Test pipeline initialization with a service"""
        mock_service = Mock(spec=EmbeddingService)
        pipeline = EmbeddingPipeline(mock_service)
        assert pipeline.embedding_service == mock_service
    
    def test_process_chunks(self):
        """Test processing chunks"""
        mock_service = Mock(spec=EmbeddingService)
        mock_service.embed_batch.return_value = [[0.1] * 1536, [0.2] * 1536]
        mock_service.model = "text-embedding-3-small"
        mock_service.dimensions = 1536
        pipeline = EmbeddingPipeline(mock_service, batch_size=100)
        
        chunks = [
            {"chunk_text": "Test chunk 1", "metadata": {"service": "Network"}},
            {"chunk_text": "Test chunk 2", "metadata": {"service": "Database"}}
        ]
        
        result = pipeline.process_chunks(chunks)
        
        assert len(result) == 2
        assert "embedding" in result[0]
        assert len(result[0]["embedding"]) == 1536
    
    def test_process_and_save(self):
        """Test that process_and_save creates output file"""
        mock_service = Mock(spec=EmbeddingService)
        mock_service.embed_batch.return_value = [[0.1] * 1536, [0.2] * 1536]
        mock_service.model = "text-embedding-3-small"
        mock_service.dimensions = 1536
        pipeline = EmbeddingPipeline(mock_service, batch_size=100)
        
        chunks = [
            {"chunk_text": "Test chunk 1", "metadata": {"service": "Network"}},
            {"chunk_text": "Test chunk 2", "metadata": {"service": "Database"}}
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "embedded", "chunks.json")
            result = pipeline.process_and_save(chunks, output_path)
            
            assert os.path.exists(output_path)
            
            with open(output_path) as f:
                data = json.load(f)
            
            assert "chunks" in data
            assert "metadata" in data
            assert data["metadata"]["total_chunks"] == 2
            assert len(result) == 2


class TestCreateEmbeddingPipeline:
    """Tests for create_embedding_pipeline factory function"""
    
    def test_create_pipeline_default(self):
        """Test creating pipeline with default settings"""
        pipeline = create_embedding_pipeline()
        assert isinstance(pipeline.embedding_service, EmbeddingService)
        assert pipeline.batch_size == 100
    
    def test_create_pipeline_custom_model(self):
        """Test creating pipeline with custom model"""
        with patch.dict(os.environ, {"USE_OLLAMA_MODELS": "false"}):
            pipeline = create_embedding_pipeline(model="text-embedding-3-large", batch_size=50)
            assert pipeline.embedding_service.model == "text-embedding-3-large"
            assert pipeline.batch_size == 50


class TestIntegrationWithIngest:
    """Integration tests between Stage 1 (Ingest) and Stage 2 (Embed)"""
    
    def test_ingest_then_embed_workflow(self):
        """Test the complete ingest -> embed workflow"""
        from supportdesk_rag.supportdesk_ingest import SupportDeskIngest
        
        test_tickets = {
            "tickets": [{
                "ticket_id": "TICK-001",
                "description": "This is a test ticket description.",
                "service": "Network Infrastructure",
                "severity": "High",
                "ticket_type": "Incident",
                "created_at": "2026-01-15T10:00:00",
                "user": {"name": "Test User", "email": "test@example.com"}
            }]
        }
        
        with open("data/test_tickets.json", "w") as f:
            json.dump(test_tickets, f)
        
        try:
            ingest = SupportDeskIngest(chunk_size=100, chunk_overlap=10)
            chunks = ingest.get_chunks("data/test_tickets.json", clean=True)
            
            assert len(chunks) >= 1
            assert "chunk_text" in chunks[0]
            assert "metadata" in chunks[0]
            
            mock_service = Mock(spec=EmbeddingService)
            mock_service.embed_batch.return_value = [[0.1] * 1536 for _ in chunks]
            
            pipeline = EmbeddingPipeline(mock_service, batch_size=100)
            embedded_chunks = pipeline.process_chunks(chunks)
            
            assert len(embedded_chunks) == len(chunks)
            assert "embedding" in embedded_chunks[0]
            
        finally:
            if os.path.exists("data/test_tickets.json"):
                os.remove("data/test_tickets.json")

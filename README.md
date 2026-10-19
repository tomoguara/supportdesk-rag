# SupportDesk RAG

A Production-Ready RAG (Retrieval-Augmented Generation) system for support desk tickets. Built with LangChain, OpenAI, and Qdrant, it helps answer customer support queries with grounded, cited answers from your ticket history.

---

**✨ Features**
- **7-Stage RAG Pipeline**: Ingest → Embed → Store → Retrieve → Re-rank → Ground → Generate
- **Hybrid Search**: Combines vector similarity with BM25 keyword matching
- **Citation-Verified Answers**: Every claim is traced back to source tickets
- **Smart Triage**: Automatically categorize and prioritize incoming tickets
- **Bulk Analysis**: Process and group multiple tickets
- **Guardrails**: Citation enforcement, relevance checking, PII redaction
- **Observability**: Full tracing and metrics collection
- **Evaluation Framework**: Measure answer quality, hallucination rates, and relevance

---

## Table of Contents

- [Part 1: User Guide - How to Use the System](#part-1-user-guide-how-to-use-the-system)
  - [Before You Start](#11-before-you-start)
  - [CLI Commands](#12-cli-commands)
  - [Examples](#13-examples)
- [Part 2: Developer Guide](#part-2-developer-guide)
  - [Setting Up Your Environment](#21-setting-up-your-environment)
  - [Architecture Overview](#22-architecture-overview)
  - [Module Reference](#23-module-reference)
  - [Testing](#24-testing)
  - [Model Configuration](#25-model-configuration)

---

# Part 1 - User Guide: How to Use the System

## 1.1 Before You Start

You need:

- **Python 3.14+** (use `pyenv` to install)
- `uv` for package management
- The project set up as described in [Setting up your environment](#21-setting-up-your-environment)
- Required API keys in a file named `.env` in the project folder

### Environment Setup

```bash
# Copy environment file and edit with your keys
cp .env.example .env
nano .env  # Add your OpenAI or Ollama configuration
```

Required environment variables:

```bash
# OpenAI Configuration
OPENAI_API_KEY=sk-...

# Or use Ollama (local models)
# USE_OLLAMA_MODELS=true
# OLLAMA_BASE_URL=http://localhost:11434/v1
# OLLAMA_EMBEDDING_MODEL=qwen3-embedding:latest
```

### Required Services

```bash
# Start Qdrant vector database (in another terminal)
./scripts/qdrant-db.sh --start

# Check status
./scripts/qdrant-db.sh --status
```

---

## 1.2 CLI Commands

The `sd-rag` CLI provides various commands for working with support tickets:

### Basic Usage

```bash
# Show all available commands
sd-rag --help

# Show help for a specific command
sd-rag answer --help
```

### Available Commands

| Command | Description |
|---------|-------------|
| `answer` | Answer support questions with citations |
| `evaluate` | Evaluate RAG responses with metrics against ground truth |
| `triage` | Automatically categorize and prioritize incoming tickets |
| `similar` | Find similar past tickets to the current issue |
| `escalate` | Generate escalation handoff summaries |
| `runbook` | Find runbook procedures for incidents |
| `gap` | Detect knowledge gaps in support tickets |
| `bulk` | Process and analyze multiple tickets at once |
| `ingest` | Load and index ticket data into the vector store |
| `qdb` | Query Qdrant database directly (debugging) |

### Command Details

#### `answer` - Answer Support Questions

```bash
# Basic usage
sd-rag answer --prompt "How do I reset my password?"

# With more results
sd-rag answer --prompt "Network timeout issues" --top-k 10

# JSON output for automation
sd-rag answer --prompt "Database connection failed" --json

# Verbose mode
sd-rag answer -p "Server downtime" -v
```

**Output**: Answer text with citations from source tickets, performance metrics.

#### `evaluate` - Evaluate RAG Responses

```bash
# Basic evaluation
sd-rag evaluate --prompt "What is Python?"

# With ground truth for comparison
sd-rag evaluate \
  --prompt "What is Python?" \
  --ground-truth "Python is a programming language created by Guido van Rossum"
```

**Output**: Relevance score, answer quality, hallucination detection, context precision/recall, latency metrics.

#### `triage` - Ticket Triage and Categorization

```bash
# Triage a new ticket
sd-rag triage --prompt "Customer reports latency issues with API"

# Get JSON output for integration
sd-rag triage -p "Memory leak in production service" --json
```

**Output**:
- Summary of the issue
- Category: password/billing/technical/account/feature-request/other
- Urgency: high/medium/low
- Troubleshooting steps for the support agent

#### `similar` - Find Similar Cases

```bash
# Find similar cases to current issue
sd-rag similar --prompt "Database connection timeout"

# Find cases similar to a specific ticket
sd-rag similar --prompt "API rate limiting" --ticket-id INC-001

# Get more results
sd-rag similar --prompt "Authentication failure" --top-k 20
```

#### `escalate` - Generate Escalation Summaries

```bash
# Create escalation handoff
sd-rag escalate \
  --prompt "Server is down and cannot be restarted" \
  --context "Already tried basic troubleshooting, reboot, and logs check"

# With more context
sd-rag escalate \
  --prompt "Payment processing failed" \
  --context "User reported error after card update, tried 3 times"
```

**Output**: Issue summary, what's been tried, relevant logs, next team, action items.

#### `runbook` - Runbook Assistance

```bash
# Get runbook for incident
sd-rag runbook --prompt "Memory leak in production microservice"

# Find procedure for database issue
sd-rag runbook -p "Slow query performance on users table"
```

**Output**: Identify incident category, relevant runbook procedure with steps, critical warnings.

#### `gap` - Knowledge Gap Detection

```bash
# Analyze pattern for gaps
sd-rag gap --prompt "Frequent password reset issues"

# Analyze specific tickets
sd-rag gap \
  --prompt "Missing documentation for two-factor authentication" \
  --ticket-ids INC-001 INC-002 INC-003
```

**Output**: Missing information, documentation needs, top priorities.

#### `bulk` - Bulk Ticket Analysis

```bash
# Process multiple tickets
sd-rag bulk --file-path tickets.json

# Different analysis types
sd-rag bulk \
  --file-path tickets.json \
  --analysis-type group_by_symptom

# Group by product
sd-rag bulk -f tickets.json -a group_by_product
```

**Input format** (JSON):
```json
{
  "tickets": [
    {"prompt": "Issue 1 description"},
    {"prompt": "Issue 2 description"},
    {"description": "Issue 3 description"}
  ]
}
```

#### `ingest` - Index Ticket Data

```bash
# Ingest tickets from JSON file
sd-rag ingest --data-path data/tickets.json

# Don't clean PII (passwords, emails)
sd-rag ingest -d data/tickets.json --no-clean

# Wipe and re-ingest
sd-rag ingest -d data/tickets.json --wipe
```

#### `qdb` - Query Qdrant Database

```bash
# Show collection information
sd-rag qdb --info

# List ticket IDs (default 20)
sd-rag qdb --list
sd-rag qdb --list 50

# Search by ticket ID
sd-rag qdb --ticket-id LAT001234

# Search by text similarity
sd-rag qdb --search "network timeout"

# Filter by metadata
sd-rag qdb --filter --service network --severity high
```

---

## 1.3 Examples

### Typical User Workflows

#### Workflow 1: Answering a Customer Question

```bash
# Step 1: Answer with citations
sd-rag answer \
  --prompt "How do I update my billing information?" \
  --top-k 5

# Step 2: Verify the answer quality
sd-rag evaluate \
  --prompt "How do I update my billing information?" \
  --ground-truth "Users can update billing info in Settings > Billing"
```

#### Workflow 2: Triage Incoming Tickets

```bash
# Process incoming tickets into a file
cat incoming_tickets.json | jq -c '.' | while read ticket; do
  sd-rag triage -p "$ticket" --json >> triaged.json
done
```

#### Workflow 3: Knowledge Gap Analysis

```bash
# Daily gap analysis
sd-rag gap \
  --prompt "Daily pattern analysis: what are users struggling with?"
```

#### Workflow 4: Bulk Processing End of Day

```bash
# Batch process all tickets
sd-rag bulk \
  --file-path /data/daily_tickets.json \
  --analysis-type group_by_product
```

### Common CLI Options

| Option | Description |
|--------|-------------|
| `--prompt` / `-p` | User's question or ticket description |
| `--json` | Output in JSON format (for automation) |
| `--verbose` / `-v` | Enable verbose output |
| `--debug` | Enable debug mode with stack trace |
| `--top-k N` | Number of results to retrieve (default: 5) |
| `--config PATH` | Override config file path |
| `--timeout SECS` | Request timeout in seconds |

---

# Part 2 - Developer Guide

## 2.1 Setting Up Your Environment

### Prerequisites

- **Python 3.14+** (use `pyenv` to install)
- `uv` for package management
- `docker` and `docker-compose` for local services (optional)

### Installation Steps

```bash
# 1. Clone the repository
cd /home/txo/codeio/my-public-ai-apps/supportdesk-rag

# 2. Set up environment variables
cp .env.example .env
# Edit .env with your API keys
nano .env

# 3. Install dependencies
uv sync

# 4. Verify installation
uv run python -c "import supportdesk_rag; print('Import successful')"
```

### Starting Local Services

#### Qdrant (Vector Database)

```bash
# Start with helper script
./scripts/qdrant-db.sh --start

# Or use Docker directly
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant

# Check status
./scripts/qdrant-db.sh --status

# Stop when done
./scripts/qdrant-db.sh --stop
```

#### Ollama (Local Embeddings - Optional)

```bash
# Install Ollama if not already installed
# https://ollama.ai/download

# Pull required models
ollama pull qwen3-embedding:latest
ollama pull llama3

# Start Ollama
ollama serve
```

---

## 2.2 Architecture Overview

![SupportDesk RAG Full System Architecture](docs/images/full_system_architecture_readme.png)

*Click image to view full-size* (PlantUML source: [`full_system_architecture_readme.puml`](docs/images/full_system_architecture_readme.puml))

### Data Flow

| Stage | Name | Description |
|-------|------|-------------|
| 1 | Ingest | Load tickets from JSON/CSV, clean PII, chunk text |
| 2 | Embed | Convert chunks to vectors using OpenAI or Ollama |
| 3 | Store | Persist vectors in Qdrant with metadata |
| 4 | Retrieve | Hybrid search (vector + BM25) with RRF fusion |
| 5 | Re-rank | Cross-encoder for precise relevance scoring |
| 6 | Ground | Assemble context with citation markers |
| 7 | Generate | LLM produces grounded answer |
| 8 | Guard | Verify citations, check relevance, redact PII |

---

## 2.3 Module Reference

### Core Pipeline Modules

| Module | Description | Main Classes |
|--------|-------------|--------------|
| `supportdesk_ingest.py` | Ticket ingestion, chunking, deduplication | `SupportDeskIngest` |
| `supportdesk_embed.py` | Embedding generation (OpenAI/Ollama) | `EmbeddingService`, `EmbeddingPipeline` |
| `supportdesk_store.py` | Vector storage in Qdrant | `VectorStore`, `StorePipeline` |
| `supportdesk_retrieve.py` | Hybrid search (vector + BM25) | `HybridRetriever`, `BM25Index` |
| `supportdesk_rerank.py` | Cross-encoder re-ranking | `ReRanker`, `ReRankerPipeline` |
| `supportdesk_ground.py` | Context assembly with citations | `ContextGrounding`, `GroundedContext` |
| `supportdesk_generate.py` | LLM answer generation | `AnswerGenerator`, `GeneratedAnswer` |

### Guard and Observability Modules

| Module | Description | Main Classes |
|--------|-------------|--------------|
| `supportdesk_guard.py` | Citation verification, PII redaction | `Guardrails` |
| `supportdesk_observability.py` | Tracing and metrics collection | `RAGPipelineObserver`, `MetricsCollector` |
| `supportdesk_eval.py` | Evaluation metrics and framework | `Evaluator`, `EvaluationPipeline` |

### CLI Command Modules

| Command | File | Purpose |
|---------|------|---------|
| `answer` | `rag_commands/rag_answer.py` | Full RAG pipeline for questions |
| `triage` | `rag_commands/rag_triage.py` | Ticket categorization |
| `similar` | `rag_commands/rag_similar.py` | Find similar cases |
| `escalate` | `rag_commands/rag_escalate.py` | Escalation handoffs |
| `runbook` | `rag_commands/rag_runbook.py` | Runbook assistance |
| `gap` | `rag_commands/rag_gap.py` | Knowledge gap detection |
| `bulk` | `rag_commands/rag_bulk.py` | Batch processing |
| `ingest` | `rag_commands/rag_ingest.py` | Data ingestion |
| `qdb` | `rag_commands/rag_qdb.py` | Qdrant debugging |

### Main Pipeline Class

**File**: `src/supportdesk_rag/supportdesk_rag_pipeline.py`

The `SupportDeskRAGPipeline` class integrates all stages:

```python
from supportdesk_rag import SupportDeskRAGPipeline

# Initialize pipeline
pipeline = SupportDeskRAGPipeline()

# Run full RAG pipeline
result = pipeline.generate_answer(
    query_text="How do I reset my password?",
    top_k=5,
    rerank_enabled=True,
    ground_enabled=True
)

# Access results
print(result['answer'].answer)  # Answer text
print(result['answer'].citations)  # Source citations
print(result['grounded_context'])  # Used context
```

---

## 2.4 Testing

### Running Tests

```bash
# Run all tests
uv run pytest

# Unit tests only (no external dependencies)
uv run pytest tests/unit/

# Integration tests with real services
uv run pytest tests/integration/

# Functional tests (with Qdrant)
uv run pytest tests/functional/

# Specific test file
uv run pytest tests/test_supportdesk_ingest.py -v

# Verbose output
uv run pytest tests/ -v -s

# Stop at first failure
uv run pytest -x
```

### Test Structure

```
tests/
├── unit/                    # Unit tests (mocked dependencies)
│   ├── test_ingest.py
│   ├── test_embed.py
│   └── ...
├── integration/             # Integration tests (real APIs)
│   ├── test_ingest.py
│   └── ...
├── functional/              # End-to-end tests
│   ├── test_store_real.py
│   ├── test_search_real.py
│   └── test_pipeline_real.py
└── test_data/               # Test fixtures and data
    └── synthetic_tickets.json
```

### Running Integration Tests

Some tests require Qdrant and Ollama running:

```bash
# Start services (in separate terminals)
./scripts/qdrant-db.sh --start
ollama serve

# Run functional tests
uv run pytest tests/functional/ -v

# Run specific test
uv run pytest tests/functional/test_pipeline_real.py -v -s
```

---

## 2.5 Model Configuration

### Environment Variables

The system supports multiple model providers based on `USE_OLLAMA_MODELS`:

| Variable | Ollama Mode | OpenAI Mode | Default |
|----------|-------------|-------------|---------|
| `USE_OLLAMA_MODELS` | `true` | `false` | `true` |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | N/A | - |
| `OLLAMA_EMBEDDING_MODEL` | `qwen3-embedding:latest` | N/A | - |
| `OLLAMA_CHAT_MODEL` | `llama3` | N/A | - |
| `OPENAI_API_KEY` | N/A | Required | - |
| `OPENAI_EMBEDDING_MODEL` | N/A | `text-embedding-3-small` | - |
| `OPENAI_CHAT_MODEL` | N/A | `gpt-4o-mini` | - |

### Default Configuration

| Component | Ollama Model | OpenAI Model |
|-----------|-------------|--------------|
| **Embeddings** | `qwen3-embedding:latest` | `text-embedding-3-small` |
| **LLM** | `llama3` | `gpt-4o-mini` |
| **Dimensions** | 4096 | 1536 |

### Using OpenAI in Production

```bash
# Edit .env
cat > .env << EOF
USE_OLLAMA_MODELS=false
OPENAI_API_KEY=sk-your-key-here
# Optional: specify models
# OPENAI_EMBEDDING_MODEL=text-embedding-3-small
# OPENAI_CHAT_MODEL=gpt-4o-mini
EOF

# Test
uv run python -c "from supportdesk_rag import print_model_config; print_model_config()"
```

---

## 2.6 Common Extensions

### Adding a New CLI Command

1. Create `src/supportdesk_rag/rag_commands/rag_<command>.py`
2. Implement `Rag<Command>Command` class with `run()` and `format_output()`
3. Register in `sd-rag` main function

### Adding a New Retriever

1. Implement in `src/supportdesk_rag/supportdesk_retrieve.py`
2. Register in `__init__.py`
3. Update `SupportDeskRAGPipeline` to use your retriever

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

*This project is developed and maintained by Ton Oguara.*

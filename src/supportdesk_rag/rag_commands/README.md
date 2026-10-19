# SupportDesk RAG CLI Commands

This directory contains the CLI command modules for the SupportDesk RAG system.

## Available Commands

### `answer` - Answer Questions
Answers support questions using the full RAG pipeline with citations.

```bash
python -m supportdesk_rag.rag_commands.rag_answer --prompt "How do I reset my password?"
python -m supportdesk_rag.rag_commands.rag_answer --prompt "What is Python?" --json
python -m supportdesk_rag.rag_commands.rag_answer --prompt "How to contact support?" --top-k 10
```

### `triage` - Triage Tickets
Triage support tickets by suggesting category, priority, and assigned team.

```bash
python -m supportdesk_rag.rag_commands.rag_triage --prompt "Customer reports latency issues"
python -m supportdesk_rag.rag_commands.rag_triage --prompt "API down for all users" --json
```

### `similar` - Find Similar Cases
Find similar support cases for a given issue.

```bash
python -m supportdesk_rag.rag_commands.rag_similar --prompt "Database connection timeout"
python -m supportdesk_rag.rag_commands.rag_similar --prompt "Password reset failed" --ticket-id INC-001
```

### `ingest` - Ingest Data
Ingest data (tickets, runbooks) into the RAG system.

```bash
python -m supportdesk_rag.rag_commands.rag_ingest --data-path data/tickets.json
python -m supportdesk_rag.rag_commands.rag_ingest --data-path data/runbooks.csv --no-clean
```

### `evaluate` - Evaluate Responses
Evaluate RAG responses with metrics (ground truth comparison).

```bash
python -m supportdesk_rag.rag_commands.rag_evaluate --prompt "What is Python?" --ground-truth "Python is a programming language"
python -m supportdesk_rag.rag_commands.rag_evaluate --prompt "How to contact support?" --json
```

### `escalate` - Escalation Handoffs
Generate concise summaries for ticket escalations including what's been tried and next steps.

```bash
python -m supportdesk_rag.rag_commands.rag_escalate --prompt "Server down" --context "Already restarted once"
python -m supportdesk_rag.rag_commands.rag_escalate --prompt "Memory leak in production" --context "Tried multiple restarts"
```

### `runbook` - Runbook Assistance
Find the right procedure for an incident and provide step-by-step guidance.

```bash
python -m supportdesk_rag.rag_commands.rag_runbook --prompt "Memory leak in production"
python -m supportdesk_rag.rag_commands.rag_runbook --prompt "Database connection timeout"
```

### `gap` - Knowledge Gap Detection
Review tickets to identify missing or outdated documentation.

```bash
python -m supportdesk_rag.rag_commands.rag_gap --prompt "Frequent password reset issues"
python -m supportdesk_rag.rag_commands.rag_gap --prompt "Common authentication issues" --ticket-ids INC-001 INC-002
```

### `bulk` - Bulk Ticket Analysis
Process multiple tickets from a file and group them by patterns.

```bash
python -m supportdesk_rag.rag_commands.rag_bulk --file-path tickets.json
python -m supportdesk_rag.rag_commands.rag_bulk --file-path tickets.json --analysis-type group_by_symptom
```

## Common Options

All commands support these options:

- `--json` - Output in JSON format
- `--verbose, -v` - Enable verbose output
- `--debug` - Enable debug mode (implies verbose)
- `--top-k N` - Number of results to retrieve (default: 5)
- `--config PATH` - Override config file path

## JSON Output Format

All commands support `--json` flag for machine-readable output. The output includes:
- Prompt/query
- Response/answer
- Citations with confidence scores
- Performance metrics

## Exit Codes

- `0` - Success
- `1` - Error (command failed or no matching results found)

## Usage with Main CLI Entry Point

```bash
# Using the main CLI entry point
sd-rag answer --prompt "How do I reset my password?"
sd-rag triage --prompt "Customer reports latency issues"
sd-rag ingest --data-path data/tickets.json
```

See `sd-rag --help` for more information.

#!/usr/bin/env python3
"""SupportDesk RAG CLI - Main entry point.

This is the main CLI entry point for the SupportDesk RAG system.
It provides sub-commands for various RAG operations.

Usage:
    sd-rag answer --prompt "Your question here"
    sd-rag evaluate --prompt "Your question here" --ground-truth "Expected answer"
    sd-rag triage --prompt "Ticket description"
    sd-rag similar --prompt "Current issue" --ticket-id REF-123
    sd-rag escalate --prompt "Issue summary" --context "What's been tried"
    sd-rag runbook --prompt "Incident description"
    sd-rag gap --prompt "Pattern analysis"
    sd-rag bulk --file-path tickets.json
    sd-rag ingest --data-path data.json
    sd-rag qdb --list
    sd-rag qdb --ticket-id TICKET-123

Advanced options:
    --config PATH    Override config file path
    --json           Output in JSON format
    --verbose/-v     Enable verbose output
    --debug          Enable debug mode (implies verbose)
    --timeout SECS   Request timeout in seconds
    --top-k N        Number of results to retrieve (default: 5)
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import Optional

# For development, allow running directly from src directory
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).parent))
    __package__ = "supportdesk_rag"

from .rag_commands import (
    RagAnswerCommand,
    RagTriageCommand,
    RagSimilarCommand,
    RagIngestCommand,
    RagEvaluateCommand,
    RagBulkCommand,
    RagEscalateCommand,
    RagRunbookCommand,
    RagGapCommand,
)
from .supportdesk_rag_pipeline import SupportDeskRAGPipeline


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog="sd-rag",
        description="SupportDesk RAG CLI - Answer questions, triage tickets, and more",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  sd-rag answer --prompt "How do I reset my password?"
  sd-rag evaluate --prompt "What is Python?" --ground-truth "Python is a language"
  sd-rag triage --prompt "Customer reports latency issues"
  sd-rag similar --prompt "Database connection timeout" --ticket-id INC-001
  sd-rag escalate --prompt "Server down" --context "Already restarted once"
  sd-rag runbook --prompt "Memory leak in production"
  sd-rag gap --prompt "Frequent password reset issues"
  sd-rag bulk --file-path /path/to/tickets.json
  sd-rag ingest --data-path /path/to/data.json
  sd-rag qdb --list
  sd-rag qdb --list 50
  sd-rag qdb --ticket-id LAT001234
  
Advanced options:
  --config PATH    Override config file path
  --json           Output in JSON format
  --verbose/-v     Enable verbose output
  --debug          Enable debug mode (implies verbose)
  --timeout SECS   Request timeout in seconds
  --top-k N        Number of results to retrieve
        """
    )
    
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
        metavar="COMMAND"
    )
    
    # answer command
    answer_parser = subparsers.add_parser(
        "answer",
        help="Answer support questions with citations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: sd-rag answer --prompt 'How do I reset my password?'"
    )
    answer_parser.add_argument("--prompt", "-p", required=True, help="User's question")
    answer_parser.add_argument("--config", type=str, default=None, help="Config file path")
    answer_parser.add_argument("--top-k", type=int, default=5, help="Results to retrieve")
    answer_parser.add_argument("--json", action="store_true", help="Output as JSON")
    answer_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    answer_parser.add_argument("--debug", action="store_true", help="Debug mode")
    answer_parser.add_argument("--timeout", type=int, default=None, help="Request timeout")
    
    # evaluate command
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate RAG responses with metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: sd-rag evaluate --prompt 'What is Python?' --ground-truth 'Python is a language'"
    )
    eval_parser.add_argument("--prompt", "-p", required=True, help="Question to evaluate")
    eval_parser.add_argument("--ground-truth", "-g", type=str, default=None, help="Expected answer")
    eval_parser.add_argument("--config", type=str, default=None, help="Config file path")
    eval_parser.add_argument("--top-k", type=int, default=5, help="Results to retrieve")
    eval_parser.add_argument("--json", action="store_true", help="Output as JSON")
    eval_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    eval_parser.add_argument("--debug", action="store_true", help="Debug mode")
    
    # triage command
    triage_parser = subparsers.add_parser(
        "triage",
        help="Triage support tickets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: sd-rag triage --prompt 'Customer reports latency issues'"
    )
    triage_parser.add_argument("--prompt", "-p", required=True, help="Ticket description")
    triage_parser.add_argument("--config", type=str, default=None, help="Config file path")
    triage_parser.add_argument("--top-k", type=int, default=5, help="Similar cases to retrieve")
    triage_parser.add_argument("--json", action="store_true", help="Output as JSON")
    triage_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    triage_parser.add_argument("--debug", action="store_true", help="Debug mode")
    
    # similar command
    similar_parser = subparsers.add_parser(
        "similar",
        help="Find similar support cases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: sd-rag similar --prompt 'Database connection timeout' --ticket-id INC-001"
    )
    similar_parser.add_argument("--prompt", "-p", required=True, help="Current issue description")
    similar_parser.add_argument("--ticket-id", "-t", type=str, default=None, help="Ticket ID to compare")
    similar_parser.add_argument("--config", type=str, default=None, help="Config file path")
    similar_parser.add_argument("--top-k", type=int, default=10, help="Similar cases to retrieve")
    similar_parser.add_argument("--json", action="store_true", help="Output as JSON")
    similar_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    similar_parser.add_argument("--debug", action="store_true", help="Debug mode")
    
    # ingest command
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest data into the RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: sd-rag ingest --data-path data.json"
    )
    ingest_parser.add_argument("--data-path", "-d", required=True, help="Path to data file")
    ingest_parser.add_argument("--config", type=str, default=None, help="Config file path")
    ingest_parser.add_argument("--no-clean", action="store_false", dest="clean", default=True, help="Don't clean PII")
    ingest_parser.add_argument("--wipe", action="store_true", default=False, help="Wipe collection before ingesting")
    ingest_parser.add_argument("--no-wipe", action="store_false", dest="wipe", default=False, help="Keep existing data (default)")
    ingest_parser.add_argument("--json", action="store_true", help="Output as JSON")
    ingest_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    ingest_parser.add_argument("--debug", action="store_true", help="Debug mode")
    
    # escalate command
    escalate_parser = subparsers.add_parser(
        "escalate",
        help="Generate escalation handoff summaries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: sd-rag escalate --prompt 'Server down' --context 'Already restarted'"
    )
    escalate_parser.add_argument("--prompt", "-p", required=True, help="Issue summary")
    escalate_parser.add_argument("--context", "-c", type=str, default=None, help="What's been tried")
    escalate_parser.add_argument("--config", type=str, default=None, help="Config file path")
    escalate_parser.add_argument("--top-k", type=int, default=5, help="Results to retrieve")
    escalate_parser.add_argument("--json", action="store_true", help="Output as JSON")
    escalate_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    escalate_parser.add_argument("--debug", action="store_true", help="Debug mode")
    
    # runbook command
    runbook_parser = subparsers.add_parser(
        "runbook",
        help="Get runbook assistance for incidents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: sd-rag runbook --prompt 'Memory leak in production'"
    )
    runbook_parser.add_argument("--prompt", "-p", required=True, help="Incident description")
    runbook_parser.add_argument("--config", type=str, default=None, help="Config file path")
    runbook_parser.add_argument("--top-k", type=int, default=5, help="Runbooks to retrieve")
    runbook_parser.add_argument("--json", action="store_true", help="Output as JSON")
    runbook_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    runbook_parser.add_argument("--debug", action="store_true", help="Debug mode")
    
    # gap command
    gap_parser = subparsers.add_parser(
        "gap",
        help="Detect knowledge gaps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: sd-rag gap --prompt 'Frequent password reset issues'"
    )
    gap_parser.add_argument("--prompt", "-p", required=True, help="Pattern analysis")
    gap_parser.add_argument("--ticket-ids", "-t", type=str, nargs='+', default=None, help="Ticket IDs to analyze")
    gap_parser.add_argument("--config", type=str, default=None, help="Config file path")
    gap_parser.add_argument("--top-k", type=int, default=5, help="Results to retrieve")
    gap_parser.add_argument("--json", action="store_true", help="Output as JSON")
    gap_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    gap_parser.add_argument("--debug", action="store_true", help="Debug mode")
    
    # bulk command
    bulk_parser = subparsers.add_parser(
        "bulk",
        help="Bulk ticket analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: sd-rag bulk --file-path tickets.json"
    )
    bulk_parser.add_argument("--file-path", "-f", required=True, help="Path to tickets JSON file")
    bulk_parser.add_argument("--analysis-type", "-a", type=str, default="group_by_product",
                            choices=["group_by_product", "group_by_symptom", "summarize"],
                            help="Type of analysis")
    bulk_parser.add_argument("--config", type=str, default=None, help="Config file path")
    bulk_parser.add_argument("--top-k", type=int, default=5, help="Results per ticket")
    bulk_parser.add_argument("--json", action="store_true", help="Output as JSON")
    bulk_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    bulk_parser.add_argument("--debug", action="store_true", help="Debug mode")

    # qdb command
    qdb_parser = subparsers.add_parser(
        "qdb",
        help="Query Qdrant database directly for debugging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  sd-rag qdb --info
  sd-rag qdb --list
  sd-rag qdb --list 50
  sd-rag qdb --ticket-id LAT001234
  sd-rag qdb --search "latency issues"
  sd-rag qdb --filter --service "Database Services"
        """
    )
    qdb_parser.add_argument("--info", action="store_true", help="Show collection information")
    qdb_parser.add_argument("--list", "-l", nargs='?', const=20, type=int, help="List ticket IDs (default: 20, or specify number)")
    qdb_parser.add_argument("--ticket-id", "-t", type=str, help="Search for specific ticket ID")
    qdb_parser.add_argument("--search", "-s", type=str, help="Search by text similarity")
    qdb_parser.add_argument("--filter", "-f", action="store_true", help="Filter by metadata")
    qdb_parser.add_argument("--service", type=str, help="Filter by service (use with --filter)")
    qdb_parser.add_argument("--severity", type=str, help="Filter by severity (use with --filter)")
    qdb_parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of results for search")
    qdb_parser.add_argument("--json", action="store_true", help="Output in JSON")
    qdb_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    qdb_parser.add_argument("--debug", action="store_true", help="Debug mode")

    # Parse arguments
    args = parser.parse_args()
    
    # If no command provided, show help
    if args.command is None:
        # Check if --prompt was provided without subcommand
        if hasattr(args, 'prompt') and args.prompt:
            # Run full pipeline with prompt
            run_full_pipeline(args)
            return
        print(parser.format_help())
        sys.exit(0)
    
    # Run the appropriate command
    command_map = {
        "answer": ("rag_answer", "RagAnswerCommand"),
        "evaluate": ("rag_evaluate", "RagEvaluateCommand"),
        "triage": ("rag_triage", "RagTriageCommand"),
        "similar": ("rag_similar", "RagSimilarCommand"),
        "ingest": ("rag_ingest", "RagIngestCommand"),
        "escalate": ("rag_escalate", "RagEscalateCommand"),
        "runbook": ("rag_runbook", "RagRunbookCommand"),
        "gap": ("rag_gap", "RagGapCommand"),
        "bulk": ("rag_bulk", "RagBulkCommand"),
        "qdb": ("rag_qdb", "RagQdbCommand"),
    }
    
    if args.command not in command_map:
        parser.print_help()
        sys.exit(1)
    
    module_name, class_name = command_map[args.command]
    
    try:
        # Import the command module
        command_module = __import__(
            f"supportdesk_rag.rag_commands.{module_name}",
            fromlist=[class_name]
        )
        command_class = getattr(command_module, class_name)
        
        # Get command-specific arguments (map config to config_path, exclude json_output)
        command_args = {}
        for k, v in vars(args).items():
            if k in ['command', 'debug', 'verbose']:
                continue
            # Map 'config' to 'config_path' for command modules
            if k == 'config':
                command_args['config_path'] = v
            # Skip json/json_output as it's handled by format_output
            elif k in ['json', 'json_output']:
                continue
            # Skip timeout since run() doesn't accept it
            elif k == 'timeout':
                continue
            # Handle list parameter - if it's True (default), use 20, otherwise use the value
            elif k == 'list' and v is True:
                command_args['list'] = 20
            else:
                command_args[k] = v
        
        # Initialize and run command
        command = command_class(
            verbose=args.verbose or args.debug,
            debug=args.debug
        )
        
        # Run the command
        result = command.run(**command_args)
        
        # Format and print output
        output_format = "json" if args.json else "markdown"
        output = command.format_output(result, output_format)
        print(output)
        
        # Exit with appropriate code
        if "error" in result:
            sys.exit(1)
        sys.exit(0)
        
    except ImportError as e:
        print(f"[ERROR] Failed to import command module: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        if args.debug:
            import traceback
            traceback.print_exc()
        else:
            print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


def run_full_pipeline(args):
    """Run the full pipeline when --prompt is provided without subcommand."""
    from . import SupportDeskRAGPipeline
    
    print(f"[INFO] Running full RAG pipeline for: {args.prompt}")
    
    pipeline = SupportDeskRAGPipeline()
    
    try:
        result = pipeline.generate_answer(
            query_text=args.prompt,
            top_k=args.top_k if hasattr(args, 'top_k') else 5,
            rerank_enabled=True,
            ground_enabled=True
        )
        
        # Format output
        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            # Markdown format
            print("# Support Desk RAG Response")
            print("")
            print(f"**Question:** {args.prompt}")
            print("")
            
            answer_obj = result.get("answer")
            if answer_obj:
                if isinstance(answer_obj, dict):
                    answer = answer_obj.get("answer", "")
                else:
                    answer = getattr(answer_obj, "answer", "")
                print("## Answer")
                print("")
                print(answer)
                print("")
            
            # Citations
            grounded_context = result.get("grounded_context")
            if grounded_context:
                citations = getattr(grounded_context, "citations", [])
                if citations:
                    print("## Citations")
                    print("")
                    for citation in citations:
                        ticket_id = citation.get('ticket_id', 'N/A')
                        score = citation.get('score', 'N/A')
                        retriever = citation.get('retriever', 'N/A')
                        print(f"- [{ticket_id}] (Score: {score:.3f}, Retriever: {retriever})")
                    print("")
        
        sys.exit(0)
        
    except Exception as e:
        if args.debug:
            import traceback
            traceback.print_exc()
        else:
            print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

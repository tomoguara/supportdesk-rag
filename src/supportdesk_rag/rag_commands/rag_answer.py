"""RAG Answer Command - Answer support questions with citations.

This module implements the answer command for the SupportDesk RAG CLI.
It runs the full pipeline: search → re-rank → ground → generate.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class RagAnswerCommand:
    """Command to answer support questions using the full RAG pipeline."""

    def __init__(self, verbose: bool = False, debug: bool = False, timeout: Optional[int] = None):
        """Initialize the answer command.
        
        Args:
            verbose: Enable verbose output
            debug: Enable debug mode (implies verbose)
            timeout: Request timeout in seconds
        """
        self.verbose = verbose or debug
        self.debug = debug
        self.timeout = timeout
        
    def run(self, prompt: str, config_path: Optional[str] = None, 
            top_k: int = 5) -> Dict[str, Any]:
        """Run the answer command.
        
        Args:
            prompt: User's question
            config_path: Optional path to config file
            top_k: Number of results to retrieve
            
        Returns:
            Dictionary with results
        """
        from supportdesk_rag import SupportDeskRAGPipeline
        
        if self.verbose:
            print("[INFO] Initializing RAG Pipeline...")
            
        # Initialize pipeline
        pipeline = SupportDeskRAGPipeline()
        
        if config_path and os.path.exists(config_path):
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            # Apply any config overrides here if needed
            if self.verbose:
                print(f"[INFO] Loaded config from {config_path}")
        
        if self.verbose:
            print("[INFO] Running full RAG pipeline...")
            
        # Run the full pipeline
        try:
            result = pipeline.generate_answer(
                query_text=prompt,
                top_k=top_k,
                rerank_enabled=True,
                ground_enabled=True
            )
            
            # Extract answer
            answer_obj = result.get("answer")
            answer_text = ""
            citations = []
            
            if answer_obj:
                if isinstance(answer_obj, dict):
                    answer_text = answer_obj.get("answer", "")
                    citations = answer_obj.get("citations", [])
                else:
                    # GeneratedAnswer dataclass
                    answer_text = getattr(answer_obj, "answer", "")
                    citations = getattr(answer_obj, "citations", [])
            
            # Extract grounded context for citations
            grounded_context = result.get("grounded_context")
            if grounded_context:
                citations = getattr(grounded_context, "citations", citations)
            
            # Get observability metrics
            metrics = pipeline.get_observability_metrics()
            
            output = {
                "prompt": prompt,
                "answer": answer_text,
                "citations": citations,
                "metrics": metrics,
                            }
            
            return output
            
        except Exception as e:
            if self.debug:
                raise
            return {
                "error": str(e),
                "prompt": prompt,
                "answer": "I encountered an error generating your answer. Please try again.",
                "citations": [],
                "metrics": {},
                            }

    def format_output(self, result: Dict[str, Any], output_format: str = "markdown") -> str:
        """Format the output for display.
        
        Args:
            result: The result dictionary from run()
            
        Returns:
            Formatted string output
        """
        if output_format == "json":
            return json.dumps(result, indent=2, default=str)
        
        # Markdown format
        lines = []
        
        # Check for error case
        if "error" in result:
            lines.append("# ❌ Error")
            lines.append("")
            lines.append(f"```text\n{result['error']}\n```")
            lines.append("")
            if "prompt" in result:
                lines.append(f"**Original Question:** {result['prompt']}")
                lines.append("")
            return "\n".join(lines)
        
        lines.append("# Support Desk RAG Response")
        lines.append("")
        lines.append(f"**Question:** {result.get('prompt', 'N/A')}")
        lines.append("")
        
        answer = result.get('answer', '')
        if answer:
            lines.append("## Answer")
            lines.append("")
            lines.append(answer)
            lines.append("")
        
        citations = result.get('citations', [])
        if citations:
            lines.append("## Citations")
            lines.append("")
            for i, citation in enumerate(citations, 1):
                ticket_id = citation.get('ticket_id', f'passage_{i}')
                score = citation.get('score', 'N/A')
                retriever = citation.get('retriever', 'N/A')
                lines.append(f"### [{citation.get('citation_id', ticket_id)}]")
                lines.append(f"- Ticket ID: {ticket_id}")
                lines.append(f"- Relevance Score: {score}")
                lines.append(f"- Retriever: {retriever}")
                lines.append("")
        
        metrics = result.get('metrics', {})
        if metrics:
            lines.append("## Performance Metrics")
            lines.append("")
            if isinstance(metrics, dict):
                for stage, stage_metrics in metrics.items():
                    if isinstance(stage_metrics, dict):
                        lines.append(f"### {stage.replace('_', ' ').title()}")
                        for key, value in stage_metrics.items():
                            lines.append(f"- {key}: {value}")
                        lines.append("")
        
        return "\n".join(lines)


def main():
    """Main entry point for the answer command."""
    parser = argparse.ArgumentParser(
        description="Answer support questions using the RAG pipeline"
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        required=True,
        help="User's question to answer"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to retrieve (default: 5)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output in JSON format"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (implies verbose)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Request timeout in seconds"
    )
    
    args = parser.parse_args()
    
    # Initialize command
    command = RagAnswerCommand(
        verbose=args.verbose,
        debug=args.debug,
        timeout=args.timeout
    )
    
    # Run the command
    result = command.run(
        prompt=args.prompt,
        config_path=args.config,
        top_k=args.top_k
    )
    
    # Format and print output
    output_format = "json" if args.json_output else "markdown"
    output = command.format_output(result, output_format)
    print(output)
    
    # Exit with appropriate code
    if "error" in result:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()

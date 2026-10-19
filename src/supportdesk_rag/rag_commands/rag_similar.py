"""RAG Similar Command - Find similar cases.

This module implements the similar command for the SupportDesk RAG CLI.
It retrieves past tickets with similar symptoms to the current issue.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class RagSimilarCommand:
    """Command to find similar cases."""

    def __init__(self, verbose: bool = False, debug: bool = False):
        self.verbose = verbose or debug
        self.debug = debug

    def run(self, prompt: str, ticket_id: Optional[str] = None,
            config_path: Optional[str] = None, top_k: int = 10) -> Dict[str, Any]:
        """Run the similar cases command.
        
        Args:
            prompt: Description of the current issue
            ticket_id: Optional ticket ID to find similar cases to
            config_path: Optional path to config file
            top_k: Number of similar cases to retrieve
            
        Returns:
            Dictionary with similar cases
        """
        from supportdesk_rag import SupportDeskRAGPipeline
        
        if self.verbose:
            print("[INFO] Finding similar cases...")
        
        pipeline = SupportDeskRAGPipeline()
        
        # If ticket_id is provided, search by that, otherwise use prompt
        query_text = prompt
        if ticket_id:
            query_text = f"Find tickets similar to ticket ID: {ticket_id}. Description: {prompt}"
        
        try:
            # Use generate_answer to get search results with citations
            result = pipeline.generate_answer(
                query_text=query_text,
                top_k=top_k,
                rerank_enabled=False,  # Skip re-ranking for faster similar case lookup
                ground_enabled=False   # Don't need grounded context for search
            )
            
            # Extract search results (citations)
            hybrid_results = []
            answer_obj = result.get('answer')
            if answer_obj:
                if isinstance(answer_obj, dict):
                    hybrid_results = answer_obj.get('citations', [])
                else:
                    hybrid_results = getattr(answer_obj, 'citations', [])
            
            output = {
                "prompt": prompt,
                "ticket_id": ticket_id,
                "similar_cases": hybrid_results,
                "metrics": pipeline.get_observability_metrics()
            }
            
            return output
            
        except Exception as e:
            if self.debug:
                raise
            return {
                "error": str(e),
                "prompt": prompt,
                "similar_cases": []
            }

    def format_output(self, result: Dict[str, Any], output_format: str = "markdown") -> str:
        """Format the output for display."""
        if output_format == "json":
            return json.dumps(result, indent=2, default=str)
        
        # Markdown format
        lines = []
        lines.append("# Similar Cases")
        lines.append("")
        
        if "error" in result:
            lines.append("## ❌ Error")
            lines.append("")
            lines.append(f"```text\n{result['error']}\n```")
            return "\n".join(lines)
        
        lines.append(f"**Query:** {result.get('prompt', 'N/A')[:200]}...")
        if result.get('ticket_id'):
            lines.append(f"**Base Ticket ID:** {result['ticket_id']}")
        lines.append("")
        
        similar_cases = result.get('similar_cases', [])
        if similar_cases:
            lines.append("## 📋 Found Cases")
            lines.append("")
            
            for i, case in enumerate(similar_cases, 1):
                ticket_id = case.get('ticket_id', case.get('id', f'case_{i}'))
                score = case.get('score', case.get('fused_score', 0))
                
                lines.append(f"### Case {i}: [{ticket_id}]")
                lines.append("")
                lines.append(f"- **Relevance Score:** {score:.3f}")
                lines.append("")
        else:
            lines.append("## No similar cases found")
            lines.append("")
        
        return "\n".join(lines)


def main():
    """Main entry point for the similar command."""
    parser = argparse.ArgumentParser(
        description="Find similar support cases"
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        required=True,
        help="Description of the current issue"
    )
    parser.add_argument(
        "--ticket-id", "-t",
        type=str,
        default=None,
        help="Ticket ID to find similar cases to"
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
        default=10,
        help="Number of similar cases to retrieve (default: 10)"
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
    
    args = parser.parse_args()
    
    command = RagSimilarCommand(
        verbose=args.verbose,
        debug=args.debug
    )
    
    result = command.run(
        prompt=args.prompt,
        ticket_id=args.ticket_id,
        config_path=args.config,
        top_k=args.top_k
    )
    
    output_format = "json" if args.json_output else "markdown"
    output = command.format_output(result, output_format)
    print(output)
    
    if "error" in result:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()

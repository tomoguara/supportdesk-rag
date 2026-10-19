"""RAG Gap Command - Knowledge gap detection.

This module implements the gap command for the SupportDesk RAG CLI.
It reviews tickets to identify missing or outdated documentation.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class RagGapCommand:
    """Command to detect knowledge gaps."""

    def __init__(self, verbose: bool = False, debug: bool = False):
        self.verbose = verbose or debug
        self.debug = debug

    def run(self, prompt: str, ticket_ids: Optional[List[str]] = None,
            config_path: Optional[str] = None, top_k: int = 5) -> Dict[str, Any]:
        """Run the gap detection command.
        
        Args:
            prompt: Description of the issue pattern or missing documentation
            ticket_ids: Optional list of ticket IDs to analyze
            config_path: Optional path to config file
            top_k: Number of results to retrieve
            
        Returns:
            Dictionary with gap analysis results
        """
        from supportdesk_rag import SupportDeskRAGPipeline
        
        if self.verbose:
            print("[INFO] Detecting knowledge gaps...")
        
        pipeline = SupportDeskRAGPipeline()
        
        # Gap detection prompt template
        gap_prompt = f"""You are a knowledge base analyst. Analyze the following support ticket information and identify knowledge gaps.

1. What information is missing from the knowledge base?
2. What documentation should be created or updated?
3. What common patterns emerge in these tickets?
4. What are the top 3 priorities for documentation updates?

Issue/Prompt:
{prompt}

Ticket IDs (if provided): {', '.join(ticket_ids) if ticket_ids else 'Not specified'}

Format your response with clear sections and actionable recommendations."""

        try:
            result = pipeline.generate_answer(
                query_text=gap_prompt,
                top_k=top_k,
                rerank_enabled=True,
                ground_enabled=True
            )
            
            answer_obj = result.get("answer")
            answer_text = ""
            
            if answer_obj:
                if isinstance(answer_obj, dict):
                    answer_text = answer_obj.get("answer", "")
                else:
                    answer_text = getattr(answer_obj, "answer", "")
            
            # Extract citations
            grounded_context = result.get("grounded_context")
            citations = []
            if grounded_context:
                citations = getattr(grounded_context, "citations", [])
            
            output = {
                "prompt": prompt,
                "ticket_ids": ticket_ids,
                "gap_analysis": answer_text,
                "citations": citations,
                "metrics": pipeline.get_observability_metrics()
            }
            
            return output
            
        except Exception as e:
            if self.debug:
                raise
            return {
                "error": str(e),
                "prompt": prompt,
                "gap_analysis": "Unable to detect knowledge gaps.",
                "citations": []
            }

    def format_output(self, result: Dict[str, Any], output_format: str = "markdown") -> str:
        """Format the output for display."""
        if output_format == "json":
            return json.dumps(result, indent=2, default=str)
        
        # Markdown format
        lines = []
        lines.append("# ⚠️ Knowledge Gap Analysis")
        lines.append("")
        
        if "error" in result:
            lines.append("## ❌ Error")
            lines.append("")
            lines.append(f"```text\n{result['error']}\n```")
            return "\n".join(lines)
        
        lines.append(f"**Analysis Prompt:** {result.get('prompt', 'N/A')[:200]}...")
        if result.get('ticket_ids'):
            lines.append(f"**Tickets Analyzed:** {', '.join(result['ticket_ids'])}")
        lines.append("")
        
        gap_analysis = result.get('gap_analysis', '')
        if gap_analysis:
            lines.append("## 📊 Gap Analysis Results")
            lines.append("")
            lines.append(gap_analysis)
            lines.append("")
        
        citations = result.get('citations', [])
        if citations:
            lines.append("## 📚 Citations (Source Tickets)")
            lines.append("")
            for citation in citations:
                ticket_id = citation.get('ticket_id', 'N/A')
                score = citation.get('score', 'N/A')
                lines.append(f"- [{ticket_id}] (Score: {score:.3f})")
            lines.append("")
        
        return "\n".join(lines)


def main():
    """Main entry point for the gap command."""
    parser = argparse.ArgumentParser(
        description="Detect knowledge gaps in support tickets"
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        required=True,
        help="Description of the issue pattern or missing documentation"
    )
    parser.add_argument(
        "--ticket-ids", "-t",
        type=str,
        default=None,
        nargs='+',
        help="List of ticket IDs to analyze"
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
    
    args = parser.parse_args()
    
    command = RagGapCommand(
        verbose=args.verbose,
        debug=args.debug
    )
    
    result = command.run(
        prompt=args.prompt,
        ticket_ids=args.ticket_ids,
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

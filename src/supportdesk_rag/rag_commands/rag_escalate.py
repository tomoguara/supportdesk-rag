"""RAG Escalate Command - Escalation handoffs.

This module implements the escalate command for the SupportDesk RAG CLI.
It generates concise summaries for ticket escalations including what's been tried and next steps.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class RagEscalateCommand:
    """Command to generate escalation handoff summaries."""

    def __init__(self, verbose: bool = False, debug: bool = False):
        self.verbose = verbose or debug
        self.debug = debug

    def run(self, prompt: str, ticket_context: Optional[str] = None,
            config_path: Optional[str] = None, top_k: int = 5) -> Dict[str, Any]:
        """Run the escalation command.
        
        Args:
            prompt: Ticket description or issue summary
            ticket_context: Additional context about what's been tried
            config_path: Optional path to config file
            top_k: Number of relevant runbooks to retrieve
            
        Returns:
            Dictionary with escalation summary
        """
        from supportdesk_rag import SupportDeskRAGPipeline
        
        if self.verbose:
            print("[INFO] Generating escalation handoff summary...")
        
        pipeline = SupportDeskRAGPipeline()
        
        # Escalation prompt template
        escalation_prompt = f"""You are an escalation handoff specialist. Create a concise, actionable summary for escalating this support ticket to the next team.

Provide the following sections:

1. Issue Summary: 1-2 sentence overview of the problem
2. What's Been Tried: Steps already attempted by support (if known)
3. Relevant Logs/Error Messages: Any error details or logs
4. Next Team: Which team should handle this (engineering, billing, etc.)
5. Suggested Action Items: 2-3 specific actions for the next team

Ticket/Issue:
{prompt}

Additional Context (if provided):
{ticket_context or 'No additional context provided'}

Format your response in a clear, structured format suitable for escalation tracking systems."""

        try:
            result = pipeline.generate_answer(
                query_text=escalation_prompt,
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
                "ticket_context": ticket_context,
                "escalation_summary": answer_text,
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
                "escalation_summary": "Unable to generate escalation summary.",
                "citations": []
            }

    def format_output(self, result: Dict[str, Any], output_format: str = "markdown") -> str:
        """Format the output for display."""
        if output_format == "json":
            return json.dumps(result, indent=2, default=str)
        
        # Markdown format
        lines = []
        lines.append("# 🚨 Escalation Handoff Summary")
        lines.append("")
        
        if "error" in result:
            lines.append("## ❌ Error")
            lines.append("")
            lines.append(f"```text\n{result['error']}\n```")
            return "\n".join(lines)
        
        lines.append(f"**Original Issue:** {result.get('prompt', 'N/A')[:200]}...")
        if result.get('ticket_context'):
            lines.append(f"**Context:** {result['ticket_context'][:200]}...")
        lines.append("")
        
        escalation_summary = result.get('escalation_summary', '')
        if escalation_summary:
            lines.append("## Escalation Summary")
            lines.append("")
            lines.append(escalation_summary)
            lines.append("")
        
        citations = result.get('citations', [])
        if citations:
            lines.append("## 📚 Citations")
            lines.append("")
            for citation in citations:
                ticket_id = citation.get('ticket_id', 'N/A')
                score = citation.get('score', 'N/A')
                lines.append(f"- [{ticket_id}] (Score: {score:.3f})")
            lines.append("")
        
        return "\n".join(lines)


def main():
    """Main entry point for the escalate command."""
    parser = argparse.ArgumentParser(
        description="Generate escalation handoff summaries"
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        required=True,
        help="Ticket description or issue summary"
    )
    parser.add_argument(
        "--context", "-c",
        type=str,
        default=None,
        help="Additional context about what's been tried"
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
        help="Number of relevant items to retrieve (default: 5)"
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
    
    command = RagEscalateCommand(
        verbose=args.verbose,
        debug=args.debug
    )
    
    result = command.run(
        prompt=args.prompt,
        ticket_context=args.context,
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

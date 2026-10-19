"""RAG Runbook Command - Runbook assistance.

This module implements the runbook command for the SupportDesk RAG CLI.
It finds the right procedure for an incident and provides step-by-step guidance.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class RagRunbookCommand:
    """Command to provide runbook assistance."""

    def __init__(self, verbose: bool = False, debug: bool = False):
        self.verbose = verbose or debug
        self.debug = debug

    def run(self, prompt: str, config_path: Optional[str] = None,
            top_k: int = 5) -> Dict[str, Any]:
        """Run the runbook command.
        
        Args:
            prompt: Description of the incident or procedure needed
            config_path: Optional path to config file
            top_k: Number of runbooks to retrieve
            
        Returns:
            Dictionary with runbook assistance
        """
        from supportdesk_rag import SupportDeskRAGPipeline
        
        if self.verbose:
            print("[INFO] Finding runbook assistance...")
        
        pipeline = SupportDeskRAGPipeline()
        
        # Runbook prompt template
        runbook_prompt = f"""You are a runbook assistant. Find the most relevant runbook procedure for this incident and provide step-by-step guidance.

1. Identify the incident category
2. List the relevant runbook procedure with step-by-step instructions
3. Highlight any critical warnings or prerequisites
4. Include any related troubleshooting steps

Incident Description:
{prompt}

Format your response with clear sections and numbered steps where applicable."""

        try:
            result = pipeline.generate_answer(
                query_text=runbook_prompt,
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
                "runbook_assistance": answer_text,
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
                "runbook_assistance": "Unable to find runbook assistance.",
                "citations": []
            }

    def format_output(self, result: Dict[str, Any], output_format: str = "markdown") -> str:
        """Format the output for display."""
        if output_format == "json":
            return json.dumps(result, indent=2, default=str)
        
        # Markdown format
        lines = []
        lines.append("# 📖 Runbook Assistance")
        lines.append("")
        
        if "error" in result:
            lines.append("## ❌ Error")
            lines.append("")
            lines.append(f"```text\n{result['error']}\n```")
            return "\n".join(lines)
        
        lines.append(f"**Incident:** {result.get('prompt', 'N/A')[:200]}...")
        lines.append("")
        
        runbook_assistance = result.get('runbook_assistance', '')
        if runbook_assistance:
            lines.append("## 📋 Runbook Assistance")
            lines.append("")
            lines.append(runbook_assistance)
            lines.append("")
        
        citations = result.get('citations', [])
        if citations:
            lines.append("## 📚 Citations (Source Runbooks)")
            lines.append("")
            for citation in citations:
                ticket_id = citation.get('ticket_id', 'N/A')
                score = citation.get('score', 'N/A')
                retriever = citation.get('retriever', 'N/A')
                lines.append(f"- [{ticket_id}] (Score: {score:.3f}, Retriever: {retriever})")
            lines.append("")
        
        return "\n".join(lines)


def main():
    """Main entry point for the runbook command."""
    parser = argparse.ArgumentParser(
        description="Get runbook assistance for incidents"
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        required=True,
        help="Description of the incident or procedure needed"
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
        help="Number of runbooks to retrieve (default: 5)"
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
    
    command = RagRunbookCommand(
        verbose=args.verbose,
        debug=args.debug
    )
    
    result = command.run(
        prompt=args.prompt,
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

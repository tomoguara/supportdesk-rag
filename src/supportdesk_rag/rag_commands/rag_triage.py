"""RAG Triage Command - Ticket triage and categorization.

This module implements the triage command for the SupportDesk RAG CLI.
It analyzes incoming tickets and categorizes them by type, urgency, and suggested resolution.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class RagTriageCommand:
    """Command to perform ticket triage."""

    def __init__(self, verbose: bool = False, debug: bool = False):
        self.verbose = verbose or debug
        self.debug = debug

    def run(self, prompt: str, config_path: Optional[str] = None,
            top_k: int = 5) -> Dict[str, Any]:
        """Run the triage command.
        
        Args:
            prompt: Ticket description to triage
            config_path: Optional path to config file
            top_k: Number of similar cases to retrieve
            
        Returns:
            Dictionary with triage results
        """
        from supportdesk_rag import SupportDeskRAGPipeline
        
        if self.verbose:
            print("[INFO] Initializing Triage System...")
        
        pipeline = SupportDeskRAGPipeline()
        
        # Triage prompt template
        triage_prompt = f"""You are a support ticket triage assistant. Analyze the following incoming ticket and provide:

1. Summary: A concise summary of the issue (1-2 sentences)
2. Category: Identify the most likely category from: password, billing, technical, account, feature-request, other
3. Urgency: Assess urgency as: high, medium, low
4. Relevant Troubleshooting Steps: List 3-5 specific actions a support agent should ask the customer to check

Ticket Description:
{prompt}

Format your response as JSON with keys: summary, category, urgency, troubleshooting_steps."""

        if self.verbose:
            print("[INFO] Running triage analysis...")
        
        try:
            result = pipeline.generate_answer(
                query_text=triage_prompt,
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
                "triage_response": answer_text,
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
                "triage_response": "Unable to triage ticket. Please check logs.",
                "citations": []
            }

    def format_output(self, result: Dict[str, Any], output_format: str = "markdown") -> str:
        """Format the output for display."""
        if output_format == "json":
            return json.dumps(result, indent=2, default=str)
        
        # Markdown format
        lines = []
        lines.append("# Ticket Triage Results")
        lines.append("")
        
        if "error" in result:
            lines.append("## ❌ Error")
            lines.append("")
            lines.append(f"```text\n{result['error']}\n```")
            return "\n".join(lines)
        
        lines.append(f"**Ticket Description:** {result.get('prompt', 'N/A')[:200]}...")
        lines.append("")
        
        triage_response = result.get('triage_response', '')
        if triage_response:
            # Try to parse JSON response
            try:
                import re
                json_match = re.search(r'\{.*\}', triage_response, re.DOTALL)
                if json_match:
                    triage_data = json.loads(json_match.group())
                    lines.append("## 📋 Triage Analysis")
                    lines.append("")
                    
                    if triage_data.get('summary'):
                        lines.append("### Summary")
                        lines.append("")
                        lines.append(triage_data['summary'])
                        lines.append("")
                    
                    if triage_data.get('category'):
                        lines.append(f"### Category: `{triage_data['category']}`")
                        lines.append("")
                    
                    if triage_data.get('urgency'):
                        urgency = triage_data['urgency'].upper()
                        urgency_icon = "🔴" if urgency == "HIGH" else ("🟡" if urgency == "MEDIUM" else "🟢")
                        lines.append(f"### Urgency: {urgency_icon} {triage_data['urgency']}")
                        lines.append("")
                    
                    if triage_data.get('troubleshooting_steps'):
                        lines.append("### 🔍 Troubleshooting Steps")
                        lines.append("")
                        steps = triage_data['troubleshooting_steps']
                        if isinstance(steps, list):
                            for i, step in enumerate(steps, 1):
                                lines.append(f"{i}. {step}")
                        else:
                            lines.append(steps)
                        lines.append("")
                    
            except (json.JSONDecodeError, ValueError):
                # Not valid JSON, display as plain text
                lines.append("## Analysis")
                lines.append("")
                lines.append(triage_response)
                lines.append("")
        
        citations = result.get('citations', [])
        if citations:
            lines.append("## 📚 Citations")
            lines.append("")
            for citation in citations[:3]:  # Show top 3
                ticket_id = citation.get('ticket_id', 'N/A')
                score = citation.get('score', 'N/A')
                lines.append(f"- [{ticket_id}] (Score: {score})")
            lines.append("")
        
        return "\n".join(lines)


def main():
    """Main entry point for the triage command."""
    parser = argparse.ArgumentParser(
        description="Triage support tickets"
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        required=True,
        help="Ticket description to triage"
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
        help="Number of similar cases to retrieve (default: 5)"
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
    
    command = RagTriageCommand(
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

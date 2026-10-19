"""RAG Bulk Command - Bulk ticket analysis.

This module implements the bulk command for the SupportDesk RAG CLI.
It processes multiple tickets from a file and groups them by patterns.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class RagBulkCommand:
    """Command for bulk ticket analysis."""

    def __init__(self, verbose: bool = False, debug: bool = False):
        self.verbose = verbose or debug
        self.debug = debug

    def run(self, file_path: str, analysis_type: str = "group_by_product",
            config_path: Optional[str] = None, top_k: int = 5) -> Dict[str, Any]:
        """Run the bulk analysis command.
        
        Args:
            file_path: Path to the file containing multiple prompts/tickets
            analysis_type: Type of analysis ('group_by_product', 'group_by_symptom', 'summarize')
            config_path: Optional path to config file
            top_k: Number of results per ticket
            
        Returns:
            Dictionary with bulk analysis results
        """
        from supportdesk_rag import SupportDeskRAGPipeline
        
        if self.verbose:
            print(f"[INFO] Loading bulk data from {file_path}...")
        
        # Load input file
        try:
            with open(file_path, 'r') as f:
                input_data = json.load(f)
            
            # Handle different input formats
            if isinstance(input_data, dict) and 'tickets' in input_data:
                tickets = input_data['tickets']
            elif isinstance(input_data, list):
                tickets = input_data
            elif isinstance(input_data, dict):
                # Assume it's a single object or has prompts
                tickets = [input_data]
            else:
                tickets = []
            
            if self.verbose:
                print(f"[INFO] Loaded {len(tickets)} tickets for analysis")
                
        except json.JSONDecodeError as e:
            return {
                "error": f"Invalid JSON file: {e}",
                "file_path": file_path,
                "tickets_loaded": 0,
                "analysis_type": analysis_type
            }
        
        if self.verbose:
            print(f"[INFO] Running {analysis_type} analysis...")
        
        pipeline = SupportDeskRAGPipeline()
        results = []
        
        try:
            for i, ticket in enumerate(tickets):
                if self.verbose and (i + 1) % 10 == 0:
                    print(f"[INFO] Processed {i + 1}/{len(tickets)} tickets...")
                
                # Extract prompt from ticket
                if isinstance(ticket, str):
                    prompt = ticket
                elif isinstance(ticket, dict):
                    prompt = ticket.get('prompt', ticket.get('description', ticket.get('query', '')))
                else:
                    continue
                
                # Run the pipeline
                result = pipeline.generate_answer(
                    query_text=prompt,
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
                
                # Store ticket ID if available
                ticket_id = ticket.get('ticket_id', ticket.get('id', f'ticket_{i+1}'))
                
                results.append({
                    "ticket_id": ticket_id,
                    "prompt": prompt,
                    "answer": answer_text,
                    "citations": citations,
                    "metrics": pipeline.get_observability_metrics()
                })
            
            output = {
                "file_path": file_path,
                "tickets_loaded": len(tickets),
                "tickets_processed": len(results),
                "analysis_type": analysis_type,
                "results": results,
                "summary": self._generate_summary(results, analysis_type)
            }
            
            return output
            
        except Exception as e:
            if self.debug:
                raise
            return {
                "error": str(e),
                "file_path": file_path,
                "tickets_loaded": len(tickets) if tickets else 0,
                "tickets_processed": len(results),
                "results": results
            }

    def _generate_summary(self, results: List[Dict], analysis_type: str) -> Dict:
        """Generate summary statistics for bulk results."""
        total = len(results)
        if total == 0:
            return {"total_tickets": 0}
        
        successful = sum(1 for r in results if 'answer' in r and r.get('answer'))
        failed = total - successful
        
        # Calculate average metrics
        avg_metrics = {}
        
        return {
            "total_tickets": total,
            "successful_answers": successful,
            "failed_answers": failed,
            "success_rate": successful / total if total > 0 else 0
        }

    def format_output(self, result: Dict[str, Any], output_format: str = "markdown") -> str:
        """Format the output for display."""
        if output_format == "json":
            return json.dumps(result, indent=2, default=str)
        
        # Markdown format
        lines = []
        lines.append("# 📊 Bulk Analysis Results")
        lines.append("")
        
        if "error" in result:
            lines.append("## ❌ Error")
            lines.append("")
            lines.append(f"```text\n{result['error']}\n```")
            return "\n".join(lines)
        
        summary = result.get('summary', {})
        lines.append("## 📋 Summary")
        lines.append("")
        lines.append(f"- **File:** {result.get('file_path', 'N/A')}")
        lines.append(f"- **Analysis Type:** {result.get('analysis_type', 'N/A')}")
        lines.append(f"- **Total Tickets:** {summary.get('total_tickets', 0)}")
        lines.append(f"- **Successful Answers:** {summary.get('successful_answers', 0)}")
        lines.append(f"- **Failed Answers:** {summary.get('failed_answers', 0)}")
        lines.append(f"- **Success Rate:** {summary.get('success_rate', 0):.1%}")
        lines.append("")
        
        results = result.get('results', [])
        if results:
            lines.append("## 📄 individual Results")
            lines.append("")
            
            for i, r in enumerate(results[:10]):  # Show first 10
                lines.append(f"### Ticket {i+1}: {r.get('ticket_id', 'N/A')}")
                lines.append("")
                lines.append(f"**Prompt:** {r.get('prompt', 'N/A')[:150]}...")
                lines.append("")
                
                if r.get('answer'):
                    lines.append(f"**Answer Preview:** {r['answer'][:300]}...")
                    lines.append("")
                
                citations = r.get('citations', [])
                if citations:
                    lines.append(f"**Citations:** {len(citations)}")
                    lines.append("")
        
        if len(results) > 10:
            lines.append(f"... and {len(results) - 10} more tickets")
            lines.append("")
        
        return "\n".join(lines)


def main():
    """Main entry point for the bulk command."""
    parser = argparse.ArgumentParser(
        description="Bulk ticket analysis"
    )
    parser.add_argument(
        "--file-path", "-f",
        type=str,
        required=True,
        help="Path to file containing tickets (JSON format)"
    )
    parser.add_argument(
        "--analysis-type", "-a",
        type=str,
        default="group_by_product",
        choices=["group_by_product", "group_by_symptom", "summarize"],
        help="Type of analysis to perform"
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
        help="Number of results per ticket (default: 5)"
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
    
    command = RagBulkCommand(
        verbose=args.verbose,
        debug=args.debug
    )
    
    result = command.run(
        file_path=args.file_path,
        analysis_type=args.analysis_type,
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

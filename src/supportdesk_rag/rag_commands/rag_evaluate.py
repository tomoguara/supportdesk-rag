"""RAG Evaluate Command - Evaluate RAG responses.

This module implements the evaluate command for the SupportDesk RAG CLI.
It runs evaluation metrics on RAG responses with optional ground truth.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class RagEvaluateCommand:
    """Command to evaluate RAG responses."""

    def __init__(self, verbose: bool = False, debug: bool = False):
        self.verbose = verbose or debug
        self.debug = debug

    def run(self, prompt: str, ground_truth: Optional[str] = None,
            config_path: Optional[str] = None, top_k: int = 5) -> Dict[str, Any]:
        """Run the evaluation command.
        
        Args:
            prompt: User's question
            ground_truth: Optional ground truth answer for comparison
            config_path: Optional path to config file
            top_k: Number of results to retrieve
            
        Returns:
            Dictionary with evaluation results
        """
        from supportdesk_rag import (
            SupportDeskRAGPipeline,
            create_evaluator,
            EvaluationPipeline
        )
        
        if self.verbose:
            print("[INFO] Initializing RAG Pipeline and Evaluator...")
        
        # Initialize RAG pipeline
        rag_pipeline = SupportDeskRAGPipeline()
        
        # Initialize evaluator
        evaluator = create_evaluator()
        
        # Evaluate the query
        try:
            if self.verbose:
                print("[INFO] Running RAG pipeline and evaluation...")
            
            # Create evaluation pipeline
            eval_pipeline = EvaluationPipeline(
                rag_pipeline=rag_pipeline,
                evaluator=evaluator
            )
            
            # Evaluate single query
            eval_result = eval_pipeline.evaluate_query(
                query_id="eval-001",
                query_text=prompt,
                ground_truth=ground_truth
            )
            
            # Get evaluation result
            result = {
                "prompt": prompt,
                "ground_truth": ground_truth,
                "predicted_answer": eval_result.predicted_answer,
                "relevance_score": eval_result.relevance_score,
                "answer_quality_score": eval_result.answer_quality_score,
                "hallucination_score": eval_result.hallucination_score,
                "context_precision": eval_result.context_precision,
                "context_recall": eval_result.context_recall,
                "latency_ms": eval_result.latency_ms,
                "model": eval_result.model,
                "timestamp": eval_result.timestamp
            }
            
            return result
            
        except Exception as e:
            if self.debug:
                raise
            return {
                "error": str(e),
                "prompt": prompt,
                "ground_truth": ground_truth,
                "predicted_answer": "",
                "relevance_score": 0.0,
                "answer_quality_score": 0.0,
                "hallucination_score": 1.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
                "latency_ms": 0.0
            }

    def format_output(self, result: Dict[str, Any], output_format: str = "markdown") -> str:
        """Format the output for display."""
        if output_format == "json":
            return json.dumps(result, indent=2, default=str)
        
        # Markdown format
        lines = []
        lines.append("# RAG Evaluation Results")
        lines.append("")
        
        if "error" in result:
            lines.append("## ❌ Error")
            lines.append("")
            lines.append(f"```text\n{result['error']}\n```")
            return "\n".join(lines)
        
        lines.append(f"**Question:** {result.get('prompt', 'N/A')}")
        lines.append("")
        
        predicted = result.get('predicted_answer', '')
        if predicted:
            lines.append("## Predicted Answer")
            lines.append("")
            lines.append(predicted)
            lines.append("")
        
        ground_truth = result.get('ground_truth')
        if ground_truth:
            lines.append("## Ground Truth")
            lines.append("")
            lines.append(ground_truth)
            lines.append("")
        
        lines.append("## Metrics")
        lines.append("")
        lines.append("| Metric | Score |")
        lines.append("|--------|-------|")
        
        metrics = [
            ("Relevance", result.get('relevance_score')),
            ("Answer Quality", result.get('answer_quality_score')),
            ("Hallucination", result.get('hallucination_score')),
            ("Context Precision", result.get('context_precision')),
            ("Context Recall", result.get('context_recall')),
            ("Latency (ms)", result.get('latency_ms')),
        ]
        
        for name, value in metrics:
            if value is not None:
                score_str = f"{value:.3f}" if isinstance(value, float) else str(value)
                lines.append(f"| {name} | {score_str} |")
        
        lines.append("")
        
        if result.get('model'):
            lines.append(f"**Model:** {result.get('model')}")
            lines.append("")
        
        if result.get('timestamp'):
            lines.append(f"**Timestamp:** {result.get('timestamp')}")
            lines.append("")
        
        return "\n".join(lines)


def main():
    """Main entry point for the evaluate command."""
    parser = argparse.ArgumentParser(
        description="Evaluate RAG responses with metrics"
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        required=True,
        help="User's question to evaluate"
    )
    parser.add_argument(
        "--ground-truth", "-g",
        type=str,
        default=None,
        help="Ground truth answer for comparison"
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
    
    command = RagEvaluateCommand(
        verbose=args.verbose,
        debug=args.debug
    )
    
    result = command.run(
        prompt=args.prompt,
        ground_truth=args.ground_truth,
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

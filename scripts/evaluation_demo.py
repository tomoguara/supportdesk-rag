#!/usr/bin/env python3
"""
Demo script for the Evaluation Framework.

This script demonstrates how to use the evaluation framework to:
1. Evaluate individual queries
2. Compute basic metrics
3. Export results to JSON
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from supportdesk_rag import create_evaluator, compute_basic_metrics, EvaluationPipeline


def demo_basic_metrics():
    """Demonstrate basic metric computation."""
    print("=" * 60)
    print("Demo 1: Basic Metrics")
    print("=" * 60)
    
    answer = "Python is a programming language developed by Guido van Rossum."
    ground_truth = "Python is a programming language created by Guido van Rossum."
    
    metrics = compute_basic_metrics(answer, ground_truth)
    
    print(f"Answer: {answer}")
    print(f"Ground Truth: {ground_truth}")
    print(f"\nMetrics:")
    print(f"  Relevance: {metrics['relevance']:.3f}")
    print(f"  Answer Quality: {metrics['answer_quality']:.3f}")
    print(f"  Hallucination: {metrics['hallucination']:.3f} (lower is better)")
    print()


def demo_evaluator():
    """Demonstrate using the Evaluator class."""
    print("=" * 60)
    print("Demo 2: Evaluator with Ground Truth")
    print("=" * 60)
    
    evaluator = create_evaluator()
    
    test_cases = [
        {
            "query_id": "Q001",
            "query_text": "What is Python?",
            "predicted_answer": "Python is a high-level programming language.",
            "ground_truth": "Python is a high-level, interpreted programming language known for its simplicity."
        },
        {
            "query_id": "Q002",
            "query_text": "What is Java?",
            "predicted_answer": "Java is a programming language.",
            "ground_truth": "Java is a class-based, object-oriented programming language."
        },
        {
            "query_id": "Q003",
            "query_text": "What is Rust?",
            "predicted_answer": "Rust is a systems programming language focused on safety.",
            "ground_truth": "Rust is a multi-paradigm programming language focused on performance and safety."
        }
    ]
    
    for test in test_cases:
        result = evaluator.evaluate(
            query_id=test["query_id"],
            query_text=test["query_text"],
            predicted_answer=test["predicted_answer"],
            ground_truth=test["ground_truth"],
            latency_ms=150 + hash(test["query_id"]) % 50
        )
        print(f"{test['query_id']}:")
        print(f"  Relevance: {result.relevance_score:.3f}")
        print(f"  Quality: {result.answer_quality_score:.3f}")
        print(f"  Hallucination: {result.hallucination_score:.3f}")
    
    print("\nSummary:")
    summary = evaluator.get_summary()
    print(f"  Total Evaluations: {summary.total_evaluations}")
    print(f"  Avg Relevance: {summary.average_relevance:.3f}")
    print(f"  Avg Quality: {summary.average_answer_quality:.3f}")
    print(f"  Avg Hallucination: {summary.average_hallucination:.3f}")
    print()


def demo_with_context():
    """Demonstrate evaluation with retrieved context."""
    print("=" * 60)
    print("Demo 3: Evaluation with Context")
    print("=" * 60)
    
    evaluator = create_evaluator()
    
    context = [
        {"text": "Python is a high-level programming language created by Guido van Rossum."},
        {"text": "Python emphasizes code readability with significant indentation."},
        {"text": "Java is another popular programming language developed by Sun Microsystems."},
    ]
    
    result = evaluator.evaluate(
        query_id="Q004",
        query_text="What is Python?",
        predicted_answer="Python is a high-level programming language created by Guido van Rossum.",
        ground_truth="Python is a high-level programming language created by Guido van Rossum.",
        retrieved_context=context
    )
    
    print(f"Query: What is Python?")
    print(f"Context retrieved: {len(context)} documents")
    print(f"\nMetrics:")
    print(f"  Relevance: {result.relevance_score:.3f}")
    print(f"  Context Precision: {result.context_precision:.3f}")
    print(f"  Context Recall: {result.context_recall:.3f}")
    print()


def demo_export():
    """Demonstrate exporting results."""
    print("=" * 60)
    print("Demo 4: Export Results to JSON")
    print("=" * 60)
    
    import tempfile
    import os
    
    evaluator = create_evaluator()
    
    evaluator.evaluate(
        query_id="export-001",
        query_text="What is Python?",
        predicted_answer="Python is a programming language.",
        ground_truth="Python is a programming language."
    )
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    
    try:
        evaluator.export_results(temp_path)
        print(f"Results exported to: {temp_path}")
        
        with open(temp_path, 'r') as f:
            import json
            data = json.load(f)
            print(f"  Total evaluations: {data['summary']['total_evaluations']}")
            print(f"  Results count: {len(data['results'])}")
    finally:
        os.unlink(temp_path)
    
    print()


def demo_full_pipeline():
    """Demonstrate using EvaluationPipeline with RAG system."""
    print("=" * 60)
    print("Demo 5: Evaluation Pipeline")
    print("=" * 60)
    
    try:
        from supportdesk_rag import SupportDeskRAGPipeline, create_evaluation_pipeline
        
        print("Usage example:")
        print("""
        # Create RAG pipeline
        rag_pipeline = SupportDeskRAGPipeline()
        
        # Create evaluation pipeline  
        eval_pipeline = create_evaluation_pipeline(rag_pipeline=rag_pipeline)
        
        # Evaluate a single query
        result = eval_pipeline.evaluate_query(
            query_id="demo-001",
            query_text="What is the system?",
            ground_truth="SupportDesk RAG is a support ticket analysis system."
        )
        
        # Save results
        eval_pipeline.export_results("evaluation_results.json")
        """)
        
    except ImportError as e:
        print(f"Note: Skipping full pipeline demo (module import issue: {e})")
    
    print()


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("Evaluation Framework Demo")
    print("=" * 60 + "\n")
    
    demo_basic_metrics()
    demo_evaluator()
    demo_with_context()
    demo_export()
    demo_full_pipeline()
    
    print("=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. compute_basic_metrics() - Quick metric computation")
    print("2. Evaluator - Detailed evaluation with ground truth")
    print("3. EvaluationPipeline - Full integration with RAG system")
    print("4. Export - Save results for analysis")


if __name__ == "__main__":
    main()

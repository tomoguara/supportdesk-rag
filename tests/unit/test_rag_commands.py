"""Tests for RAG command modules."""

import sys
import os
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def test_imports():
    """Test that all command classes can be imported."""
    from supportdesk_rag.rag_commands import (
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
    
    # Test that all classes exist and have required methods
    commands = [
        RagAnswerCommand,
        RagTriageCommand,
        RagSimilarCommand,
        RagIngestCommand,
        RagEvaluateCommand,
        RagBulkCommand,
        RagEscalateCommand,
        RagRunbookCommand,
        RagGapCommand,
    ]
    
    for cmd in commands:
        instance = cmd(verbose=True, debug=True)
        # Check that run method exists
        assert hasattr(instance, 'run'), f"{cmd.__name__} missing run method"
        # Check that format_output method exists
        assert hasattr(instance, 'format_output'), f"{cmd.__name__} missing format_output method"
    
    print("All command classes imported and have required methods")


def test_answer_command():
    """Test RagAnswerCommand initialization."""
    from supportdesk_rag.rag_commands.rag_answer import RagAnswerCommand
    
    cmd = RagAnswerCommand(verbose=False, debug=False)
    assert cmd.verbose == False
    assert cmd.debug == False
    
    cmd = RagAnswerCommand(verbose=True, debug=False)
    assert cmd.verbose == True
    assert cmd.debug == False
    
    cmd = RagAnswerCommand(verbose=True, debug=True)
    assert cmd.verbose == True
    assert cmd.debug == True
    
    print("RagAnswerCommand initialization tests passed")


def test_evaluate_command():
    """Test RagEvaluateCommand initialization."""
    from supportdesk_rag.rag_commands.rag_evaluate import RagEvaluateCommand
    
    cmd = RagEvaluateCommand(verbose=False, debug=False)
    assert cmd.verbose == False
    assert cmd.debug == False
    
    cmd = RagEvaluateCommand(verbose=True, debug=True)
    assert cmd.verbose == True
    assert cmd.debug == True
    
    print("RagEvaluateCommand initialization tests passed")


def test_triage_command():
    """Test RagTriageCommand initialization."""
    from supportdesk_rag.rag_commands.rag_triage import RagTriageCommand
    
    cmd = RagTriageCommand(verbose=False, debug=False)
    assert cmd.verbose == False
    assert cmd.debug == False
    
    print("RagTriageCommand initialization tests passed")


def test_similar_command():
    """Test RagSimilarCommand initialization."""
    from supportdesk_rag.rag_commands.rag_similar import RagSimilarCommand
    
    cmd = RagSimilarCommand(verbose=False, debug=False)
    assert cmd.verbose == False
    assert cmd.debug == False
    
    print("RagSimilarCommand initialization tests passed")


def test_ingest_command():
    """Test RagIngestCommand initialization."""
    from supportdesk_rag.rag_commands.rag_ingest import RagIngestCommand
    
    cmd = RagIngestCommand(verbose=False, debug=False)
    assert cmd.verbose == False
    assert cmd.debug == False
    
    print("RagIngestCommand initialization tests passed")


def test_escalate_command():
    """Test RagEscalateCommand initialization."""
    from supportdesk_rag.rag_commands.rag_escalate import RagEscalateCommand
    
    cmd = RagEscalateCommand(verbose=False, debug=False)
    assert cmd.verbose == False
    assert cmd.debug == False
    
    print("RagEscalateCommand initialization tests passed")


def test_runbook_command():
    """Test RagRunbookCommand initialization."""
    from supportdesk_rag.rag_commands.rag_runbook import RagRunbookCommand
    
    cmd = RagRunbookCommand(verbose=False, debug=False)
    assert cmd.verbose == False
    assert cmd.debug == False
    
    print("RagRunbookCommand initialization tests passed")


def test_gap_command():
    """Test RagGapCommand initialization."""
    from supportdesk_rag.rag_commands.rag_gap import RagGapCommand
    
    cmd = RagGapCommand(verbose=False, debug=False)
    assert cmd.verbose == False
    assert cmd.debug == False
    
    print("RagGapCommand initialization tests passed")


def test_bulk_command():
    """Test RagBulkCommand initialization."""
    from supportdesk_rag.rag_commands.rag_bulk import RagBulkCommand
    
    cmd = RagBulkCommand(verbose=False, debug=False)
    assert cmd.verbose == False
    assert cmd.debug == False
    
    print("RagBulkCommand initialization tests passed")


def test_format_output_markdown():
    """Test markdown output formatting."""
    from supportdesk_rag.rag_commands.rag_answer import RagAnswerCommand
    
    cmd = RagAnswerCommand()
    
    # Test answer output
    result = {
        "prompt": "Test question",
        "answer": "This is a test answer.",
        "citations": [{"ticket_id": "T001", "score": 0.85}]
    }
    
    output = cmd.format_output(result, output_format="markdown")
    assert "## Answer" in output
    assert "Test question" in output
    assert "This is a test answer" in output
    assert "## Citations" in output
    assert "T001" in output
    
    # Test error output (contains error key)
    error_result = {"error": "Test error"}
    error_output = cmd.format_output(error_result, output_format="markdown")
    assert "❌ Error" in error_output
    
    print("Markdown output formatting tests passed")


def test_format_output_json():
    """Test JSON output formatting."""
    from supportdesk_rag.rag_commands.rag_answer import RagAnswerCommand
    
    cmd = RagAnswerCommand()
    
    result = {
        "prompt": "Test question",
        "answer": "This is a test answer.",
        "citations": []
    }
    
    output = cmd.format_output(result, output_format="json")
    
    # Verify it's valid JSON
    parsed = json.loads(output)
    assert parsed["prompt"] == "Test question"
    assert parsed["answer"] == "This is a test answer."
    
    print("JSON output formatting tests passed")


def test_cli_entry_point():
    """Test that the CLI entry point is accessible."""
    # Check that cli.py exists (moved from support-rag.py at project root)
    cli_path = Path(__file__).parent.parent.parent / "src" / "supportdesk_rag" / "cli.py"
    assert cli_path.exists(), f"CLI entry point not found at {cli_path}"
    
    # Check that __main__.py exists for module execution support
    main_module_path = Path(__file__).parent.parent.parent / "src" / "supportdesk_rag" / "__main__.py"
    assert main_module_path.exists(), f"__main__.py not found at {main_module_path}"
    
    # Check that all command modules exist
    commands_dir = Path(__file__).parent.parent.parent / "src" / "supportdesk_rag" / "rag_commands"
    assert commands_dir.exists(), f"Commands directory not found at {commands_dir}"
    
    required_files = [
        "rag_answer.py",
        "rag_triage.py", 
        "rag_similar.py",
        "rag_ingest.py",
        "rag_evaluate.py",
        "rag_bulk.py",
        "rag_escalate.py",
        "rag_runbook.py",
        "rag_gap.py",
    ]
    
    for file in required_files:
        file_path = commands_dir / file
        assert file_path.exists(), f"Command module not found: {file}"
    
    print("CLI entry point and all command modules exist")


if __name__ == "__main__":
    test_imports()
    test_answer_command()
    test_evaluate_command()
    test_triage_command()
    test_similar_command()
    test_ingest_command()
    test_escalate_command()
    test_runbook_command()
    test_gap_command()
    test_bulk_command()
    test_format_output_markdown()
    test_format_output_json()
    test_cli_entry_point()
    print("\n✅ All tests passed!")

"""Functional tests for Guard Wrapper with real components.

These tests verify the guard wrapper integrations with real answers and citations
without using mocks. The tests verify:
- Citation verification with real answer content
- Refusal detection with actual refusal patterns
- PII redaction with real PII patterns
- Length constraint validation
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from supportdesk_rag.supportdesk_guard import (
    Guardrails,
)


class TestGuardrailsFunctional:
    """Functional tests for Guardrails with real content."""

    def test_citation_verification_with_real_answer(self):
        """Test that guard correctly verifies citations in real answers."""
        guard = Guardrails(enable_citation_check=True)

        # Real answer with proper citations
        answer = """The network timeout issue can be resolved by adjusting the connection timeout settings.

Based on the ticket INC-2024-001, the default timeout is 30 seconds.

You can find more details in [CITATION:INC-2024-001].

Additional configuration options are available in [CITATION:INC-2024-002]."""

        # Context citations matching the answer
        context_citations = [
            {"ticket_id": "INC-2024-001"},
            {"ticket_id": "INC-2024-002"},
        ]

        result = guard.verify_citations(answer, context_citations)

        # All citations should be valid
        assert result["valid"] is True
        assert result["missing"] == []
        assert len(result["answer_citations"]) == 2

    def test_citation_verification_with_missing_citation(self):
        """Test that guard correctly identifies missing citations."""
        guard = Guardrails(enable_citation_check=True)

        # Answer claims to cite INC-999 which doesn't exist in context
        answer = """The solution involves multiple steps.

See [CITATION:INC-2024-001] for first step.

See [CITATION:INC-999] for second step."""

        context_citations = [
            {"ticket_id": "INC-2024-001"},
        ]

        result = guard.verify_citations(answer, context_citations)

        # INC-999 should be flagged as missing
        assert result["valid"] is False
        assert "INC-999" in result["missing"]
        assert "INC-2024-001" not in result["missing"]

    def test_full_process_answer_validation_flow(self):
        """Test the complete process_answer flow with a valid answer."""
        guard = Guardrails(
            enable_citation_check=True,
            enable_relevance_gate=True,
            enable_pii_redact=True,
            min_citations_per_answer=1,
            max_tokens=1000,
            min_tokens=10,
        )

        # Realistic answer with proper citation
        answer = """The network timeout can be fixed by adjusting the connection timeout settings in the configuration file.

According to [CITATION:INC-2024-001], the default value is 30 seconds.

You can also refer to [CITATION:INC-2024-002] for additional configuration options."""

        question = "How do I fix the network timeout issue?"

        context_citations = [
            {"ticket_id": "INC-2024-001"},
            {"ticket_id": "INC-2024-002"},
        ]

        result = guard.process_answer(answer, question, context_citations)

        # Should be allowed with valid citations
        assert result["allowed"] is True
        assert len(result["checks"]["citation"]["answer_citations"]) == 2

    def test_full_process_answer_with_refusal(self):
        """Test the complete process_answer flow with a refusal answer."""
        guard = Guardrails(
            enable_citation_check=False,  # Skip citation check for refusals
            enable_relevance_gate=False,
            enable_pii_redact=False,
            min_citations_per_answer=0,
        )

        # Real refusal answer
        answer = "I cannot find sufficient information to answer this question."

        question = "What is the solution?"

        context_citations = []

        result = guard.process_answer(answer, question, context_citations)

        # Refusals should be allowed but flagged
        assert result["allowed"] is True
        assert any("Refusal detected" in action for action in result["actions"])

    def test_full_process_answer_with_pii(self):
        """Test the complete process_answer flow with PII that needs redaction."""
        guard = Guardrails(
            enable_citation_check=False,
            enable_relevance_gate=False,
            enable_pii_redact=True,
            min_citations_per_answer=0,
        )

        # Answer containing PII
        answer = "The user's SSN is 123-45-6789 and can be reached at 555-123-4567."

        question = "What is the user's contact info?"

        context_citations = []

        result = guard.process_answer(answer, question, context_citations)

        # PII should be redacted
        assert result["allowed"] is True
        assert "[SSN_REDACTED]" in result["modified_answer"]
        assert "[PHONE_REDACTED]" in result["modified_answer"]
        assert "123-45-6789" not in result["modified_answer"]
        assert "555-123-4567" not in result["modified_answer"]
        assert result["modified_answer"] != answer

    def test_full_process_answer_with_insufficient_citations(self):
        """Test the complete process_answer flow with insufficient citations."""
        guard = Guardrails(
            enable_citation_check=True,
            enable_relevance_gate=False,
            enable_pii_redact=False,
            min_citations_per_answer=2,  # Require at least 2 citations
            min_tokens=5,
            max_tokens=1000,
        )

        # Answer with only 1 citation but 2 required
        answer = "The solution involves [CITATION:INC-001] and additional information from the ticket."

        question = "What is the solution?"

        context_citations = [
            {"ticket_id": "INC-001"},
        ]

        result = guard.process_answer(answer, question, context_citations)

        # Answer should be blocked due to insufficient citations
        assert result["allowed"] is False
        assert any("Not enough citations" in action for action in result["actions"])

    def test_full_process_answer_with_too_long_answer(self):
        """Test the complete process_answer flow with answer too long."""
        guard = Guardrails(
            enable_citation_check=False,
            enable_relevance_gate=False,
            enable_pii_redact=False,
            max_tokens=50,  # Very small limit for testing
            min_tokens=5,
            min_citations_per_answer=0,
        )

        # Create answer that's too long (more than 50 tokens)
        answer = "word " * 60  # 60 words = 60 tokens (approx)

        question = "What is the solution?"

        context_citations = []

        result = guard.process_answer(answer, question, context_citations)

        # Answer should be blocked due to length
        assert result["allowed"] is False
        assert any("Length" in action for action in result["actions"])


class TestGuardrailsEdgeCases:
    """Edge case tests for Guardrails."""

    def test_empty_answer(self):
        """Test guard with empty answer."""
        guard = Guardrails(min_citations_per_answer=0)

        result = guard.process_answer("", "Question?", [])

        # Empty answer should fail length check
        assert result["allowed"] is False
        assert "length" in result["checks"]

    def test_answer_with_no_meaningful_content(self):
        """Test guard with answer that has no meaningful content."""
        guard = Guardrails(min_tokens=5)

        # Very short answer (less than 5 tokens)
        answer = "Yes."

        result = guard.process_answer(answer, "Question?", [])

        # Should fail length check
        assert result["allowed"] is False

    def test_answer_with_citations_in_context_but_not_in_answer(self):
        """Test guard when context has citations but answer doesn't use them."""
        guard = Guardrails(
            enable_citation_check=True,
            min_citations_per_answer=1,
        )

        answer = "The solution is this without any citations."

        context_citations = [
            {"ticket_id": "INC-001"},
            {"ticket_id": "INC-002"},
        ]

        result = guard.process_answer(answer, "Question?", context_citations)

        # Should fail citation count check
        assert result["allowed"] is False
        assert any("Not enough citations" in action for action in result["actions"])

    def test_citation_verification_with_empty_context(self):
        """Test citation verification with empty context citations."""
        guard = Guardrails()

        answer = "Solution without citations."

        context_citations = []

        result = guard.verify_citations(answer, context_citations)

        # When both answer and context have no citations, it's considered valid
        # (there are no missing citations)
        assert result["valid"] is True
        assert result["missing"] == []

    def test_citation_extraction_with_invalid_format(self):
        """Test citation extraction with incorrectly formatted citations."""
        guard = Guardrails()

        # Invalid citation format (missing closing bracket)
        answer = "Solution with [CITATION:INC-001 incomplete"

        citations = guard.extract_citations(answer)

        # Should not extract invalid citations
        assert citations == []

    def test_citation_extraction_with_nested_brackets(self):
        """Test citation extraction with nested brackets in text."""
        guard = Guardrails()

        # Answer with proper citation among brackets
        answer = "Solution with [CITATION:INC-001] and [text]"

        citations = guard.extract_citations(answer)

        # Should only extract valid citation format
        assert citations == ["INC-001"]

    def test_multiple_refusal_patterns(self):
        """Test refusal detection with various refusal patterns."""
        guard = Guardrails()

        refusal_patterns = [
            "I cannot answer that",
            "I cannot find the information",
            "No sufficient information available",
            "I don't know the answer",
            "I am unable to assist with this",
            "Not enough information provided",
        ]

        for refusal in refusal_patterns:
            assert guard.check_refusal(refusal) is True, f"Should detect: {refusal}"


class TestGuardrailsIntegrationWithPipeline:
    """Integration tests that simulate guard usage in the pipeline."""

    def test_guard_result_structure(self):
        """Test that guard results have the expected structure."""
        guard = Guardrails()

        answer = "The answer is [CITATION:INC-001]."
        question = "What is the answer?"
        citations = [{"ticket_id": "INC-001"}]

        result = guard.process_answer(answer, question, citations)

        # Check all expected keys are present
        assert "allowed" in result
        assert "actions" in result
        assert "modified_answer" in result
        assert "checks" in result

        # Check types
        assert isinstance(result["allowed"], bool)
        assert isinstance(result["actions"], list)
        assert isinstance(result["modified_answer"], str)
        assert isinstance(result["checks"], dict)

    def test_modified_answer_preserves_answer_content(self):
        """Test that modified_answer preserves the original answer when no changes."""
        guard = Guardrails(
            enable_pii_redact=True,
        )

        answer = "This is a clean answer with [CITATION:INC-001]."
        question = "Question?"
        citations = [{"ticket_id": "INC-001"}]

        result = guard.process_answer(answer, question, citations)

        # Since there's no PII, modified should equal original
        assert result["modified_answer"] == answer

    def test_check_relevance_with_no_embedding_service(self):
        """Test relevance check when no embedding service is available."""
        guard = Guardrails(
            enable_relevance_gate=True,
            embedding_service=None,
        )

        question = "What is the solution?"
        answer = "The solution is this."

        result = guard.check_relevance(question, answer)

        # Should return default values when no service available
        assert result["relevant"] is True
        assert result["similarity"] == 1.0
        assert "warning" in result

    def test_check_refusal_case_insensitive(self):
        """Test that refusal detection is case insensitive."""
        guard = Guardrails()

        refusal_variations = [
            "I CANNOT answer",
            "I cannot ANSWER",
            "I Cannot Answer",
            "I CaNnOt FiNd",
        ]

        for refusal in refusal_variations:
            assert guard.check_refusal(refusal) is True, f"Should detect: {refusal}"

    def test_check_length_token_count_calculation(self):
        """Test that token count is calculated correctly (rough estimate)."""
        guard = Guardrails()

        # 40 chars = 10 tokens (4 chars per token estimate)
        answer = "A" * 40

        result = guard.check_length(answer)

        assert result["token_count"] == 10
        assert result["valid"] is True

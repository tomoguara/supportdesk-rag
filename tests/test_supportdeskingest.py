import pytest
from supportdesk_rag.supportdesk_ingest import SupportDeskIngest

def test_load_tickets():
    data_path = "./data/synthetic_tickets.json"
    tickets = SupportDeskIngest().load_ticket(data_path)
    assert tickets is not None

@pytest.mark.parametrize(
        "text_with_pii, expected_redaction", [
            ("This is my SSN numbed: 123-45-6789", "[SSN_REDACTED]"),
            ("This is my email address numbed: test@example.com", "[EMAIL_REDACTED]"),
            ("This is my Credit card number: 1234567890123456", "[CARD_REDACTED]"),
            ("This is not a valid string", "[TEXT_REDACTED]")
        ]
)
def test_clean_text_remove(text_with_pii, expected_redaction):
    result = SupportDeskIngest().clean_text(text_with_pii)
    print(result)
    valid_redactions = [
        "[SSN_REDACTED]",
        "[EMAIL_REDACTED]",
        "[CARD_REDACTED]"
    ]
    if expected_redaction in valid_redactions:
        assert expected_redaction in result, f"Expected {expected_redaction} to be in result"
    else:
        assert expected_redaction not in result, f"Expected{expected_redaction} to not be in result"


def test_get_chunks():
    data_path = "./data/synthetic_tickets.json"
    ingest = SupportDeskIngest(chunk_size=100, chunk_overlap=10)
    tickets = ingest.get_chunks(data_path, clean=True)


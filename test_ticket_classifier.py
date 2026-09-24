from unittest.mock import MagicMock, patch

from ticket_classifier import SupportTicket, classify_ticket


@patch("ticket_classifier.get_client")
def test_classify_ticket(mock_get_client):
    fake_client = MagicMock()

    fake_client.responses.parse.return_value.output_parsed = SupportTicket(
        category="billing",
        priority="high",
        summary="Customer was charged twice"
    )

    mock_get_client.return_value = fake_client

    result = classify_ticket(
        "I was charged twice for my subscription."
    )

    assert result.category == "billing"
    assert result.priority == "high"
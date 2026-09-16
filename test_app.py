from unittest.mock import MagicMock, patch

from app import ask_ai


def test_empty_prompt():
    result = ask_ai("   ")
    assert result == "Please enter a question."


@patch("app.get_client")
def test_normal_prompt(mock_get_client):
    fake_client = MagicMock()
    fake_client.responses.create.return_value.output_text = "Hello"

    mock_get_client.return_value = fake_client

    result = ask_ai("Say hello")

    assert result == "Hello"
from app import ask_ai
from unittest.mock import patch

def test_empty_prompt():
    result = ask_ai("   ")
    assert result == "Please enter a question."


@patch("app.client.responses.create")
def test_normal_prompt(mock_create):
    mock_create.return_value.output_text = "Hello"

    result = ask_ai("Say hello")

    assert result == "Hello"
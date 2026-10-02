import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent import run_agent


@patch("agent.get_client")
def test_run_agent_multi_step(mock_get_client):
    fake_client = MagicMock()

    first_response = SimpleNamespace(
        id="response_1",
        output=[
            SimpleNamespace(
                type="function_call",
                name="add",
                arguments='{"a": 5, "b": 7}',
                call_id="call_1",
            )
        ],
        output_text="",
    )

    second_response = SimpleNamespace(
        id="response_2",
        output=[
            SimpleNamespace(
                type="function_call",
                name="multiply",
                arguments='{"a": 12, "b": 3}',
                call_id="call_2",
            )
        ],
        output_text="",
    )

    third_response = SimpleNamespace(
        id="response_3",
        output=[],
        output_text="The answer is 36.",
    )

    fake_client.responses.create.side_effect = [
        first_response,
        second_response,
        third_response,
    ]

    mock_get_client.return_value = fake_client

    result = run_agent(
        "Add 5 and 7, then multiply the result by 3."
    )

    assert result == "The answer is 36."

    calls = fake_client.responses.create.call_args_list

    second_call = calls[1]
    third_call = calls[2]

    assert second_call.kwargs["input"][0]["output"] == "12"
    assert third_call.kwargs["input"][0]["output"] == "36"

@patch("agent.get_client")
def test_agent_iteration_limit(mock_get_client):

    fake_client = MagicMock()

    def fake_response(**kwargs):

        return SimpleNamespace(
            id="response_1",
            output=[
                SimpleNamespace(
                    type="function_call",
                    name="add",
                    arguments='{"a": 1, "b": 2}',
                    call_id="call_1",
                )
            ],
            output_text="",
        )

    fake_client.responses.create.side_effect = fake_response

    mock_get_client.return_value = fake_client

    with pytest.raises(RuntimeError):
        run_agent(
            "Keep adding numbers",
            max_iterations=3
        )

    assert fake_client.responses.create.call_count == 3

@patch("agent.get_client")
def test_agent_tool_limit(mock_get_client):
    fake_client = MagicMock()

    # Simulate the LLM requesting three tools simultaneously.
    fake_response = SimpleNamespace(
        id="response_1",
        output=[
            SimpleNamespace(
                type="function_call",
                name="add",
                arguments='{"a": 1, "b": 2}',
                call_id=f"call_{i}",
            )
            for i in range(3)
        ],
        output_text="",
    )

    fake_client.responses.create.return_value = fake_response
    mock_get_client.return_value = fake_client

    with pytest.raises(
        RuntimeError,
        match="Agent exceeded maximum tool executions"
    ):
        run_agent(
            "Perform three calculations",
            max_iterations=5,
            max_tool_calls=2,
        )

    assert fake_client.responses.create.call_count == 1
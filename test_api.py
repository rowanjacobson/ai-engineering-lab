import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from api import app, limiter


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok"
    }


def test_ask_requires_api_key():
    response = client.post(
        "/ask",
        json={
            "prompt": "Hello"
        }
    )

    assert response.status_code in (401, 403)


def test_ask_rejects_wrong_api_key():
    os.environ["APP_API_KEY"] = "correct-key"

    response = client.post(
        "/ask",
        headers={
            "X-API-Key": "wrong-key"
        },
        json={
            "prompt": "Hello"
        }
    )

    assert response.status_code == 401


@patch("api.ask_ai")
def test_ask_with_valid_api_key(mock_ask_ai):
    os.environ["APP_API_KEY"] = "correct-key"

    mock_ask_ai.return_value = (
        "Authentication checks who is allowed to access a system."
    )

    response = client.post(
        "/ask",
        headers={
            "X-API-Key": "correct-key"
        },
        json={
            "prompt": "Explain authentication"
        }
    )

    assert response.status_code == 200

    assert response.json() == {
        "answer":
            "Authentication checks who is allowed to access a system."
    }


@patch("api.ask_ai")
def test_rate_limit(mock_ask_ai):
    limiter.reset()

    os.environ["APP_API_KEY"] = "correct-key"

    mock_ask_ai.return_value = "ok"

    headers = {
        "X-API-Key": "correct-key"
    }

    for _ in range(10):
        response = client.post(
            "/ask",
            headers=headers,
            json={
                "prompt": "Hello"
            }
        )

        assert response.status_code == 200

    response = client.post(
        "/ask",
        headers=headers,
        json={
            "prompt": "Hello"
        }
    )

    assert response.status_code == 429


def test_agent_endpoint():
    os.environ["APP_API_KEY"] = "correct-key"

    mock_graph = MagicMock()

    mock_graph.invoke.return_value = {
        "messages": [
            SimpleNamespace(
                content="The answer is 36."
            )
        ]
    }

    app.state.agent_graph = mock_graph

    response = client.post(
        "/agent",
        headers={
            "X-API-Key": "correct-key"
        },
        json={
            "message":
                "Add 5 and 7, then multiply by 3.",
            "thread_id":
                "test-thread-1"
        }
    )

    assert response.status_code == 200

    assert response.json() == {
        "answer": "The answer is 36."
    }

    assert mock_graph.invoke.call_count == 1
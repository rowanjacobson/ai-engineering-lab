import json

from app import get_client
from config import OPENAI_MODEL


def multiply(a: float, b: float) -> float:
    return a * b


def add(a: float, b: float) -> float:
    return a + b


multiply_tool = {
    "type": "function",
    "name": "multiply",
    "description": "Multiply two numbers together.",
    "parameters": {
        "type": "object",
        "properties": {
            "a": {"type": "number"},
            "b": {"type": "number"},
        },
        "required": ["a", "b"],
        "additionalProperties": False,
    },
}


add_tool = {
    "type": "function",
    "name": "add",
    "description": "Add two numbers together.",
    "parameters": {
        "type": "object",
        "properties": {
            "a": {"type": "number"},
            "b": {"type": "number"},
        },
        "required": ["a", "b"],
        "additionalProperties": False,
    },
}


tools = [
    multiply_tool,
    add_tool,
]


tool_functions = {
    "multiply": multiply,
    "add": add,
}


def run_agent(prompt: str) -> str:
    client = get_client()

    response = client.responses.create(
        model=OPENAI_MODEL,
        input=prompt,
        tools=tools,
    )

    while True:
        tool_calls = [
            item for item in response.output
            if item.type == "function_call"
        ]

        if not tool_calls:
            return response.output_text

        tool_outputs = []

        for tool_call in tool_calls:
            arguments = json.loads(tool_call.arguments)

            tool_function = tool_functions[tool_call.name]

            result = tool_function(**arguments)

            tool_outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": tool_call.call_id,
                    "output": str(result),
                }
            )

        response = client.responses.create(
            model=OPENAI_MODEL,
            previous_response_id=response.id,
            input=tool_outputs,
            tools=tools,
        )
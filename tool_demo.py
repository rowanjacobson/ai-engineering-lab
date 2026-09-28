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


client = get_client()

response = client.responses.create(
    model=OPENAI_MODEL,
    input="Add 5 and 7, then multiply the result by 3. Use tools.",
    tools=tools,
)


while True:
    tool_calls = [
        item for item in response.output
        if item.type == "function_call"
    ]

    if not tool_calls:
        print(response.output_text)
        break

    tool_outputs = []

    for tool_call in tool_calls:
        arguments = json.loads(tool_call.arguments)

        tool_function = tool_functions[tool_call.name]

        result = tool_function(**arguments)

        print(
            f"Tool: {tool_call.name}, "
            f"Arguments: {arguments}, "
            f"Result: {result}"
        )

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
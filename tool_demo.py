def multiply(a: float, b: float) -> float:
    return a * b

multiply_tool = {
    "type": "function",
    "name": "multiply",
    "description": "Multiply two numbers together.",
    "parameters": {
        "type": "object",
        "properties": {
            "a": {"type": "number"},
            "b": {"type": "number"}
        },
        "required": ["a", "b"],
        "additionalProperties": False
    }
}

from app import get_client
from config import OPENAI_MODEL

client = get_client()

response = client.responses.create(
    model=OPENAI_MODEL,
    input="What is 17 multiplied by 23?",
    tools=[multiply_tool],
)

for item in response.output:
    print(item)


import json

tool_call = next(
    item for item in response.output
    if item.type == "function_call"
)

arguments = json.loads(tool_call.arguments)

result = multiply(
    arguments["a"],
    arguments["b"],
)

print("Tool result:", result)

final_response = client.responses.create(
    model=OPENAI_MODEL,
    previous_response_id=response.id,
    input=[
        {
            "type": "function_call_output",
            "call_id": tool_call.call_id,
            "output": str(result),
        }
    ],
    tools=[multiply_tool],
)

print(final_response.output_text)
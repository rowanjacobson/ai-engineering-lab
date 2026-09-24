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

tool_call = response.output[0]

arguments = json.loads(tool_call.arguments)

result = multiply(
    arguments["a"],
    arguments["b"],
)

print("Tool result:", result)
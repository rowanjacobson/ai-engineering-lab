from app import get_client
from config import OPENAI_MODEL
import json

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
            "b": {"type": "number"}
        },
        "required": ["a", "b"],
        "additionalProperties": False
    }
}

add_tool = {
    "type": "function",
    "name": "add",
    "description": "Add two numbers together.",
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

tools = [
    multiply_tool,
    add_tool,
]

client = get_client()

tool_functions = {
    "multiply": multiply,
    "add": add,
}

response = client.responses.create(
    model=OPENAI_MODEL,
    input="What is 17 plus 23? always use a tool never calculate the answer yourself",
    tools=tools,
)

for item in response.output:
    print(item)

tool_calls = [
    item for item in response.output
    if item.type == "function_call"
]


if not tool_calls:
    print(response.output_text)
else:
    tool_call = tool_calls[0]

arguments = json.loads(tool_call.arguments)


tool_function = tool_functions[tool_call.name]

result = tool_function(**arguments)

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
    tools=tools,
)

print(final_response.output_text)
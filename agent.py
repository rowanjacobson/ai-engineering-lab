import json

from app import get_client
from config import OPENAI_MODEL
from pydantic import BaseModel, ValidationError


def multiply(a: float, b: float) -> float:
    return a * b


def add(a: float, b: float) -> float:
    return a + b

class MultiplyArgs(BaseModel):
    a: float
    b: float


class AddArgs(BaseModel):
    a: float
    b: float

tool_argument_models = {
    "multiply": MultiplyArgs,
    "add": AddArgs,
}

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


def run_agent(prompt: str, max_iterations: int = 5,max_tool_calls: int = 10) -> str:

    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")

    if max_tool_calls < 1:
        raise ValueError("max_tool_calls must be at least 1")

    client = get_client()

    response = client.responses.create(
        model=OPENAI_MODEL,
        input=prompt,
        tools=tools,
    )

    iterations = 1
    tool_calls_executed = 0
    
    while True:

        tool_calls = [
            item for item in response.output
            if item.type == "function_call"
        ]

        # If no further tools are requested, return the answer.
        if not tool_calls:
            return response.output_text

        # Stop if another model request would exceed our limit.
        if iterations >= max_iterations:
            raise RuntimeError(
                "Agent exceeded maximum iterations"
            )

        tool_outputs = []

        for tool_call in tool_calls:

            if tool_calls_executed >= max_tool_calls:
                raise RuntimeError(
                    "Agent exceeded maximum tool executions"
                )

            arguments = json.loads(tool_call.arguments)

            tool_function = tool_functions.get(tool_call.name)
            argument_model = tool_argument_models.get(tool_call.name)

            if tool_function is None or argument_model is None:
                    raise ValueError(
                        f"Unknown tool requested: {tool_call.name}"
                    )
            try:
                    validated_arguments = argument_model(**arguments)

            except ValidationError as error:
                raise ValueError(
                    f"Invalid arguments for tool {tool_call.name}: {error}"
                )

            result = tool_function(
            **validated_arguments.model_dump()
            )

            tool_calls_executed += 1

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

        iterations += 1
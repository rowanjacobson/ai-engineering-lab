from langgraph.graph import StateGraph, START, END, MessagesState
from typing import TypedDict, List
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from config import OPENAI_MODEL
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver

llm = ChatOpenAI(
    model=OPENAI_MODEL,
    reasoning_effort="none"
)

@tool
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

tools = [add]

llm_with_tools = llm.bind_tools(tools)

tool_node = ToolNode(tools)

tools = [add]
tool_node = ToolNode(tools)

def llm_node(state: MessagesState):
    response = llm_with_tools.invoke(state["messages"])

    return {
        "messages": [response]
    }

def should_continue(state: MessagesState):
    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tools"

    return END

builder = StateGraph(MessagesState)

builder.add_node("llm", llm_node)
builder.add_node("tools", tool_node)

builder.add_edge(START, "llm")

builder.add_conditional_edges(
    "llm",
    should_continue,
)

builder.add_edge("tools", "llm")

memory = InMemorySaver()

graph = builder.compile(checkpointer=memory)

config = {
    "configurable": {
        "thread_id": "conversation-1"
    }
}

result = graph.invoke(
    {
        "messages": [
            HumanMessage(content="My name is Rowan.")
        ]
    },
    config=config,
)

print(result["messages"][-1].content)


result = graph.invoke(
    {
        "messages": [
            HumanMessage(content="What is my name?")
        ]
    },
    config=config,
)

print(result["messages"][-1].content)

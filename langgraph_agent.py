from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.postgres import PostgresSaver

from config import OPENAI_MODEL, DATABASE_URL


# --------------------------------------------------
# 1. Define tools
# --------------------------------------------------

@tool
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b


tools = [add]

tool_node = ToolNode(tools)


# --------------------------------------------------
# 2. Create the LLM only when needed
# --------------------------------------------------

def get_llm_with_tools():
    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        reasoning_effort="none",
    )

    return llm.bind_tools(tools)


# --------------------------------------------------
# 3. Define graph nodes
# --------------------------------------------------

def llm_node(state: MessagesState):
    llm_with_tools = get_llm_with_tools()

    response = llm_with_tools.invoke(
        state["messages"]
    )

    return {
        "messages": [response]
    }


# --------------------------------------------------
# 4. Decide whether to use a tool or finish
# --------------------------------------------------

def should_continue(state: MessagesState):
    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tools"

    return END


# --------------------------------------------------
# 5. Build graph structure
# --------------------------------------------------

builder = StateGraph(MessagesState)

builder.add_node(
    "llm",
    llm_node,
)

builder.add_node(
    "tools",
    tool_node,
)

builder.add_edge(
    START,
    "llm",
)

builder.add_conditional_edges(
    "llm",
    should_continue,
)

builder.add_edge(
    "tools",
    "llm",
)


# --------------------------------------------------
# 6. Reusable agent function
# --------------------------------------------------

def run_langgraph_agent(
    message: str,
    thread_id: str,
) -> str:

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    with PostgresSaver.from_conn_string(
        DATABASE_URL
    ) as memory:

        # Create LangGraph checkpoint tables if needed.
        memory.setup()

        graph = builder.compile(
            checkpointer=memory
        )

        result = graph.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=message
                    )
                ]
            },
            config=config,
        )

    return result["messages"][-1].content
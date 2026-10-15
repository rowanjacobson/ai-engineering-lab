from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.postgres import PostgresSaver

from config import OPENAI_MODEL, DATABASE_URL


# --------------------------------------------------
# 1. Configure the model
# --------------------------------------------------

llm = ChatOpenAI(
    model=OPENAI_MODEL,
    reasoning_effort="none",
)


# --------------------------------------------------
# 2. Define tools
# --------------------------------------------------

@tool
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b


tools = [add]

llm_with_tools = llm.bind_tools(tools)

tool_node = ToolNode(tools)


# --------------------------------------------------
# 3. Define the LLM node
# --------------------------------------------------

def llm_node(state: MessagesState):
    response = llm_with_tools.invoke(
        state["messages"]
    )

    return {
        "messages": [response]
    }


# --------------------------------------------------
# 4. Decide whether to call tools or finish
# --------------------------------------------------

def should_continue(state: MessagesState):
    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tools"

    return END


# --------------------------------------------------
# 5. Build the graph structure
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
# 6. Reusable LangGraph agent function
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

        # Creates the LangGraph checkpoint tables if needed.
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
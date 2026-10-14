from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from config import OPENAI_MODEL
from langgraph.checkpoint.postgres import PostgresSaver
from config import OPENAI_MODEL, DATABASE_URL


# --------------------------------------------------
# 1. Configure model
# --------------------------------------------------

llm = ChatOpenAI(
    model=OPENAI_MODEL,
    reasoning_effort="none"
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
# 3. Define graph nodes
# --------------------------------------------------

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


# --------------------------------------------------
# 4. Build graph
# --------------------------------------------------

builder = StateGraph(MessagesState)

builder.add_node("llm", llm_node)
builder.add_node("tools", tool_node)

builder.add_edge(START, "llm")

builder.add_conditional_edges(
    "llm",
    should_continue,
)

builder.add_edge("tools", "llm")


# --------------------------------------------------
# 5. Persistent Postgres memory
# --------------------------------------------------

memory = PostgresSaver.from_conn_string(DATABASE_URL)

memory.setup()

graph = builder.compile(
    checkpointer=memory
)

# --------------------------------------------------
# 6. Conversation configuration
# --------------------------------------------------

config = {
    "configurable": {
        "thread_id": "sqlite-test-1"
    }
}


# --------------------------------------------------
# 7. Run conversation
# --------------------------------------------------

result = graph.invoke(
    {
        "messages": [
            HumanMessage(
                content="what is my favourite animal"
            )
        ]
    },
    config=config,
)

print(result["messages"][-1].content)
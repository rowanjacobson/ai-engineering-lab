import logging
import os
import secrets
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import APIKeyHeader
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres import PostgresSaver
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app import ask_ai
from config import DATABASE_URL, LOG_LEVEL, RATE_LIMIT
from langgraph_agent import build_graph


# --------------------------------------------------
# 1. Logging
# --------------------------------------------------

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# 2. Authentication
# --------------------------------------------------

api_key_header = APIKeyHeader(name="X-API-Key")


def verify_api_key(
    api_key: str = Depends(api_key_header),
):
    expected_key = os.getenv("APP_API_KEY")

    if not expected_key:
        raise RuntimeError(
            "APP_API_KEY is not configured"
        )

    if not secrets.compare_digest(
        api_key,
        expected_key,
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
        )


# --------------------------------------------------
# 3. Rate limiting
# --------------------------------------------------

limiter = Limiter(
    key_func=get_remote_address
)


# --------------------------------------------------
# 4. FastAPI startup / shutdown lifecycle
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    with PostgresSaver.from_conn_string(
        DATABASE_URL
    ) as memory:

        # Create the LangGraph checkpoint tables
        # if they do not already exist.
        memory.setup()

        # Compile the graph once when FastAPI starts.
        app.state.agent_graph = build_graph(
            memory
        )

        # FastAPI runs normally while execution
        # is paused here.
        yield

    # When FastAPI shuts down, leaving the
    # "with" block closes the Postgres connection.


# --------------------------------------------------
# 5. Create the FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="AI Engineering Lab",
    lifespan=lifespan,
)

app.state.limiter = limiter

app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler,
)


# --------------------------------------------------
# 6. Request body models
# --------------------------------------------------

class AskRequest(BaseModel):
    prompt: str


class AgentRequest(BaseModel):
    message: str
    thread_id: str


# --------------------------------------------------
# 7. Health endpoint
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok"
    }


# --------------------------------------------------
# 8. Basic OpenAI endpoint
# --------------------------------------------------

@app.post(
    "/ask",
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit(RATE_LIMIT)
def ask(
    request: Request,
    body: AskRequest,
):
    answer = ask_ai(
        body.prompt
    )

    return {
        "answer": answer
    }


# --------------------------------------------------
# 9. LangGraph agent endpoint
# --------------------------------------------------

@app.post(
    "/agent",
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit(RATE_LIMIT)
def agent_endpoint(
    request: Request,
    body: AgentRequest,
):
    config = {
        "configurable": {
            "thread_id": body.thread_id
        }
    }

    result = (
        request.app.state.agent_graph.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=body.message
                    )
                ]
            },
            config=config,
        )
    )

    return {
        "answer": result["messages"][-1].content
    }
import os
import secrets

from fastapi import Depends, FastAPI, HTTPException, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from app import ask_ai

app = FastAPI()

api_key_header = APIKeyHeader(name="X-API-Key")

limiter = Limiter(key_func=get_remote_address)

app = FastAPI()

app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler
)

class AskRequest(BaseModel):
    prompt: str


def verify_api_key(api_key: str = Depends(api_key_header)):
    expected_key = os.getenv("APP_API_KEY")

    if not expected_key:
        raise RuntimeError("APP_API_KEY is not configured")

    if not secrets.compare_digest(api_key, expected_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask", dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
def ask(request: Request, body: AskRequest):
    answer = ask_ai(body.prompt)

    return {
        "answer": answer
    }
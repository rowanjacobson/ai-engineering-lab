import os
import secrets

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from app import ask_ai

app = FastAPI()

api_key_header = APIKeyHeader(name="X-API-Key")


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
def ask(request: AskRequest):
    answer = ask_ai(request.prompt)

    return {
        "answer": answer
    }
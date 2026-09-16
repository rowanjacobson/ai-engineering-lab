from fastapi import FastAPI
from pydantic import BaseModel

from app import ask_ai

app = FastAPI()


class AskRequest(BaseModel):
    prompt: str


@app.post("/ask")
def ask(request: AskRequest):
    answer = ask_ai(request.prompt)

    return {
        "answer": answer
    }

@app.get("/health")
def health():
    return {"status": "ok"}
from typing import Literal

from pydantic import BaseModel

from app import get_client
from config import OPENAI_MODEL


class SupportTicket(BaseModel):
    category: Literal["billing", "technical", "account", "other"]
    priority: Literal["low", "medium", "high"]
    summary: str


def classify_ticket(text: str) -> SupportTicket:
    client = get_client()

    response = client.responses.parse(
        model=OPENAI_MODEL,
        input=text,
        text_format=SupportTicket,
    )

    return response.output_parsed
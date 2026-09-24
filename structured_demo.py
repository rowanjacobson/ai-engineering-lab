from typing import Literal

from pydantic import BaseModel

from app import get_client
from config import OPENAI_MODEL


class SupportTicket(BaseModel):
    category: Literal["billing", "technical", "account", "other"]
    priority: Literal["low", "medium", "high"]
    summary: str


client = get_client()

response = client.responses.parse(
    model=OPENAI_MODEL,
    input="I forgot my password and can't log in.",
    text_format=SupportTicket,
)

ticket = response.output_parsed

print(ticket)
print(ticket.category)
print(ticket.priority)
print(ticket.summary)
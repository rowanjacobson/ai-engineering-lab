import os

from dotenv import load_dotenv
from openai import OpenAI

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)

load_dotenv()

logger = logging.getLogger(__name__)

def get_client():
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing from the environment")

    return OpenAI(api_key=api_key)


def ask_ai(prompt):
    if not prompt.strip():
        return "Please enter a question."

    try:
        client = get_client()

        response = client.responses.create(
            model="gpt-5.6-luna",
            input=prompt
        )

        return response.output_text

    except Exception:
            logger.exception("OpenAI request failed")
            return "The AI service is temporarily unavailable."


def main():
    user_question = input("Ask the AI something: ")
    answer = ask_ai(user_question)
    print(answer)


if __name__ == "__main__":
    main()
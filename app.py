import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY is missing from the .env file")

client = OpenAI(api_key=api_key)


def ask_ai(prompt):
    if not prompt.strip():
        return "Please enter a question."

    try:
        response = client.responses.create(
            model="gpt-5.6-luna",
            input=prompt
        )
        return response.output_text

    except Exception as error:
        return f"Something went wrong: {error}"

user_question = input("Ask the AI something: ")

answer = ask_ai(user_question)

print(answer)
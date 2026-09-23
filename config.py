import os

from dotenv import load_dotenv

load_dotenv()


APP_ENV = os.getenv("APP_ENV", "development")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
RATE_LIMIT = os.getenv("RATE_LIMIT", "10/minute")
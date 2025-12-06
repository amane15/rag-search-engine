import os

from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("gemini_api_key")
client = genai.Client(api_key=api_key)
model = "gemini-2.0-flash"


def ask_gemini(prompt: str):
    response = client.models.generate_content(model=model, contents=prompt)
    return response

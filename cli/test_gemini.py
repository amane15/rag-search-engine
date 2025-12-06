import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)

# response = client.models.generate_content(
#     model="gemini-2.0-flash-001",
#     contents="Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum",
# )
#
# print(response.text)
# print("Prompt Tokens:", response.usage_metadata.prompt_token_count)
# print("Response Tokens:", response.usage_metadata.candidates_token_count)

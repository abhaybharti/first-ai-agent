import os
from dotenv import load_dotenv

load_dotenv()

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

if not HUGGINGFACE_API_KEY:
    raise ValueError("HUGGINGFACE_API_KEY is not configured")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# print("GOOGLE_API_KEY:", GOOGLE_API_KEY)
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not configured")
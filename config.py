import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load from .env file (optional but recommended)
load_dotenv()

# Google Gemini API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Where Chroma DB will be stored
CHROMA_DB_DIR = "my_chroma_db"
CHROMA_COLLECTION = "sample"

# Create Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    google_api_key=os.environ["GOOGLE_API_KEY"]  # you already set this earlier
)

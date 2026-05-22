import os
# from groq import Groq
from openai import OpenAI
from langfuse import Langfuse
from pathlib import Path
from dotenv import load_dotenv

from src.utils.logger import logger

env_path = Path(os.getcwd()) / ".env"
load_dotenv(dotenv_path=env_path, override=True)

PINECONE_API_KEY         = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME      = os.getenv("PINECONE_INDEX_NAME", "enterprise-docs")
PINECONE_CACHE_NAMESPACE = 'cache'

LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST")

logger.info('Initializing Langfuse client...')
langfuse_client = Langfuse(
    public_key=LANGFUSE_PUBLIC_KEY,
    secret_key=LANGFUSE_SECRET_KEY,
    host=LANGFUSE_HOST
)


HF_TOKEN = os.getenv("HF_TOKEN", "")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print(GROQ_API_KEY)
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL")
GROQ_GENERATION_MODEL = os.getenv("GROQ_GENERATION_MODEL")  

openai_client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url=GROQ_BASE_URL
)
# groq_client = Groq(api_key=GROQ_API_KEY)#, base_url=GROQ_BASE_URL)
# print(groq_client)
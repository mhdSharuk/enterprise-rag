# import os
# from dotenv import load_dotenv

# load_dotenv()

import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(os.getcwd()) / ".env"
load_dotenv(dotenv_path=env_path, override=True)


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "enterprise-docs")

HF_TOKEN = os.getenv("HF_TOKEN", "")

GENERATION_MODEL_ID = os.getenv("GENERATION_MODEL")
GENERATION_ONNX_FILE = os.getenv("GENERATION_ONNX_FILE")

MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 1000))
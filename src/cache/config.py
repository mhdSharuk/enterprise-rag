import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(os.getcwd()) / ".env"
load_dotenv(dotenv_path=env_path, override=True)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE")

HF_TOKEN = os.getenv("HF_TOKEN")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
EMBEDDING_ONNX_FILE = os.getenv('EMBEDDING_ONNX_FILE')

GLINER_MODEL = "urchade/gliner_mediumv2.1"
CACHE_CONFIDENCE_THRESHOLD = 0.9

ENTITY_TYPES = [
    "FEATURE", "METRIC", "LIMIT", "COMPONENT",
    "STATUS", "VALUE", "UNIT", "CONDITION"
]

MODEL_STORE = Path("model_store")
GLINER_PATH = MODEL_STORE / GLINER_MODEL.replace("/", "_")
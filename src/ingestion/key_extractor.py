import re
import json
import torch
from src.ingestion.prompts import KEY_EXTRACTION_PROMPT

FALLBACK_KEYS = {
    "embedding_keys": ["title", "content"],
    "metadata_keys": ["owner", "created_at", "last_modified", "status", "labels"],
    "hybrid_keys": []
}


def extract_keys_from_schema(json_data: dict) -> dict:
    embedding_keys = [json_data["title_field_name"]] + json_data["content_field_names"]
    metadata_keys = [k for k in json_data.keys() if k not in embedding_keys]
    return {
        "embedding_keys": embedding_keys,
        "metadata_keys": metadata_keys,
        "hybrid_keys": []
    }

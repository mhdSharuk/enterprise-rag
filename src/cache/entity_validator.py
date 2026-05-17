from pathlib import Path
from gliner import GLiNER
from src.utils.logger import logger
from src.cache.config import GLINER_MODEL, HF_TOKEN, MODEL_STORE, GLINER_PATH, ENTITY_TYPES

gliner_tokenizer = None
gliner_model = None

def load_gliner_model():
    global gliner_tokenizer, gliner_model

    if gliner_model is not None:
        return gliner_model

    if GLINER_PATH.exists():
        logger.info(f"Loading GLiNER model from {GLINER_PATH}")
        gliner_model = GLiNER.from_pretrained(str(GLINER_PATH))
        return gliner_model

    logger.info(f"Downloading GLiNER model {GLINER_MODEL} ...")
    MODEL_STORE.mkdir(parents=True, exist_ok=True)

    gliner_model = GLiNER.from_pretrained(GLINER_MODEL)
    gliner_model.save_pretrained(str(GLINER_PATH))

    logger.info(f"Saved GLiNER model to {GLINER_PATH}")
    return gliner_model

def extract_entities(query: str, model=None) -> list[dict]:
    if model is None:
        model = load_gliner_model()

    entities = model.predict_entities(query, ENTITY_TYPES)

    result = []
    for entity in entities:
        result.append({
            "text": entity["text"],
            "label": entity["label"],
            "score": entity["score"]
        })

    return result

def compare_entities(current_entities: list[dict], cached_entities: list[dict]) -> bool:
    if not current_entities and not cached_entities:
        return True

    if len(current_entities) != len(cached_entities):
        return False

    current_set = set()
    for e in current_entities:
        current_set.add((e["text"].lower(), e["label"]))

    cached_set = set()
    for e in cached_entities:
        cached_set.add((e["text"].lower(), e["label"]))

    return current_set == cached_set
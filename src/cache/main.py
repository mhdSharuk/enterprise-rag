from src.utils.logger import logger
from src.retrieval.embedder import get_dense_embedding
from src.cache.cache_store import store_in_vectorstore
from src.cache.semantic_match import check_semantic_match
from src.cache.entity_validator import extract_entities, compare_entities, load_gliner_model

from src.cache.config import CACHE_CONFIDENCE_THRESHOLD


def check_cache(query, index, 
                embedding_tokenizer, embedding_model, 
                threshold = CACHE_CONFIDENCE_THRESHOLD) -> tuple[bool, str | None]:
    
    semantic_result = check_semantic_match(index, query, embedding_tokenizer, 
                                           embedding_model, threshold)

    if semantic_result:
        logger.info(f"Cache hit: semantic match (score: {semantic_result['score']:.3f})")

        gliner_model = load_gliner_model()
        current_entities = extract_entities(query, gliner_model)
        cached_entities = semantic_result["entities"]

        if compare_entities(current_entities, cached_entities):
            logger.info("Entity validation passed")
            return True, semantic_result["response"]
        else:
            logger.info("Entity validation failed")

    logger.info("Cache miss")
    return False, None

def store_in_cache(index, query: str, response: str, embedding_tokenizer, embedding_model):

    embedding = get_dense_embedding(embedding_tokenizer, embedding_model, query)

    gliner_model = load_gliner_model()
    entities = extract_entities(query, gliner_model)

    store_in_vectorstore(index, query, response, entities, embedding)
    logger.info(f"Stored in cache: {query[:50]}...")
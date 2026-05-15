from langfuse import Langfuse

from src.utils.logger import logger
from src.ingestion.config import (LANGFUSE_BASE_URL, LANGFUSE_PUBLIC_KEY, 
                                  LANGFUSE_SECRET_KEY, KEY_EXTRACTION_PROMPT_NAME)

langfuse = Langfuse(
    public_key = LANGFUSE_PUBLIC_KEY,
    secret_key = LANGFUSE_SECRET_KEY,
    host = LANGFUSE_BASE_URL
)

KEY_EXTRACTION_PROMPT = None

if langfuse.auth_check():
    logger.info("Langfuse is authenticated!")

try:
    prompt_obj = langfuse.get_prompt(KEY_EXTRACTION_PROMPT_NAME, label='production')
    KEY_EXTRACTION_PROMPT = prompt_obj.prompt
    
except Exception as err:
    logger.error(f'Error occured')
    logger.error(err)

# prompt_obj = langfuse.get_prompt('markdown_conversion_prompt', label='production')
# MARKDOWN_CONVERSION_PROMPT = prompt_obj.prompt
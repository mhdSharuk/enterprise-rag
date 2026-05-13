from langfuse import Langfuse

langfuse = Langfuse()

prompt_obj = langfuse.get_prompt('key_extraction_prompt', label='production')
KEY_EXTRACTION_PROMPT = prompt_obj.prompt

prompt_obj = langfuse.get_prompt('markdown_conversion_prompt', label='production')
MARKDOWN_CONVERSION_PROMPT = prompt_obj.prompt
import os
from src.generation.config import langfuse_client, groq_client, GROQ_GENERATION_MODEL

GENERATION_SYSTEM_PROMPT = langfuse_client.get_prompt("generation_system_prompt", label="production").prompt

def build_context_block(query: str, merged_docs: list[dict]) -> str:
    context = f"Context:"
    for doc in merged_docs:
        source = doc.get("id", "")
        text = doc.get("text", "")
        chunk_range = doc.get("chunk_range")
        label = f"Document: {source} (chunk {chunk_range})" if chunk_range else f"Document: {source}"
        context += f"\n{label}\n{text}\n{'=' * 20}\n"

    context += f"Query : {query}"
    return context

def generate_response(query: str, merged_docs: list[dict]) -> str:
    context = build_context_block(query, merged_docs)

    response = groq_client.chat.completions.create(
      model=GROQ_GENERATION_MODEL,
      messages=[
        {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
        {"role": "user", "content": context}
      ],

      temperature=0,
      max_completion_tokens=8192,
      reasoning_effort="medium",
      stream=False,
      stop=None

    )

    generated_text = response.choices[0].message.content
    total_tokens = response.usage.total_tokens
    finish_reason = response.choices[0].finish_reason

    return generated_text, total_tokens, finish_reason
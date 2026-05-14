from threading import Thread
from transformers import TextIteratorStreamer
from src.generation.prompt_templates import SYSTEM_PROMPT, build_context_block
from src.generation.config import MAX_NEW_TOKENS

def generate(query: str, merged_docs: list[dict], tokenizer, model, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    context = build_context_block(query, merged_docs)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": context}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    generated = ""
    for token in streamer:
        print(token, end="", flush=True)
        generated += token

    thread.join()
    print()

    return generated
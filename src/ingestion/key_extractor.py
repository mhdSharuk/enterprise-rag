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

def extract_keys_with_llm(json_data: dict, tokenizer, model, streamer) -> dict:
    torch.cuda.empty_cache()

    json_text = json.dumps(json_data, ensure_ascii=True)
    content = f"{KEY_EXTRACTION_PROMPT}\n\nExtract keys from this JSON:\n{json_text}"

    messages = [{"role": "user", "content": content}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    dynamic_max = min(1000, len(tokenizer(json_text)["input_ids"]))

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=dynamic_max,
            streamer=streamer,
            do_sample=False,
            temperature=0,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
        )

    input_length = inputs["input_ids"].shape[1]
    raw = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

    del inputs, outputs
    torch.cuda.empty_cache()

    try:
        extracted_keys = json.loads(re.search(r"\{.*\}", raw, re.DOTALL).group())
        return extracted_keys
    
    except Exception:
        return FALLBACK_KEYS
import re
from tqdm import tqdm

def merge_ranked_chunks(result) -> list[dict]:
    docs = [
        {
            "id": d["id"],
            "score": d["score"],
            "text": d.get("chunk_text", ""),
            "base_id": d["id"].rsplit("_chunk", 1)[0],
            "chunk_num": int(d["id"].rsplit("_", 1)[-1])
        }
        for d in result
    ]

    grouped = {}
    for doc in docs:
        grouped.setdefault(doc["base_id"], []).append(doc)

    for base_id in grouped:
        grouped[base_id].sort(key=lambda x: x["chunk_num"])

    merged_groups = []
    for base_id, chunks in grouped.items():
        current_group = [chunks[0]]
        for i in range(1, len(chunks)):
            if chunks[i]["chunk_num"] - chunks[i - 1]["chunk_num"] <= 1:
                current_group.append(chunks[i])
            else:
                merged_groups.append(current_group)
                current_group = [chunks[i]]
        merged_groups.append(current_group)

    merged = []
    for group in merged_groups:
        merged.append({
            "id": group[0]["base_id"],
            "doc_id": group[0]["base_id"],
            "chunk_range": (group[0]["chunk_num"], group[-1]["chunk_num"]),
            "score": max(c["score"] for c in group),
            "text": " ".join(c["text"] for c in group)
        })

    return sorted(merged, key=lambda x: x["score"], reverse=True)

def get_neighbour_chunk_ids(rerank_result_data: list[dict]) -> set[str]:
    retrieved_ids = {d["id"] for d in rerank_result_data}

    neighbour_ids = set()
    for chunk_id in retrieved_ids:
        match = re.match(r"(.+)_chunk_(\d+)$", chunk_id)
        if not match:
            continue

        prefix, num_str = match.groups()
        num = int(num_str)
        pad = len(num_str)

        if num > 0:
            neighbour_ids.add(f"{prefix}_chunk_{str(num - 1).zfill(pad)}")

        neighbour_ids.add(chunk_id)
        neighbour_ids.add(f"{prefix}_chunk_{str(num + 1).zfill(pad)}")

    return neighbour_ids
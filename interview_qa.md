# Enterprise RAG — Interview Q&A

---

## Section 1: RAG Fundamentals

---

<question>
What is RAG and why did you build it instead of just fine-tuning an LLM?
</question>

<answer>
RAG stands for Retrieval-Augmented Generation. The core idea is to give an LLM access to a dynamic, searchable knowledge base at query time rather than baking knowledge into the model's weights.

I chose RAG over fine-tuning for a few strong reasons:
- **Knowledge freshness** — enterprise data changes constantly. Fine-tuning is a snapshot; RAG lets you update the knowledge base without retraining anything.
- **Transparency** — RAG can cite sources. The model tells you exactly which document it answered from. Fine-tuned models can't do that reliably.
- **Cost** — fine-tuning a large model is expensive and time-consuming. With RAG, you only update your vector index when data changes.
- **Hallucination reduction** — by grounding the LLM with retrieved context, you constrain it to what's actually in your documents rather than what it "remembers."
- **Access control** — in an enterprise setting, different users should see different data. RAG lets you filter at retrieval time. Fine-tuning has no concept of per-user access.
</answer>

---

<question>
What are the limitations of RAG?
</question>

<answer>
RAG is powerful but it has real limitations I had to design around:

- **Retrieval ceiling** — if the right chunk isn't retrieved, the LLM can't answer correctly no matter how good it is. Garbage in, garbage out.
- **Context window limits** — you can only pass so many chunks to the LLM. If the answer spans many documents, you might miss parts.
- **Chunking sensitivity** — if a chunk cuts an answer in half, retrieval scores drop and the answer quality suffers. This is why I implemented adjacent chunk merging.
- **Latency** — there are multiple hops: embed the query, hit the vector DB, rerank, then call the LLM. Each adds latency. I mitigated this with semantic caching.
- **Semantic gap** — dense embeddings don't always capture exact keyword matches. A query for a specific ID or code might score poorly against dense embeddings. That's why I use hybrid (dense + sparse) search.
- **Multi-hop reasoning** — if answering a question requires connecting information from two unrelated documents, RAG struggles. It retrieves, it doesn't reason across documents.
</answer>

---

<question>
What is the difference between naive RAG and what you built?
</question>

<answer>
Naive RAG is basically: embed the query, do a cosine similarity search, stuff the top-k results into the LLM prompt. It works for demos but breaks in production.

What I built goes significantly further:

- **Hybrid search** instead of pure dense — I combine dense (semantic) with sparse (keyword/SPLADE) embeddings. Naive RAG misses exact keyword matches; mine doesn't.
- **Cross-encoder reranking** — I take the top results and re-score them with a more powerful cross-encoder model. Bi-encoder retrieval is fast but approximate; reranking improves precision significantly.
- **Adjacent chunk merging** — when two consecutive chunks from the same document are retrieved, I merge them into one coherent passage instead of passing fragmented text to the LLM.
- **Semantic caching** — identical or near-identical queries are served from cache without hitting the LLM. This cuts latency and cost dramatically for repeated questions.
- **User-level access control** — each vector has a `user_access` metadata field. Every query is filtered so users only retrieve documents they're authorized to see.
- **Observability** — prompts are managed in Langfuse, evaluation scores are tracked, and everything is traceable.
</answer>

---

## Section 2: Hybrid Search

---

<question>
What is hybrid search and why do you use it?
</question>

<answer>
Hybrid search combines two fundamentally different retrieval signals:

- **Dense retrieval** — uses a neural embedding model (like BGE or E5) to map text into a high-dimensional vector space. Similar meanings cluster together. Great for semantic queries like "what's the leave policy" even if the document says "annual time off rules."
- **Sparse retrieval** — uses term-frequency-based scoring (like BM25 or SPLADE). Great for exact keyword matches, product codes, names, specific IDs.

Neither alone is sufficient:
- Dense-only fails on exact keyword queries — "what is contract number CT-2024-89?" scores poorly if no document uses similar wording.
- Sparse-only fails on semantic queries — it can't understand that "show me all HR policies" and "employee guidelines" are related.

Hybrid search gets the best of both. In my implementation I use a configurable alpha (`HYBRID_ALPHA = 0.5`) to weight between the two. The final score is `alpha * dense_score + (1 - alpha) * sparse_score`.
</answer>

---

<question>
What is SPLADE and why did you choose it over BM25 for sparse embeddings?
</question>

<answer>
SPLADE (Sparse Lexical and Expansion Model) is a learned sparse retrieval model. Unlike BM25 which is purely statistical, SPLADE is a transformer-based model that produces sparse vectors over the vocabulary.

Why SPLADE over BM25:
- **Query expansion** — SPLADE implicitly expands queries. If you search for "car," it also activates tokens like "vehicle," "automobile." BM25 only matches exact tokens.
- **Learned weights** — token weights in SPLADE are learned from relevance data, not just term frequency. More important terms get higher weights.
- **Same interface as dense vectors** — SPLADE outputs a sparse vector (indices + values), which Pinecone supports natively. BM25 would require a separate search index.
- **Still interpretable** — unlike dense embeddings which are opaque, SPLADE vectors show which vocabulary terms drove the match.

The trade-off is that SPLADE is heavier than BM25 at inference time, which is why I export it to ONNX for faster CPU inference.
</answer>

---

<question>
How does the hybrid alpha parameter work and how did you choose 0.5?
</question>

<answer>
Alpha is the interpolation weight between dense and sparse scores. In my `hybrid_score_norm` function:
- `h_dense = dense_vector * alpha`
- `h_sparse_values = sparse_values * (1 - alpha)`

So `alpha = 1.0` means pure dense retrieval, `alpha = 0.0` means pure sparse.

I chose 0.5 as the default because:
- It's a neutral starting point when you don't have domain-specific tuning data.
- The evaluation results (Recall@5: 0.886, MRR: 0.880) confirm the system performs well at this balance.

In practice, the right alpha depends on the query distribution:
- If your users ask mostly semantic questions → push alpha higher (0.6–0.7).
- If your users ask a lot of exact-match queries (IDs, names, codes) → push alpha lower (0.3–0.4).
- You'd ideally tune this on a labeled query set.
</answer>

---

## Section 3: Embeddings & ONNX

---

<question>
Why did you export models to ONNX instead of running them natively with PyTorch?
</question>

<answer>
ONNX (Open Neural Network Exchange) is a runtime-optimized format for model inference. The reasons I use it:

- **Speed** — ONNX Runtime with graph optimizations is consistently faster than PyTorch eager mode for inference. No autograd overhead, fused kernels, optimized memory layout.
- **Lower memory** — ONNX models can be quantized (INT8) reducing memory footprint by 3–4x. Relevant when loading three models (dense embedder, SPLADE, reranker) simultaneously.
- **CPU-friendly** — ONNX Runtime's CPUExecutionProvider is highly optimized for inference without a GPU. My app can run on CPU-only machines like cloud VMs without CUDA.
- **Provider flexibility** — `get_onnx_provider()` in my code auto-detects CUDA. If a GPU is available, it uses `CUDAExecutionProvider`; otherwise falls back to CPU. Zero code changes needed.
- **Consistent environment** — ONNX decouples the model from the training framework. You don't need a specific PyTorch version to run inference.

The trade-off is model export complexity — you need `optimum` from HuggingFace to export properly, and not all architectures export cleanly.
</answer>

---

<question>
What is mean pooling and why do you use it for dense embeddings?
</question>

<answer>
Mean pooling is how I convert a sequence of token embeddings (one vector per token) into a single document-level vector.

The process:
- The transformer outputs a tensor of shape `[batch, sequence_length, hidden_size]` — one vector per token.
- I take the weighted average across the sequence dimension, using the attention mask so padding tokens contribute zero weight.
- The result is one vector of `hidden_size` dimensions representing the whole input.

Why mean pooling over alternatives:
- **[CLS] token pooling** — using just the first token vector is simpler but performs worse on similarity tasks because CLS is trained for classification, not retrieval.
- **Max pooling** — captures the most activated feature per dimension but loses distributional information.
- **Mean pooling** — empirically the best for sentence similarity and retrieval. Models like BGE are specifically trained with mean pooling objectives.

After pooling I L2-normalize the vector so dot product equals cosine similarity, which is what Pinecone uses for scoring.
</answer>

---

<question>
How does the SPLADE inference work in your code?
</question>

<answer>
SPLADE uses a masked language model (MLM) backbone. Here's the inference pipeline in my code:

- Tokenize the input text using the SPLADE tokenizer.
- Run it through the ONNX session (`ort.InferenceSession`) to get logits of shape `[batch, seq_len, vocab_size]`.
- Apply `log(1 + ReLU(logits))` — the log-saturation activation from the SPLADE paper that produces the sparse vocabulary distribution.
- Take the max across the sequence dimension to get a single vocabulary-size vector.
- Extract non-zero indices and their values as the sparse representation.

The key insight:
- A regular transformer outputs one vector per token — that's dense.
- SPLADE outputs a score for every vocabulary token — that's sparse.
- Most vocabulary tokens score near zero, giving you natural sparsity.
- The indices are actual token IDs, making it interpretable.

This `(indices, values)` format maps directly to what Pinecone expects for its `sparse_values` field.
</answer>

---

## Section 4: Semantic Caching

---

<question>
What is semantic caching and how does it differ from a regular key-value cache like Redis?
</question>

<answer>
A regular cache (Redis, Memcached) does exact-match lookup — the key must match exactly. If a user asks "what is the leave policy?" and previously you cached "what are leave policies?", a regular cache misses.

Semantic caching uses vector similarity to find cache hits:
- When I get an answer, I store the query embedding, the answer, and the source documents in a dedicated Pinecone namespace.
- On the next query, I first embed the new query and search the cache namespace for semantically similar past queries.
- I then rerank the candidates with the cross-encoder to verify genuine semantic match.
- If the top result scores above the threshold (0.9), I return the cached answer immediately.

The benefits:
- Handles paraphrased questions — same question asked differently still hits the cache.
- Cuts LLM costs dramatically for repeated or similar queries.
- Reduces latency from ~3–5 seconds to under 200ms for cache hits.

The threshold of 0.9 is intentionally high — I'd rather miss a cache hit than return the wrong answer.
</answer>

---

<question>
How do you handle cache invalidation when documents are updated?
</question>

<answer>
Honestly, this is a known limitation in the current implementation. The semantic cache doesn't have automatic invalidation.

What the current system does:
- Cache entries are stored with the query hash as the vector ID.
- There's no TTL (time-to-live) or document version tracking.
- The cache entry includes `source_deps` (which documents the answer came from), but this isn't checked on retrieval.

How I'd improve it:
- **TTL-based expiry** — add a timestamp to cache metadata and filter out entries older than N days during lookup.
- **Source-hash invalidation** — when a document is re-ingested, compute its hash. If the hash changes, delete all cache entries whose `source_deps` include that document ID.
- **Namespace clear** — for full refresh scenarios, just clear the cache namespace in Pinecone and let it rebuild naturally.

For the current use case (relatively stable enterprise documents), the absence of invalidation is acceptable. But for frequently-changing data, this would need to be addressed.
</answer>

---

<question>
Why do you use the reranker for cache lookup instead of just the cosine similarity score from Pinecone?
</question>

<answer>
This is a deliberate two-stage design:

- **Stage 1 — Pinecone hybrid search** gives me the top-K candidates fast. It uses approximate nearest neighbor (ANN) which trades some accuracy for speed. The scores here aren't precise enough to trust for cache hits.
- **Stage 2 — Cross-encoder reranking** gives me a much more accurate relevance score between the new query and each cached query. Cross-encoders look at the query-document pair jointly rather than independently, which gives far better precision.

The problem with relying purely on Pinecone's score:
- ANN scores are relative ranks, not absolute similarity. A score of 0.8 doesn't consistently mean "same question."
- The query and cached query are different strings — even if semantically identical, their embedding cosine might not hit 0.9 consistently.

The cross-encoder essentially asks "are these two queries asking the same thing?" — which is exactly what I need. The 0.9 threshold on the cross-encoder score is much more reliable than the same threshold on a bi-encoder score.
</answer>

---

## Section 5: Reranking

---

<question>
What is a cross-encoder and how is it different from the bi-encoder you use for retrieval?
</question>

<answer>
This is one of the most important distinctions in modern retrieval:

**Bi-encoder (used for retrieval):**
- Encodes the query and each document independently into separate vectors.
- Similarity is measured by dot product or cosine — very fast.
- Can be pre-computed — you embed all documents offline and store them in the vector DB.
- Approximate — it can't see the interaction between query and document tokens.

**Cross-encoder (used for reranking):**
- Takes the query and a document concatenated together as a single input.
- The transformer attention layers see both the query tokens and the document tokens simultaneously.
- Outputs a single relevance score — much more accurate than bi-encoder.
- Cannot be pre-computed — you must run inference for every query-document pair.
- Slow if used on thousands of documents.

The two-stage pipeline exploits the best of both:
1. Bi-encoder retrieves the top-45 candidates quickly (9 sources × 5 each).
2. Cross-encoder reranks those 45 candidates accurately and selects the top-10.

You'd never use a cross-encoder on the full index — that would be O(n) inference per query.
</answer>

---

<question>
Why do you apply sigmoid to the reranker logits?
</question>

<answer>
The cross-encoder for sequence classification outputs a raw logit — a single unbounded scalar value.

The sigmoid converts it to a probability between 0 and 1:
- `score = 1 / (1 + exp(-logit))`

Why I do this:
- It gives me an interpretable threshold — a score of 0.9 means 90% confident relevant, consistently.
- Without sigmoid, the raw logit could be 2.3 or -1.5 and you'd need to calibrate a different threshold for each model.
- It makes the `RERANK_THRESHOLD` and `CACHE_SCORE_THRESHOLD` config values meaningful across different model versions.

A subtle caveat: cross-encoder models are not strictly calibrated as probabilities. A sigmoid-converted score of 0.9 doesn't literally mean 90% probability of relevance — it's more of a relative confidence measure. But it's stable and interpretable for threshold-based filtering, which is what I need.
</answer>

---

<question>
What is the purpose of adjacent chunk merging and why not just retrieve larger chunks upfront?
</question>

<answer>
Chunk merging solves a real tension in RAG — the conflict between retrieval quality and context quality.

**Why not just use large chunks:**
- Larger chunks = more text per vector = the embedding averages over more content = worse retrieval precision. A 3000-character chunk embedding is less specific than a 1200-character one.
- Smaller chunks have sharper, more focused embeddings and score more accurately against queries.

**The problem with small chunks:**
- An answer might span two consecutive chunks. You retrieve chunk 4 but chunk 5 has the conclusion. The LLM gets an incomplete answer.

**What chunk merging does:**
- After reranking, I look at the IDs of retrieved chunks. My IDs follow the pattern `source_filename_chunk_0004`.
- If chunks 4 and 5 from the same document are both retrieved, I merge their text into one coherent passage.
- If they're not adjacent, they stay separate — no false merging.

The net result: small chunks for accurate retrieval, large coherent passages for LLM generation. You get the best of both worlds.

The `get_neighbour_chunk_ids` function also fetches ±1 neighbors from Pinecone to expand context even when only one chunk of a pair was retrieved.
</answer>

---

## Section 6: Chunking Strategy

---

<question>
How did you decide on chunk size 1200 characters with 200 overlap?
</question>

<answer>
The chunk size is a balance between retrieval precision and completeness of information:

- **1200 characters** is roughly 180–250 words, which is enough to contain a complete thought, paragraph, or short policy statement without being so broad the embedding loses specificity.
- Too small (e.g., 200 chars) — individual sentences don't give the LLM enough context. Retrieval might be precise but answers are thin.
- Too large (e.g., 3000 chars) — embeddings become averaged over too much content. You might retrieve a chunk that mentions your keyword on line 1 but the actual answer is buried at line 30.

**200 character overlap:**
- Prevents answers from falling through the cracks between chunks. If a sentence straddles two chunks, both chunks contain it partially.
- Combined with chunk merging, this means adjacent retrieved chunks actually overlap slightly, avoiding hard cuts in the merged passage.

In practice these numbers should be tuned per domain. Dense technical documentation might warrant smaller chunks (700–900 chars). Narrative content like emails or meeting transcripts might do better with larger chunks (1500–2000 chars).
</answer>

---

<question>
How do you handle different document types — Slack messages are very different from Confluence pages?
</question>

<answer>
Good question. The current implementation uses a uniform chunking strategy via `RecursiveCharacterTextSplitter` regardless of source type. The `extract_necessary_data` function extracts the embedding-relevant fields (`title_field_name` + `content_field_names`) from the JSON schema, then `convert_to_text` formats them as `key: value` pairs.

This is source-agnostic which is pragmatic but imperfect:
- **Slack messages** are short and conversational — chunking may produce very small chunks or single-message chunks, which is fine.
- **Confluence pages** are long and structured — chunking works well but loses document hierarchy.
- **Email threads** have metadata like sender/subject that's useful for context but not always in the content fields.

To improve this:
- Source-specific chunking strategies (e.g., no chunking for short Slack messages, heading-aware splitting for Confluence).
- Store the source type in metadata and use it in the generation prompt to help the LLM interpret the context correctly.
- Currently the `source` metadata field does tell the LLM where the chunk came from, which partially compensates.
</answer>

---

## Section 7: Pinecone & Vector Database

---

<question>
Why Pinecone over alternatives like Weaviate, Qdrant, or pgvector?
</question>

<answer>
I chose Pinecone for a few specific reasons relevant to this project:

- **Native hybrid search** — Pinecone supports sparse-dense hybrid search natively with a single API call. With Qdrant or Weaviate I'd need to run two separate searches and merge results myself.
- **Managed serverless** — zero infrastructure to manage. No cluster sizing, no replication config. For an enterprise system where the focus is the RAG logic, not the DB ops, this matters.
- **Metadata filtering** — the `user_access: {$in: [username]}` filter runs at the vector DB level, not post-retrieval. This is both more efficient and more secure than filtering in Python after fetching results.
- **Namespace isolation** — I use separate namespaces for document vectors and cache vectors in the same index. Clean separation without needing two databases.

Trade-offs vs alternatives:
- **Qdrant** — open source, self-hostable, cheaper at scale. Better if you want full control.
- **pgvector** — great if you're already on Postgres. But no native sparse support.
- **Weaviate** — has hybrid search but more complex setup. Good for multi-modal.
- **Pinecone** costs more at scale but the managed experience justified it for this project.
</answer>

---

<question>
How does the access control filtering work in Pinecone?
</question>

<answer>
Every vector stored in Pinecone has a `user_access` metadata field — a list of usernames who are authorized to see that document.

During ingestion, `extract_necessary_data` scans the JSON document for participant fields:
- `author`, `reviewers`, `owner`, `collaborators`, `assignee`, `reporter`, `creator`, `participants`, and others.
- All non-empty values from these fields are collected into the `user_access` list for that document's vectors.

At query time, every Pinecone query includes the filter:
```python
filter={"user_access": {"$in": [current_user_name]}}
```

This means Pinecone only returns vectors where the user's name appears in the `user_access` array. The filtering happens inside Pinecone before results are returned — it's not post-processing in Python.

Limitations I'm aware of:
- If a document isn't properly tagged with participant fields, it might have an empty `user_access` and become invisible to everyone.
- This is a static snapshot of access at ingestion time — if permissions change, you need to re-ingest.
- Group/role-based access (e.g., "everyone in the finance team") isn't handled — it would need to be resolved to individual names during ingestion.
</answer>

---

## Section 8: LLM Generation

---

<question>
Why did you use Groq instead of OpenAI or another provider?
</question>

<answer>
Groq is specifically optimized for LLM inference speed using custom LPU (Language Processing Unit) hardware. My reasons:

- **Latency** — Groq is significantly faster than OpenAI for the same model. For a RAG system where retrieval already adds latency, shaving 1–2 seconds off generation makes a real difference.
- **OpenAI SDK compatibility** — Groq's API is OpenAI-compatible. I use the OpenAI Python SDK with Groq's base URL, meaning I can swap providers by changing one environment variable.
- **Cost** — Groq is cheaper per token than OpenAI for comparable models like Llama.
- **`reasoning_effort` parameter** — I use `"reasoning_effort": "medium"` for a balance between response quality and speed.

The design is provider-agnostic at the interface level. If I need to switch to OpenAI GPT-4 or Anthropic Claude, I just change `GROQ_BASE_URL` and `GROQ_GENERATION_MODEL` in the `.env` file.
</answer>

---

<question>
Why do you manage the system prompt in Langfuse instead of hardcoding it?
</question>

<answer>
Langfuse is a prompt management and observability platform. The system prompt is fetched at startup via:
```python
langfuse_client.get_prompt("generation_system_prompt", label="production")
```

The benefits of this over hardcoding:

- **Zero-downtime prompt updates** — I can change the system prompt in the Langfuse UI and restart the service without changing code. In production, prompt engineering is iterative.
- **Version control** — Langfuse maintains a history of prompt versions. If a prompt change degrades quality, I can roll back.
- **Environment separation** — the `label="production"` parameter means I can have a `"development"` version of the prompt for testing without affecting production.
- **Collaboration** — non-engineers can iterate on the prompt in Langfuse without touching the codebase.

Current limitation: the prompt is fetched once at module import time. If the prompt is updated in Langfuse, the running service won't pick it up without a restart. To fix this properly, the prompt should be re-fetched per request with a short in-memory TTL cache (e.g., 5 minutes).
</answer>

---

<question>
How do you construct the context passed to the LLM?
</question>

<answer>
The `build_context_block` function in `generator.py` formats the merged chunks into a structured string:

```
Context:
Document: source_filename (chunk 3-5)
<merged text from chunks 3, 4, 5>
====================
Document: another_source (chunk 0-1)
<merged text>
====================
Query: <user's question>
```

Design choices here:
- **Source attribution** — each passage is labeled with its document ID and chunk range. The LLM can reference sources in its answer and I can display them in the UI.
- **Separator lines** — `=` separators clearly delineate documents so the LLM doesn't blend them.
- **Query appended last** — placing the query at the end after context follows the "query at end" pattern, which empirically works better for answer grounding than putting it first.
- **No truncation logic** — this is a known gap. If many long documents are retrieved, the context could exceed the model's context window. A production system should count tokens and truncate or summarize.

The merged chunks come sorted by reranker score (highest first), so the most relevant context appears at the top.
</answer>

---

## Section 9: Evaluation

---

<question>
Walk me through each evaluation metric and what the score means for your system.
</question>

<answer>
I evaluated the system using DeepEval with Groq as the judge model, logged to Langfuse. Here's what each score means:

- **Faithfulness: 0.937** — Measures whether the generated answer is factually grounded in the retrieved context. 0.937 means ~94% of claims in the answers are traceable to the retrieved documents. This is excellent — the LLM is rarely hallucinating beyond what it was given.

- **Recall@10: 0.886** — Out of all relevant documents for a query, 88.6% appear somewhere in the top-10 retrieved results. Shows the retrieval pipeline has strong coverage.

- **Recall@5: 0.886** — Same score as @10, which means the relevant documents are being found in the first 5 results. The reranker is effectively pushing them to the top.

- **MRR (Mean Reciprocal Rank): 0.880** — On average, the first truly relevant document appears at approximately rank 1.1 (1/0.88 ≈ 1.14). The system almost always puts the right document first.

- **Answer Relevancy: 0.845** — The generated answer directly addresses the user's question 84.5% of the time. Slight room for improvement — could be improved with better prompt engineering.

- **Context Recall: 0.772** — 77.2% of the ground-truth answer information is covered by the retrieved context. This is the weakest metric and points to cases where relevant documents weren't retrieved.

- **Answer Correctness: 0.742** — The factual accuracy of answers compared to ground truth. Lower than faithfulness because the context itself might be incomplete (tied to the Context Recall gap).
</answer>

---

<question>
Why is Answer Correctness (0.742) lower than Faithfulness (0.937)?
</question>

<answer>
This gap is actually expected and informative — it tells you where the system's weak point is.

Faithfulness measures: "Does the answer match what's in the retrieved context?"
Answer Correctness measures: "Does the answer match the ground truth?"

The gap between 0.937 and 0.742 tells you:
- The LLM is faithfully generating from its context — it's not hallucinating.
- But the context itself is sometimes incomplete or missing information.

The root cause is context recall (0.772) — 22.8% of the time, relevant information wasn't retrieved. When the retrieval misses something, the LLM can't include it in the answer, which drops correctness even though the LLM behaved correctly given what it had.

This means the improvement path is in **retrieval**, not generation:
- Better chunking strategies for specific document types.
- Tuning the hybrid alpha for the query distribution.
- Increasing `TOP_K_PER_SOURCE` to cast a wider net before reranking.
- The commented-out multi-source loop in `vector_store.py` — if enabled, it would query per source and likely improve recall significantly.
</answer>

---

<question>
How did you create the golden dataset for evaluation?
</question>

<answer>
The evaluation is done in `src/evaluate/enterprise-rag-evaluation.ipynb`. The golden dataset consists of question-answer-context triplets where:

- **Questions** — sampled from `questions.json` at the project root, which contains representative queries across different enterprise data sources.
- **Ground truth answers** — either manually written or generated by a stronger model (GPT-4/Claude) given the full document context.
- **Expected context** — the specific document IDs or passages that should be retrieved to answer each question correctly.

Limitations of the current evaluation:
- The golden dataset is static. As new documents are ingested, the coverage of the evaluation may not represent the full index.
- LLM-as-judge metrics (faithfulness, answer correctness) have inherent subjectivity — different judge models can give different scores.
- The evaluation doesn't test access control — all questions are run as a single user.

For a production evaluation, I'd also include adversarial questions (questions designed to retrieve the wrong document), multi-hop questions, and questions with no answer in the corpus.
</answer>

---

## Section 10: Architecture Decisions

---

<question>
Why did you separate the FastAPI backend and Streamlit UI into different containers?
</question>

<answer>
The separation serves both technical and operational purposes:

- **Dependency isolation** — the API needs torch, ONNX, transformers (~2 GB dependencies). The UI just needs streamlit and requests (~200 MB). Without separation, every UI deployment would unnecessarily carry 1.8 GB of ML libraries.
- **Independent scaling** — in a production setting, you might run 3 UI replicas for user traffic but only 1 API instance (because ONNX models consume ~3 GB RAM each). Containers let you scale them independently.
- **Separation of concerns** — the API is the intelligence; the UI is just a display layer. They can be updated independently. You could replace Streamlit with a React frontend without touching the API.
- **Health-check safety** — docker-compose waits for the API to be healthy before starting the UI. If the API fails to load models, the UI shows a clear "server unavailable" message instead of confusing errors.
- **Security** — in production, you'd put the API on an internal network and only expose the UI publicly. Separation makes this network policy trivial.
</answer>

---

<question>
Why is the ingestion pipeline not a separate Docker image?
</question>

<answer>
Ingestion reuses the same API image rather than having its own Dockerfile. The reasoning:

- **Shared dependencies** — both ingestion and the API need the same ONNX models (dense embedder, SPLADE) loaded and running. Building a separate identical image wastes both build time and disk space.
- **Shared model_store** — by reusing the API image with a bind mount to `./model_store`, the ONNX models downloaded during ingestion are immediately available to the API. No re-download needed.
- **One-shot behavior** — ingestion runs once, exits when done. Docker Compose profiles (`profiles: [ingestion]`) mean it's not started by default with `docker compose up` — only on explicit request.
- **Maintenance simplicity** — one Dockerfile to maintain instead of two nearly-identical ones.

If ingestion ever needed different base dependencies (e.g., GPU-heavy augmentation), it would warrant its own Dockerfile. For now, it's just a different command (`python -m src.ingestion.main`) on the same image.
</answer>

---

<question>
How does the pipeline initialization work and why does it happen at startup rather than per request?
</question>

<answer>
`initialize_search_pipeline()` in `generation/main.py` loads all four model components at API startup:
1. Dense embedding tokenizer + ONNX model
2. SPLADE sparse tokenizer + ONNX session  
3. Reranker tokenizer + ONNX model
4. Pinecone connection + index handle

These are stored in the `pipeline` global in `api.py` and unpacked per request.

Why at startup not per request:
- **Model loading takes 30–60 seconds** — doing this per request is completely impractical. Users would wait a minute before getting their first answer.
- **Models are stateless for inference** — once loaded, they can handle any number of requests. There's no per-user or per-request state in the model weights.
- **Memory efficiency** — loading once means three models sharing system RAM. Per-request loading would peak at 3× the memory per concurrent request.

The trade-off:
- Cold startup is slow (1–2 minutes). The Docker health check has a 120-second `start_period` for exactly this reason.
- The API is "unhealthy" during startup, so the UI waits via `depends_on: condition: service_healthy`.
- If the models fail to load, the service stays unhealthy and needs a restart — there's no partial recovery logic.
</answer>

---

## Section 11: Codebase-Specific Questions

---

<question>
Your `query_all_sources` function has commented-out loops. What was the original design intent?
</question>

<answer>
The original design was true multi-source retrieval — querying Pinecone separately for each of the 9 data sources (confluence, github, slack, gmail, etc.) in parallel using `async_req=True`.

The intent was:
- Each source would get its own `top_k=5` query, guaranteeing at least 5 candidates per source.
- All 9 queries fire asynchronously using Pinecone's future-based async interface.
- Results from all 9 are collected, merged, and passed to the reranker.

Why it's commented out:
- The current implementation does a single query across the full namespace with a `user_access` filter. This is simpler but means some sources could dominate the results if they have more data.
- The per-source loop was likely commented out during development/debugging and never re-enabled.

The current behavior versus the stated architecture:
- README and diagrams say "9 sources · top-5 each" — this is technically incorrect in the current code.
- The single-namespace query still returns diverse results because the `user_access` filter ensures user-relevant content, but there's no source-level diversity guarantee.

To restore the original intent, the loop needs to be uncommented, the per-source filter added, and the future collection loop restored.
</answer>

---

<question>
Why do you have two nearly-identical embedder files in ingestion and retrieval?
</question>

<answer>
This is a code duplication issue I'm aware of. Both `src/ingestion/embedder.py` and `src/retrieval/embedder.py` implement the same ONNX-based dense and sparse embedding logic — they differ only in which config module they import from.

How it came to be:
- Ingestion and retrieval originally used different embedding mechanisms. Ingestion used SentenceTransformer, retrieval used ONNX.
- When I migrated ingestion to ONNX to match retrieval, the cleanest path was a full rewrite mirroring the retrieval module.
- The proper refactor (extracting to `src/shared/embedder.py`) was deferred.

The risk:
- Any bug fix or optimization applied to `retrieval/embedder.py` won't automatically apply to `ingestion/embedder.py`.
- Embeddings must be computed with identical models and identical inference code. If the two diverge, you'll have vectors in Pinecone (from ingestion) and queries (from retrieval) that aren't comparable.

The fix is a shared `src/utils/embedder.py` that both modules import, parameterized by config values. Both ingestion and retrieval configs now have matching variable names (`DENSE_EMBEDDING_MODEL`, `SPARSE_EMBEDDING_MODEL`, etc.) specifically to make this refactor straightforward.
</answer>

---

<question>
What does `get_neighbour_chunk_ids` do and when is it used?
</question>

<answer>
`get_neighbour_chunk_ids` takes a list of retrieved chunk IDs and expands each one to include its immediate neighbors (chunk N-1 and chunk N+1 from the same document).

How it works:
- Chunk IDs follow the pattern `source_filename_chunk_0004` (zero-padded number).
- Using regex, it extracts the prefix and chunk number.
- For each retrieved chunk N, it adds N-1 and N+1 to the neighbor set.

Why this exists:
- The reranker might retrieve chunk 5 of a document but the critical sentence is at the start of chunk 6.
- Instead of merging (which combines already-retrieved chunks), neighbor fetching actively pulls adjacent chunks from Pinecone using `fetch_vectors_by_id`.
- This gives the LLM richer surrounding context without widening the initial retrieval net.

The relationship to chunk merging:
- `get_neighbour_chunk_ids` expands retrieval — fetch more chunks.
- `merge_ranked_chunks` consolidates retrieval — combine adjacent retrieved chunks into coherent passages.
- Both work together: first expand context, then merge overlapping expansions.

Note: in the current `query.py`, only `rerank_local` is called — the neighbor fetching step from `chunk_utils.py` isn't actually wired into the live pipeline. It exists but isn't invoked.
</answer>

---

<question>
Your `generate_response` function has `_, _, _ = generate_response()` at the bottom. What is that?
</question>

<answer>
That's dead code — a leftover from development or testing. It's a call to `generate_response()` with no arguments at module import time, which would immediately throw a `TypeError` because the function requires `query` and `merged_docs` parameters.

The fact that it starts with `_, _, _` suggests it was written to test that the return value unpacks into three items correctly. It should be deleted.

In the current state it's harmless only because:
- Python's module import system caches modules. If `generator.py` is imported by `generation/main.py`, Python runs the file once and caches it.
- The line `_, _, _ = generate_response()` only executes if the module is run directly (`python generator.py`), not when imported.
- But technically, it would crash if someone ran `python -m src.generation.generator` directly.

It should be removed before any interview demo. It's a code hygiene issue that suggests the file wasn't cleaned up properly before submission.
</answer>

---

## Section 12: Scalability & Production

---

<question>
How would this system perform under high concurrent load?
</question>

<answer>
The current design has several scalability bottlenecks I can walk through honestly:

**Bottleneck 1 — ONNX inference is synchronous and single-threaded:**
- FastAPI is async but the ONNX inference calls in embedder and reranker are CPU-bound synchronous operations.
- Under concurrent load, requests queue up on the embedding step. The async def in `query_endpoint` doesn't help for CPU-bound work.
- Fix: run ONNX inference in a `ThreadPoolExecutor` via `asyncio.run_in_executor` to unblock the event loop.

**Bottleneck 2 — Single API instance, no horizontal scaling:**
- The pipeline is loaded once into memory (~3 GB). You can't easily run 5 replicas on one machine.
- Fix: containerize properly with a load balancer, but each replica needs its own 3 GB of model memory — expensive.

**Bottleneck 3 — Pinecone has rate limits:**
- Under high load, parallel Pinecone queries could hit API rate limits.
- Fix: connection pooling, request queuing, or Pinecone's dedicated pod tier.

**What scales well:**
- The semantic cache dramatically reduces LLM load — frequently asked questions never reach Groq.
- Pinecone itself is horizontally scalable — it's a managed service.
- The Groq API is also externally scaled.
</answer>

---

<question>
What would you change about this architecture if you were taking it to production with 10,000 daily users?
</question>

<answer>
At 10,000 daily users (roughly 7 queries per minute average), several things need to change:

- **Separate the embedding service** — move ONNX inference to a dedicated microservice (e.g., Triton Inference Server). The API becomes a thin orchestrator. This lets you scale inference independently and use GPU hardware efficiently.
- **Async embedding** — the current synchronous ONNX calls need to be moved to a worker pool so the FastAPI event loop isn't blocked.
- **Cache layer upgrade** — the semantic cache in Pinecone is great for semantic matching, but add Redis in front for exact-match query caching (same query, no embedding needed).
- **Re-enable multi-source retrieval** — the commented-out per-source loop needs to be restored to ensure diverse, balanced retrieval.
- **Access control hardening** — move from embedding user names directly to an identity provider (OAuth/OIDC). Resolve group memberships to individual names at query time from a proper directory service.
- **Streaming responses** — set `stream=True` on the Groq call and stream tokens back to the UI. Users perceive streaming as faster even if total latency is the same.
- **Monitoring** — add Prometheus metrics (latency percentiles, cache hit rate, error rate), not just Langfuse traces.
- **Prompt refresh** — fetch the Langfuse system prompt with a TTL cache so prompt updates don't require restarts.
</answer>

---

<question>
How would you handle a document that's too large to fit in one Pinecone upsert batch?
</question>

<answer>
The current code handles large documents by chunking them into 1200-character pieces and batching upserts in groups of 25 vectors. A very large document (say a 500-page PDF) would produce many chunks but they'd all get upserted in batches of 25.

There are two edge cases the current code doesn't handle well:

1. **Very high-dimensional batches** — Pinecone has a request size limit (4 MB per upsert call). If chunk metadata is large (long `user_access` lists, long text fields), even 25 vectors might exceed the limit. The fix is to dynamically size batches based on estimated payload size, not a fixed count of 25.

2. **Final batch data loss** — this is an actual bug in the current code. If the total number of chunks isn't a multiple of 25, the last partial batch is returned but never upserted. A final flush after the loop is missing. Adding `if results: index.upsert(vectors=results)` after the main loop would fix this.

For very large corpora (millions of documents), you'd also want:
- Parallelism at the file level (multiple processes, not threads, to avoid Python GIL on ONNX inference).
- A job queue (Celery, RQ) with retry logic and dead-letter queues for failed files.
- Progress persistence so a restart doesn't reprocess already-ingested documents.
</answer>

---

## Section 13: Design Trade-offs

---

<question>
Why use ONNX locally for reranking instead of Pinecone's managed reranking API?
</question>

<answer>
Pinecone offers a managed reranking API (`pinecone-rerank-v0`) — the `rerank_pinecone` function in my codebase shows I evaluated it. I chose local ONNX reranking instead.

**Reasons for local ONNX:**
- **Latency** — a local ONNX cross-encoder running on CPU completes reranking in 50–150ms. A round-trip to Pinecone's reranking API adds network latency on top.
- **Cost** — Pinecone charges per reranking call. At scale, local inference is zero marginal cost.
- **Control** — I can choose my own reranker model and tune the threshold. With the managed API you're locked into `pinecone-rerank-v0`.
- **Offline capability** — local inference works without any external API dependency.

**When Pinecone reranking makes sense:**
- You don't want to manage model files.
- You're on a CPU-constrained machine where even a small ONNX model is too heavy.
- You want zero-latency model updates (Pinecone can update their model without your code changing).

The `LOCAL_RERANK` config flag in `retrieval/config.py` exists specifically to make this choice switchable without code changes.
</answer>

---

<question>
Why store dense embeddings in Pinecone if you're also using hybrid search? Couldn't you use pure sparse?
</question>

<answer>
Pure sparse retrieval would miss semantic relationships that dense embeddings capture. This is the fundamental reason hybrid exists.

Consider these query examples:
- Query: "show me all quarterly business reviews"
- Document: "Q3 strategy presentation and KPI summary"

A pure sparse search looks for the exact tokens "quarterly," "business," "review" — none of which appear in the document. Score = 0.

Dense embeddings map both to similar vector regions because BGE/E5 models learn that QBR, quarterly reviews, and strategy presentations are semantically related.

Conversely:
- Query: "invoice number INV-2024-0089"
- Document: "Payment for INV-2024-0089 approved"

Dense embeddings might not give this a high score because invoice numbers are arbitrary strings. Sparse (SPLADE) would give it a perfect score because the exact token matches.

Enterprise data spans both types of queries. Pure dense fails on exact IDs, codes, names. Pure sparse fails on semantic paraphrases and concept-level queries. Hybrid covers both, which is why the Recall@5 and Recall@10 metrics are both 0.886 — strong across diverse query types.
</answer>

---

<question>
What is the risk of using SHA256 of the query as the cache key?
</question>

<answer>
SHA256 of the query text is used as the Pinecone vector ID for cache entries. This has a subtle collision property:

- Two different queries that map to the same hash (SHA256 collision) would overwrite each other. In practice, SHA256 collisions are computationally infeasible — this isn't a real risk.

The actual design concerns are:
- **Case sensitivity** — "What is the leave policy" and "what is the leave policy" produce different hashes even though they're the same query. The semantic cache search would still find the match via embedding similarity, but if someone is also querying by ID they'd get separate cache entries.
- **User-independent key** — two different users asking the same question produce the same hash, so they'd overwrite each other's cache entry. The `user_access` filter at retrieval time ensures users only get cache entries they're authorized to see, but the stored vector is shared. If User A and User B have different access levels and ask the same question, User A's cached answer (with User A's access-filtered documents) might be returned to User B if B has broader access. This is a subtle access control edge case.

The right fix is to include `user_name` in the hash: `hashlib.sha256(f"{user_name}:{query}".encode())`. But this eliminates cross-user cache sharing for identical queries, reducing cache hit rate.
</answer>

---

## Section 14: Observability & Operations

---

<question>
How do you monitor the system in production and how would you know something is wrong?
</question>

<answer>
Currently the observability is built around Langfuse with some gaps:

**What's in place:**
- **Langfuse traces** — evaluation scores (faithfulness, answer correctness, recall, MRR) are logged per trace.
- **Structured logging** — the `get_logger()` utility provides consistent log formatting across all modules. Log level is configurable via `LOG_LEVEL` env var.
- **API health endpoint** — `GET /health` returns pipeline load status. Docker monitors this for container health.
- **Streamlit sidebar** — shows live eval scores from Langfuse refreshed on demand.

**Gaps in the current setup:**
- No request-level latency metrics (no Prometheus, no StatsD).
- No alerting — there's no way to know cache hit rate dropped or error rate spiked without actively looking.
- Cache errors are silently swallowed — `check_cache` returns `False, [], None` on any exception, so cache failures are invisible unless you read logs.
- `print()` statements mixed with `logger` calls mean some operational events bypass structured logging entirely.

**What I'd add for production:**
- Prometheus metrics endpoint on the FastAPI app (request count, latency percentiles, cache hit rate, embedding latency).
- Alerting on error rate > 1% and P95 latency > 10 seconds.
- Distributed tracing (OpenTelemetry) across the embedder, retriever, reranker, and generator steps.
</answer>

---

<question>
If the API starts returning wrong answers, how would you debug it?
</question>

<answer>
I'd work through the pipeline layer by layer:

**Step 1 — Is it a retrieval issue or a generation issue?**
- Call the `/query` endpoint and check `is_cached` in the response. If `True`, the answer came from cache — check if the cached answer is wrong or the cache entry is stale.
- Add a debug endpoint that returns the retrieved chunks alongside the answer. If the retrieved chunks are irrelevant, it's a retrieval problem. If they're relevant but the answer is wrong, it's a generation problem.

**Step 2 — Diagnose retrieval problems:**
- Check `user_access` metadata — is the user filtered out of the right documents?
- Check hybrid alpha — are queries semantically vague (needs more dense) or keyword-specific (needs more sparse)?
- Run `evaluate_retrieval` against the golden dataset to see if Recall@5/MRR have dropped.
- Check if new document ingestion used a different model version than retrieval (embedding drift).

**Step 3 — Diagnose generation problems:**
- Check the Langfuse trace for the system prompt version — was it updated?
- Check `finish_reason` — if it's `length`, the model hit the token limit and truncated the answer.
- Check faithfulness score — if it drops, the LLM is ignoring context or hallucinating.

**Step 4 — Check infrastructure:**
- Pinecone index consistency — were vectors corrupted or deleted?
- Model store — was an ONNX model overwritten with a different version?
</answer>

---

## Section 15: General ML System Design

---

<question>
How would you handle a user query for which there is genuinely no answer in the knowledge base?
</question>

<answer>
This is the "no answer" or "unanswerable" case, and it's handled partially in the current system:

**Current behavior:**
- If retrieval returns nothing (empty `retrieved_docs.data` after access control filtering), `run_query` returns the message "You dont have the permission to access the file." This is misleading — it conflates "no access" with "no documents found."
- If retrieval returns documents but none are relevant, the LLM will generate an answer from irrelevant context — potentially a wrong or hallucinated answer.

**What the system should do:**
- **Low reranker scores** — if all reranked results score below a confidence threshold, return "I don't have enough information to answer this question" rather than generating from weak context.
- **Explicit abstention** — prompt the LLM with an instruction like "If the provided context doesn't contain enough information to answer the question, say 'I don't have this information' rather than guessing."
- **Distinguish no-access from no-content** — separate the permission check from the empty results check and give the user different messages.

**Confidence scoring:**
- The reranker score of the top result is a useful proxy for answer confidence. If `max(reranker_scores) < 0.3`, the system probably doesn't have the answer. You could return this confidence alongside the answer for the UI to display appropriately.
</answer>

---

<question>
How does embedding model choice affect retrieval quality, and how would you pick a different model?
</question>

<answer>
The embedding model is the most impactful single choice in a RAG system — it defines the semantic space everything else operates in.

**What makes a good retrieval embedding model:**
- Trained on retrieval tasks (not just sentence similarity) — models like BGE-large, E5-large, or GTE are specifically fine-tuned for retrieval.
- Dimensionality vs. quality trade-off — larger dimensions (1024 vs 384) generally mean better representation but more Pinecone storage and slower search.
- Domain fit — a model trained on general web text may underperform on highly technical documents. Domain-specific fine-tuning helps.

**How to evaluate a candidate model:**
1. Re-embed a sample of your document corpus with the new model.
2. Re-embed the golden question set.
3. Run retrieval without reranking and compute Recall@10 to see raw retrieval quality.
4. Run the full pipeline and compute the full evaluation suite.
5. Compare against the baseline.

**Practical considerations:**
- Both the ingestion embedder and retrieval embedder must use the same model. Changing the model means re-ingesting all documents.
- The model must be exportable to ONNX via `optimum`. Some models don't export cleanly.
- The Pinecone index dimension must match the model's output dimension. Changing models requires re-creating the index.

In my config, `DENSE_EMBEDDING_MODEL` is an environment variable specifically so the model can be swapped without code changes.
</answer>

---

<question>
What would happen if you used a different embedding model for ingestion than for retrieval?
</question>

<answer>
This is a critical consistency requirement — using different models for ingestion and retrieval would completely break the system.

Here's why:
- Dense retrieval works by finding vectors that are close in a learned embedding space.
- Each model defines its own unique vector space. Model A's embedding for "leave policy" might be the vector `[0.2, -0.5, 0.8, ...]`. Model B's embedding for the same text is `[0.7, 0.1, -0.3, ...]`.
- A query embedded with Model B searching through vectors created with Model A would find the nearest neighbors in the wrong space — completely meaningless results.

The symptoms would be:
- Near-zero retrieval scores for all queries.
- Totally irrelevant documents returned.
- The system would appear to "work" (no crashes) but every answer would be wrong.

This is why in my code both `src/ingestion/embedder.py` and `src/retrieval/embedder.py` share the same config variable names (`DENSE_EMBEDDING_MODEL`, `DENSE_EMBEDDING_ONNX_FILE`). If you set them to different values in `.env`, the ingestion and retrieval models diverge — that would be a bug.

Best practice: the embedding model version should be stored as metadata in the Pinecone index. On startup, the retrieval service should verify it's using the same model version that was used for ingestion.
</answer>

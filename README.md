# Medium Article RAG Assistant

A retrieval-augmented assistant over the public Medium Articles dataset (~7,600 English articles). Answers four question types — precise fact retrieval, multi-result topic listing, summary extraction, and evidence-based recommendation — using only the indexed corpus.

## Stack

- **Embeddings**: `4UHRUIN-text-embedding-3-small` (1536 dim) via llmod.ai proxy, accessed through `langchain_openai.OpenAIEmbeddings`
- **Chat**: `4UHRUIN-gpt-5-mini` via the same proxy, accessed through `langchain_openai.ChatOpenAI`
- **Vector store**: Pinecone serverless (cosine, 1536 dim)
- **Runtime**: Python serverless functions on Vercel (`api/prompt.py`, `api/stats.py`)
- **Ingestion**: local one-time script (`scripts/ingest.py`) — never re-embeds; state file tracks completed articles

## Hyperparameters

Reported via `GET /api/stats`:

| Param | Value | Rationale |
|---|---|---|
| `chunk_size` | 512 tokens | Captures coherent passages for `text-embedding-3-small` without diluting signal |
| `overlap_ratio` | 0.10 | Avoids losing ideas at chunk boundaries while keeping embedding cost ~10% lower than the assignment max of 0.30 |
| `top_k` | 8 | Enough chunks to cover up-to-3 distinct articles for list-style questions; we dedupe to ≤2 chunks per article before sending to the LLM |

These can be changed via `.env.local` — `/api/stats` always reflects the current env values.

## API

### `POST /api/prompt`

Request:
```json
{ "question": "List exactly 3 articles about education. Return only the titles." }
```

Response:
```json
{
  "response": "...",
  "context": [
    { "article_id": "...", "title": "...", "chunk": "...", "score": 0.5234 }
  ],
  "Augmented_prompt": { "System": "...", "User": "..." }
}
```

### `GET /api/stats`

```json
{ "chunk_size": 512, "overlap_ratio": 0.10, "top_k": 8 }
```

## Local development

```bash
# 1. Create venv and install
uv venv
uv pip install -r requirements.txt

# 2. Configure
cp .env.example .env.local
# then fill in OPENAI_API_KEY and PINECONE_API_KEY

# 3. Ingest (one-time)
.venv/bin/python scripts/ingest.py --dry-run --limit 5   # estimate cost
.venv/bin/python scripts/ingest.py --limit 50            # validate end-to-end
.venv/bin/python scripts/ingest.py                       # full corpus

# 4. Smoke-test retrieval + LLM
.venv/bin/python scripts/test_query.py

# 5. Deploy
vercel
```

## Ingestion design (budget-aware)

- **Idempotent**: `ingestion_state.json` tracks completed `article_id`s; re-runs skip them
- **Batched**: 100 chunks per embedding request, 100 vectors per Pinecone upsert
- **Subset-first**: start with `--limit 50` to validate before paying for the full corpus
- **Cost estimate printed** at the end of every run

Estimated full-corpus cost: ~$0.20 in embeddings + a few cents in chat usage per ~50 queries. Total well under the $5 budget.

## Retrieval design

- Query is embedded with the same model + dimensions as the corpus
- Pinecone cosine-similarity top-k (k=8) → deduped to at most 2 chunks per article (max 4 distinct articles in context)
- Context block is rendered with metadata (`title`, `author`, `url`) plus the chunk text, so the LLM can cite by title and stay grounded
- The required system prompt is enforced verbatim; the user prompt instructs the model to use only the supplied context

## File layout

```
api/
  _lib/              # shared modules (underscore = not a serverless route)
    config.py
    chunking.py
    embedding.py
    llm.py
    prompts.py
    retrieval.py
  prompt.py          # POST /api/prompt
  stats.py           # GET  /api/stats
scripts/
  ingest.py          # one-time embed+upsert
  test_query.py      # local smoke test
.env.example
requirements.txt
vercel.json
```

"""Local one-time ingestion: read CSV, chunk, embed, upsert to Pinecone.

Usage:
    python scripts/ingest.py --limit 50          # small subset for validation
    python scripts/ingest.py                     # full corpus
    python scripts/ingest.py --force             # re-embed everything
    python scripts/ingest.py --dry-run --limit 5 # chunk only, no API calls

Idempotent: tracks which article_ids were ingested in `ingestion_state.json`
and skips them on re-run unless `--force` is passed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from api._lib.chunking import build_chunks, token_count
from api._lib.config import load_config
from api._lib.embedding import embed_texts, make_embeddings
from api._lib.retrieval import ensure_index, get_index, get_pinecone

STATE_PATH = PROJECT_ROOT / "ingestion_state.json"
EMBED_BATCH = 100
UPSERT_BATCH = 100


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"ingested_article_ids": []}


def save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2))


def iter_rows(csv_path: str, limit: int | None):
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    if limit is not None:
        df = df.head(limit)
    for _, row in df.iterrows():
        yield row.to_dict()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="max articles to ingest (for subset testing)")
    parser.add_argument("--force", action="store_true", help="ignore state file and re-embed everything")
    parser.add_argument("--dry-run", action="store_true", help="chunk only, no embedding/upsert (estimates token cost)")
    args = parser.parse_args()

    cfg = load_config()
    print(f"[cfg] chunk_size={cfg.chunk_size} overlap_ratio={cfg.overlap_ratio} top_k={cfg.top_k}")
    print(f"[cfg] csv={cfg.csv_path} index={cfg.pinecone_index} embed_model={cfg.embedding_model}")

    state = {"ingested_article_ids": []} if args.force else load_state()
    done_ids = set(state.get("ingested_article_ids", []))
    print(f"[state] {len(done_ids)} article_ids already ingested (will skip; pass --force to override)")

    if not args.dry_run:
        if not cfg.openai_api_key:
            print("ERROR: OPENAI_API_KEY is missing", file=sys.stderr)
            return 1
        if not cfg.pinecone_api_key:
            print("ERROR: PINECONE_API_KEY is missing", file=sys.stderr)
            return 1
        embeddings = make_embeddings(cfg)
        pc = get_pinecone(cfg)
        ensure_index(pc, cfg)
        index = get_index(pc, cfg)
    else:
        embeddings = None
        index = None

    total_articles = 0
    skipped_articles = 0
    total_chunks = 0
    total_tokens = 0
    t0 = time.time()

    pending_texts: list[str] = []
    pending_records: list[dict] = []

    def flush() -> None:
        nonlocal pending_texts, pending_records, total_tokens
        if not pending_records:
            return
        if args.dry_run:
            print(f"[dry-run] would embed batch of {len(pending_texts)} chunks")
            pending_texts = []
            pending_records = []
            return
        vectors = embed_texts(embeddings, pending_texts)
        upserts = []
        for vec, rec in zip(vectors, pending_records):
            upserts.append(
                {
                    "id": rec["chunk_id"],
                    "values": vec,
                    "metadata": {
                        "article_id": rec["article_id"],
                        "chunk_index": rec["chunk_index"],
                        "title": rec["title"],
                        "author": rec["author"],
                        "url": rec["url"],
                        "tags": rec["tags"],
                        "timestamp": rec["timestamp"],
                        "text": rec["text"],
                    },
                }
            )
        for i in range(0, len(upserts), UPSERT_BATCH):
            index.upsert(vectors=upserts[i : i + UPSERT_BATCH])
        print(f"[upsert] {len(upserts)} vectors  (elapsed {time.time() - t0:.1f}s)")
        pending_texts = []
        pending_records = []

    for row in iter_rows(cfg.csv_path, args.limit):
        chunks = build_chunks(row, cfg.chunk_size, cfg.overlap_ratio)
        if not chunks:
            continue
        aid = chunks[0].article_id
        total_articles += 1
        if aid in done_ids:
            skipped_articles += 1
            continue

        for ch in chunks:
            total_chunks += 1
            total_tokens += token_count(ch.text)
            pending_texts.append(ch.text)
            pending_records.append(
                {
                    "chunk_id": ch.chunk_id,
                    "article_id": ch.article_id,
                    "chunk_index": ch.chunk_index,
                    "title": ch.title,
                    "author": ch.author,
                    "url": ch.url,
                    "tags": ch.tags,
                    "timestamp": ch.timestamp,
                    "text": ch.text,
                }
            )

            if len(pending_texts) >= EMBED_BATCH:
                flush()

        done_ids.add(aid)
        if not args.dry_run and total_articles % 50 == 0:
            state["ingested_article_ids"] = sorted(done_ids)
            save_state(state)

    flush()

    if not args.dry_run:
        state["ingested_article_ids"] = sorted(done_ids)
        save_state(state)

    print()
    print(f"[done] articles seen={total_articles} skipped={skipped_articles} new_chunks={total_chunks}")
    print(f"[done] embedded tokens (approx)={total_tokens}")
    if total_tokens:
        est_cost_usd = total_tokens / 1_000_000 * 0.02
        print(f"[done] estimated embedding cost @ $0.02/1M tokens: ${est_cost_usd:.4f}")
    print(f"[done] total elapsed {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

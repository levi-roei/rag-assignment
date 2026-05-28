"""Local end-to-end smoke test: runs the same code path as /api/prompt."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "api"))

from _lib.config import load_config
from _lib.embedding import embed_query, make_embeddings
from _lib.llm import chat_complete, make_chat
from _lib.prompts import SYSTEM_PROMPT, build_user_prompt
from _lib.retrieval import dedupe_by_article, get_index, get_pinecone, query_index


def answer(question: str) -> None:
    cfg = load_config()
    embeddings = make_embeddings(cfg)
    chat = make_chat(cfg)
    pc = get_pinecone(cfg)
    index = get_index(pc, cfg)

    qvec = embed_query(embeddings, question)
    raw_hits = query_index(index, qvec, top_k=cfg.top_k)
    hits = dedupe_by_article(raw_hits, max_per_article=2)

    print("=" * 80)
    print(f"Q: {question}")
    print("-" * 80)
    print(f"Retrieved {len(hits)} chunks (dedup from {len(raw_hits)}):")
    for h in hits:
        print(f"  score={h['score']:.3f}  title={h['title'][:70]!r}")

    user_prompt = build_user_prompt(question, [
        {
            "article_id": h["article_id"],
            "title": h["title"],
            "author": h.get("author", ""),
            "url": h.get("url", ""),
            "chunk": h["chunk"],
        }
        for h in hits
    ])
    resp = chat_complete(chat, SYSTEM_PROMPT, user_prompt)
    print("-" * 80)
    print("A:", resp)
    print()


if __name__ == "__main__":
    queries = sys.argv[1:] or [
        "Find an article about turning a popular blog series into a bestselling book. Provide the title and author.",
        "List exactly 3 articles about writing. Return only the titles.",
        "Find an article about being productive and creative during times of panic. Summarise its central argument.",
        "I want practical marketing advice for introverts. Which article would you recommend, and why?",
    ]
    for q in queries:
        answer(q)

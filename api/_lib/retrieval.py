from __future__ import annotations

from typing import Any

from pinecone import Pinecone, ServerlessSpec

from .config import Config


def get_pinecone(cfg: Config) -> Pinecone:
    return Pinecone(api_key=cfg.pinecone_api_key)


def ensure_index(pc: Pinecone, cfg: Config) -> None:
    existing = {i["name"] for i in pc.list_indexes()}
    if cfg.pinecone_index in existing:
        return
    pc.create_index(
        name=cfg.pinecone_index,
        dimension=cfg.embedding_dimensions,
        metric="cosine",
        spec=ServerlessSpec(cloud=cfg.pinecone_cloud, region=cfg.pinecone_region),
    )


def get_index(pc: Pinecone, cfg: Config):
    return pc.Index(cfg.pinecone_index)


def query_index(
    index,
    vector: list[float],
    top_k: int,
) -> list[dict[str, Any]]:
    resp = index.query(vector=vector, top_k=top_k, include_metadata=True)
    hits = []
    for m in resp.get("matches", []):
        md = m.get("metadata", {}) or {}
        hits.append(
            {
                "id": m.get("id"),
                "score": float(m.get("score", 0.0)),
                "article_id": md.get("article_id", ""),
                "title": md.get("title", ""),
                "author": md.get("author", ""),
                "url": md.get("url", ""),
                "tags": md.get("tags", ""),
                "timestamp": md.get("timestamp", ""),
                "chunk": md.get("text", ""),
            }
        )
    return hits


def dedupe_by_article(hits: list[dict[str, Any]], max_per_article: int = 2) -> list[dict[str, Any]]:
    seen: dict[str, int] = {}
    out: list[dict[str, Any]] = []
    for h in hits:
        aid = h.get("article_id", "")
        n = seen.get(aid, 0)
        if n >= max_per_article:
            continue
        seen[aid] = n + 1
        out.append(h)
    return out

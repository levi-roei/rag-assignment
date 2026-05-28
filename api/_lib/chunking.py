from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import tiktoken

_ENCODER = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    chunk_id: str
    article_id: str
    chunk_index: int
    title: str
    author: str
    url: str
    tags: str
    timestamp: str
    text: str


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def article_id_from_url(url: str, fallback: str) -> str:
    seed = (url or fallback or "").strip()
    if not seed:
        seed = fallback
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]


def chunk_text(text: str, chunk_size: int, overlap_ratio: float) -> list[str]:
    if not text:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if not 0 <= overlap_ratio < 1:
        raise ValueError("overlap_ratio must be in [0, 1)")

    tokens = _ENCODER.encode(text, disallowed_special=())
    if len(tokens) <= chunk_size:
        return [text]

    step = max(1, int(chunk_size * (1 - overlap_ratio)))
    chunks: list[str] = []
    for start in range(0, len(tokens), step):
        window = tokens[start : start + chunk_size]
        if not window:
            break
        chunks.append(_ENCODER.decode(window))
        if start + chunk_size >= len(tokens):
            break
    return chunks


def build_chunks(
    row: dict,
    chunk_size: int,
    overlap_ratio: float,
) -> list[Chunk]:
    title = (row.get("title") or "").strip()
    text = _clean_text(row.get("text") or "")
    if not text:
        return []

    url = (row.get("url") or "").strip()
    author = (row.get("authors") or "").strip()
    tags = (row.get("tags") or "").strip()
    timestamp = (row.get("timestamp") or "").strip()

    aid = article_id_from_url(url, fallback=title)

    pieces = chunk_text(text, chunk_size, overlap_ratio)
    chunks: list[Chunk] = []
    for i, piece in enumerate(pieces):
        chunks.append(
            Chunk(
                chunk_id=f"{aid}-{i}",
                article_id=aid,
                chunk_index=i,
                title=title,
                author=author,
                url=url,
                tags=tags,
                timestamp=timestamp,
                text=piece,
            )
        )
    return chunks


def token_count(text: str) -> int:
    return len(_ENCODER.encode(text or "", disallowed_special=()))

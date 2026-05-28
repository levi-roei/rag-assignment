from __future__ import annotations

from langchain_openai import OpenAIEmbeddings

from .config import Config


def make_embeddings(cfg: Config) -> OpenAIEmbeddings:
    kwargs = {
        "api_key": cfg.openai_api_key,
        "model": cfg.embedding_model,
        "dimensions": cfg.embedding_dimensions,
    }
    if cfg.openai_base_url:
        kwargs["base_url"] = cfg.openai_base_url
    return OpenAIEmbeddings(**kwargs)


def embed_texts(embeddings: OpenAIEmbeddings, texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    return embeddings.embed_documents(texts)


def embed_query(embeddings: OpenAIEmbeddings, query: str) -> list[float]:
    return embeddings.embed_query(query)

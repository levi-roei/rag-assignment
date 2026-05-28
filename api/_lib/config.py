import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    load_dotenv(".env.local")
    load_dotenv()
except ImportError:
    pass


@dataclass(frozen=True)
class Config:
    openai_api_key: str
    openai_base_url: str | None
    embedding_model: str
    chat_model: str
    embedding_dimensions: int

    pinecone_api_key: str
    pinecone_index: str
    pinecone_cloud: str
    pinecone_region: str

    csv_path: str

    chunk_size: int
    overlap_ratio: float
    top_k: int


def load_config() -> Config:
    return Config(
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
        openai_base_url=os.environ.get("OPENAI_BASE_URL") or None,
        embedding_model=os.environ.get("EMBEDDING_MODEL", "4UHRUIN-text-embedding-3-small"),
        chat_model=os.environ.get("CHAT_MODEL", "4UHRUIN-gpt-5-mini"),
        embedding_dimensions=int(os.environ.get("EMBEDDING_DIMENSIONS", "1536")),
        pinecone_api_key=os.environ.get("PINECONE_API_KEY", ""),
        pinecone_index=os.environ.get("PINECONE_INDEX", "medium-rag"),
        pinecone_cloud=os.environ.get("PINECONE_CLOUD", "aws"),
        pinecone_region=os.environ.get("PINECONE_REGION", "us-east-1"),
        csv_path=os.environ.get("CSV_PATH", "medium-english-50mb.csv"),
        chunk_size=int(os.environ.get("CHUNK_SIZE", "512")),
        overlap_ratio=float(os.environ.get("OVERLAP_RATIO", "0.10")),
        top_k=int(os.environ.get("TOP_K", "8")),
    )

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .config import Config


def make_chat(cfg: Config) -> ChatOpenAI:
    kwargs = {
        "api_key": cfg.openai_api_key,
        "model": cfg.chat_model,
    }
    if cfg.openai_base_url:
        kwargs["base_url"] = cfg.openai_base_url
    return ChatOpenAI(**kwargs)


def chat_complete(chat: ChatOpenAI, system: str, user: str) -> str:
    resp = chat.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    content = getattr(resp, "content", "")
    if isinstance(content, list):
        content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return (content or "").strip()

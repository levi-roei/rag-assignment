from __future__ import annotations

import json
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _lib.config import load_config
from _lib.embedding import embed_query, make_embeddings
from _lib.llm import chat_complete, make_chat
from _lib.prompts import SYSTEM_PROMPT, build_user_prompt
from _lib.retrieval import dedupe_by_article, get_index, get_pinecone, query_index


def _answer(question: str) -> dict:
    cfg = load_config()
    embeddings = make_embeddings(cfg)
    chat = make_chat(cfg)
    pc = get_pinecone(cfg)
    index = get_index(pc, cfg)

    qvec = embed_query(embeddings, question)
    raw_hits = query_index(index, qvec, top_k=cfg.top_k)
    hits = dedupe_by_article(raw_hits, max_per_article=2)

    contexts = [
        {
            "article_id": h["article_id"],
            "title": h["title"],
            "author": h.get("author", ""),
            "url": h.get("url", ""),
            "chunk": h["chunk"],
            "score": round(h["score"], 4),
        }
        for h in hits
    ]

    user_prompt = build_user_prompt(question, contexts)
    response_text = chat_complete(chat, SYSTEM_PROMPT, user_prompt)

    return {
        "response": response_text,
        "context": contexts,
        "Augmented_prompt": {
            "System": SYSTEM_PROMPT,
            "User": user_prompt,
        },
    }


class handler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        try:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            data = json.loads(raw.decode("utf-8") or "{}")
            question = (data.get("question") or "").strip()
            if not question:
                self._send_json(400, {"error": "Missing 'question' in request body."})
                return
            self._send_json(200, _answer(question))
        except Exception as e:  # noqa: BLE001
            self._send_json(
                500,
                {"error": str(e), "trace": traceback.format_exc()},
            )

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _lib.config import load_config


class handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        cfg = load_config()
        payload = {
            "chunk_size": cfg.chunk_size,
            "overlap_ratio": cfg.overlap_ratio,
            "top_k": cfg.top_k,
        }
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

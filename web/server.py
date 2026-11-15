"""QuahogGPT web app: a chat page over qa.ask.QA, with streamed answers and cited sources.

Standard library only. The browser keeps the conversation and sends it with each message;
POST /api/chat answers with Server-Sent Events:
  event: meta   {"standalone": "...", "refused": false}
  event: token  "text piece"
  event: done   {"answer": "...", "sources": [{"n", "title", "url", "text"}]}
  event: error  {"error": "..."}
Model work runs on one worker thread (MLX models belong to the thread that loaded them);
requests queue for it, so answers are produced one at a time.

Usage (MLX environment, from the repo root):
    .venv-mlx/bin/python -m web.server            # http://127.0.0.1:8000
    .venv-mlx/bin/python -m web.server --port 8080
"""

import argparse
import json
import queue
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from qa.ask import QA

STATIC = (Path(__file__).parent / "static").resolve()
TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8"}
MAX_BODY = 64_000  # bytes
MAX_MESSAGE = 2_000  # characters
MAX_HISTORY = 20  # messages kept from the client's history


def clean_history(raw):
    """Valid {"role", "content"} turns only, newest MAX_HISTORY."""
    if not isinstance(raw, list):
        return []
    turns = [
        {"role": t["role"], "content": t["content"][: MAX_MESSAGE * 2]}
        for t in raw
        if isinstance(t, dict) and t.get("role") in ("user", "assistant") and isinstance(t.get("content"), str)
    ]
    return turns[-MAX_HISTORY:]


class ModelWorker(threading.Thread):
    """Owns the models. MLX binds a model to the thread that loaded it, so loading and every
    generation happen here; request threads submit jobs and relay the events."""

    def __init__(self):
        super().__init__(daemon=True)
        self.jobs = queue.Queue()
        self.ready = threading.Event()

    def run(self):
        qa = QA()
        self.ready.set()
        while True:
            history, message, events, cancel = self.jobs.get()
            stream = qa.stream(history, message)
            try:
                for event in stream:
                    if cancel.is_set():
                        stream.close()
                        break
                    events.put(event)
            except Exception as e:  # report model errors to the page instead of a dead stream
                events.put(("error", {"error": str(e)}))
            events.put(None)

    def submit(self, history, message):
        """Queue a chat job; returns (events queue ending with None, cancel event)."""
        events, cancel = queue.Queue(), threading.Event()
        self.jobs.put((history, message, events, cancel))
        return events, cancel


class Handler(BaseHTTPRequestHandler):
    worker = None

    def log_message(self, fmt, *args):  # one short line per request
        print(f"{self.command} {self.path} -> {args[1] if len(args) > 1 else ''}", flush=True)

    def send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        route = urllib.parse.urlsplit(self.path).path
        if route == "/api/health":
            return self.send_json(200, {"ok": True})
        name = "index.html" if route in ("/", "/index.html") else route.removeprefix("/static/")
        path = (STATIC / name).resolve()
        if path.parent != STATIC or path.suffix not in TYPES or not path.is_file():
            return self.send_json(404, {"error": "not found"})
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", TYPES[path.suffix])
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if urllib.parse.urlsplit(self.path).path != "/api/chat":
            return self.send_json(404, {"error": "not found"})
        length = int(self.headers.get("Content-Length") or 0)
        if not 0 < length <= MAX_BODY:
            return self.send_json(413, {"error": "request too large"})
        try:
            body = json.loads(self.rfile.read(length))
        except json.JSONDecodeError:
            return self.send_json(400, {"error": "invalid JSON"})
        message = body.get("message") if isinstance(body, dict) else None
        if not isinstance(message, str) or not message.strip() or len(message) > MAX_MESSAGE:
            return self.send_json(400, {"error": f"message must be 1-{MAX_MESSAGE} characters"})
        history = clean_history(body.get("history"))

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        events, cancel = self.worker.submit(history, message.strip())
        try:
            while (event := events.get()) is not None:
                kind, data = event
                self.wfile.write(f"event: {kind}\ndata: {json.dumps(data)}\n\n".encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            cancel.set()  # the browser left; the worker stops at the next token


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    print("loading models...", flush=True)
    Handler.worker = ModelWorker()
    Handler.worker.start()
    Handler.worker.ready.wait()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"QuahogGPT ready on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

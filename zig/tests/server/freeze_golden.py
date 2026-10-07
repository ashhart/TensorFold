"""The Python server's parity answers frozen as golden files under golden/, one normalized reply set per case."""

from __future__ import annotations

import json
import os
import signal
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE), str(ROOT / "src")]

import corpus  # noqa: E402
import python_server  # noqa: E402
from fake_text import FakeTokenizer  # noqa: E402
from wire import exchange, normal, request  # noqa: E402

FIXTURES = HERE / "fixtures"
GOLDEN = HERE / "golden"
CONTEXT = 1024


def capture(port: int, group: str, name: str, raws: list[Any], **opts: Any) -> None:
    """The Python exchange for one case, through parity's own normal(), as the case's golden file."""
    py = exchange(port, raws, **opts)
    replies = [{"status": n["status"], "headers": [[k, v] for k, v in n["headers"]],
                "body": n["body"], "interim": n["interim"]}
               for n in (normal(r) for r in py["replies"])]
    out = GOLDEN / group
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps({"closed": py["closed"], "replies": replies}, indent=1) + "\n")


def main() -> int:
    tokenizer = FakeTokenizer(json.loads((FIXTURES / "vocab.json").read_text())["pieces"])
    logs = Path(tempfile.mkdtemp())
    os.environ["TENSORFOLD_REQUEST_LOG"] = str(logs / "python.jsonl")
    server, port = python_server.start(FIXTURES, context=CONTEXT)
    try:
        cases = (corpus.chat_cases() + corpus.tool_cases() + corpus.error_cases() + corpus.completion_cases(tokenizer.encode)
                 + corpus.response_cases() + corpus.anthropic_cases() + corpus.token_cases() + corpus.status_cases())
        for group, name, raws in cases:
            capture(port, group, name, raws)
        for group, name, raws, opts in corpus.framing_cases():
            capture(port, group, name, raws, **opts)
        leave = corpus.post("/v1/chat/completions", corpus.chat("slow", stream=True))
        capture(port, "cancel", "client-leaves", [leave], leave_after_events=3)
        time.sleep(1.0)
        capture(port, "cancel", "metrics-after", [request("GET", "/metrics")])
    finally:
        server.shutdown()
        server.server_close()
    (GOLDEN / "lifecycle").mkdir(parents=True, exist_ok=True)
    (GOLDEN / "lifecycle" / "request-log.json").write_bytes((logs / "python.jsonl").read_bytes())
    (GOLDEN / "lifecycle" / "sigterm-exit-0.json").write_text('{"exit": 0}\n')

    server, port = python_server.start(FIXTURES, context=CONTEXT, keys=["sk-cli"], metrics_open=True)
    try:
        for name, path in (("metrics", "/metrics"), ("v1-metrics", "/v1/metrics/"), ("models", "/v1/models"), ("health", "/health")):
            capture(port, "metrics-open", name, [request("GET", path)])
    finally:
        server.shutdown()
        server.server_close()
    with tempfile.TemporaryDirectory() as tmp:
        key_file = Path(tmp) / "keys"
        key_file.write_text("# keys\nclient: sk-file\nsk-bare\n")
        key_file.chmod(0o600)
        server, port = python_server.start(FIXTURES, context=CONTEXT, keys=["sk-cli"], key_file=str(key_file), environment="sk-env")
        store = server.RequestHandlerClass.auth
        signal.signal(signal.SIGHUP, store.request_reload)
        try:
            chat = corpus.chat("plain")
            gets = {"health": ("/health", {}), "models-none": ("/v1/models", {}), "models-cli": ("/v1/models", {"Authorization": "Bearer sk-cli"}),
                    "models-file": ("/v1/models", {"x-api-key": "sk-file"}), "models-bare": ("/v1/models", {"Authorization": "bearer   sk-bare "}),
                    "models-env": ("/v1/models", {"x-api-key": "sk-env"}), "models-wrong": ("/v1/models", {"Authorization": "Bearer sk-nope"}),
                    "models-basic": ("/v1/models", {"Authorization": "Basic sk-cli"}), "metrics-none": ("/metrics", {}), "messages-none": ("/v1/messages", {}),
                    "tokenize-none": ("/tokenize", {}), "alt-models": ("/alt/models", {}), "options": ("/v1/models", {})}
            for name, (path, headers) in gets.items():
                method = "OPTIONS" if name == "options" else "POST" if name in ("messages-none", "tokenize-none") else "GET"
                capture(port, "keys", name, [request(method, path, b"{}" if method == "POST" else None, headers)])
            capture(port, "keys", "twice-x-api-key", [b"GET /v1/models HTTP/1.1\r\nx-api-key: sk-cli\r\nx-api-key: sk-cli\r\n\r\n"])
            capture(port, "keys", "chat-with-key", [request("POST", "/v1/chat/completions", chat, {"x-api-key": "sk-cli"})])
            capture(port, "keys", "expect-without-key", [request("POST", "/v1/chat/completions", chat, {"Expect": "100-continue"})])
            capture(port, "keys", "pooled-identity", [request("GET", "/v1/models", None, {"x-api-key": "sk-cli"}), request("GET", "/v1/models")])
            key_file.write_text("next: sk-next\n")
            os.kill(os.getpid(), signal.SIGHUP)
            time.sleep(0.3)
            capture(port, "keys", "rotated-old", [request("GET", "/v1/models", None, {"x-api-key": "sk-file"})])
            capture(port, "keys", "rotated-new", [request("GET", "/v1/models", None, {"x-api-key": "sk-next"})])
            capture(port, "keys", "metrics-counted", [request("GET", "/metrics", None, {"x-api-key": "sk-cli"})])
        finally:
            server.shutdown()
            server.server_close()
    cases = sum(1 for f in GOLDEN.rglob("*.json") if f.parent.name != "lifecycle")
    print(f"{cases} cases plus the request log and the SIGTERM exit written under {GOLDEN.relative_to(HERE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

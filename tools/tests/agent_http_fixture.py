"""A fake HTTP server usable through real loopback sockets or standard HTTP parsing over memory streams."""
import contextlib
import hashlib
import http.client
import http.server
import io
import json
import threading
import urllib.error
import urllib.parse


class Fixture:
    def __init__(self, mode="normal", cached=0, key=None):
        self.mode, self.cached, self.key = mode, cached, key
        self.model = "fixture"
        self.seen, self.chat = [], []

    def response(self, path, body, authorization):
        self.seen.append((path, body, authorization))
        if self.key and authorization != "Bearer " + self.key:
            return 401, {"error": "credential required"}
        if path == "/v1/models":
            return 200, {"data": [{"id": self.model, "capabilities": {"thinking_switch": self.mode != "thinking_default"}}]}
        if path == "/tokenize":
            return 200, {"tokens": [True] if self.mode == "bad_tokenizer" else list(body["prompt"].encode())}
        if path == "/detokenize":
            return 200, {"prompt": bytes(body["tokens"]).decode()}
        if path != "/v1/chat/completions":
            return 404, {"error": "unknown route"}
        self.chat.append(body)
        if self.mode == "http_error":
            return 503, {"error": self.key or "unavailable"}
        count = len(self.chat)
        prompt = 10000 + count
        cached = 0 if count == 1 else self.cached
        if self.mode == "bad_cache":
            cached = True
        elif self.mode == "too_large_cache":
            cached = prompt + 1
        elif self.mode == "negative_cache":
            cached = -1
        elif self.mode == "float_cache":
            cached = 1.0
        elif self.mode == "string_cache":
            cached = "1"
        usage = {"prompt_tokens": prompt, "completion_tokens": 2,
                 "prompt_tokens_details": {"cached_tokens": cached}}
        if self.mode == "bad_prompt":
            usage["prompt_tokens"] = "10000"
        elif self.mode == "missing_cache":
            usage.pop("prompt_tokens_details")
        token_sha = hashlib.sha256(json.dumps({k: v for k, v in body.items() if k != "model"}, sort_keys=True).encode()).hexdigest()
        if self.mode == "mismatch":
            token_sha = "f" * 64
        if self.mode == "echo_key":
            token_sha = self.key
        runtime = {"token_sha": token_sha[:12], "token_sha256": token_sha, "ignored": self.key}
        if self.mode == "no_hash":
            runtime.pop("token_sha256")
        if self.mode == "tool_only":
            delta = {"tool_calls": [{"index": 0, "type": "function", "function": {"name": "run", "arguments": "{}"}}]}
        else:
            delta = {"reasoning_content": "fixture thought"} if self.mode == "reasoning" else {"content": "MODEL-PRIVATE-REPLY"}
        if self.mode == "echo_key_text":
            delta = {"content": self.key}
        chunks = [] if self.mode == "no_output" else [{"choices": [{"index": 0, "delta": delta, "finish_reason": "stop"}]}]
        if self.mode == "role_first":
            chunks.insert(0, {"choices": [{"index": 0, "delta": {"role": "assistant"}}]})
        final = {"choices": [], "tensorfold": runtime}
        if self.mode != "missing_usage":
            final["usage"] = usage
        chunks.append(final)
        wire = b"".join(b"data: " + json.dumps(chunk).encode() + b"\n\n" for chunk in chunks)
        if self.mode != "incomplete":
            wire += b"data: [DONE]\n\n"
        return 200, wire


def handler(fixture):
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, body):
            status, value = fixture.response(self.path, body, self.headers.get("Authorization"))
            wire = value if isinstance(value, bytes) else json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "text/event-stream" if isinstance(value, bytes) else "application/json")
            self.send_header("Content-Length", str(len(wire)))
            self.end_headers()
            self.wfile.write(wire)

        def do_GET(self):
            self.respond(None)

        def do_POST(self):
            self.respond(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
    return Handler


class MemorySocket:
    def __init__(self, incoming):
        self.incoming, self.outgoing = io.BytesIO(incoming), bytearray()

    def makefile(self, *args):
        return self.incoming

    def sendall(self, value):
        self.outgoing.extend(value)


class MemoryHTTP:
    def __init__(self, fixture):
        self.fixture, self.previous = fixture, None

    def __call__(self, request, **kwargs):
        if self.previous is not None and not self.previous.isclosed():
            raise AssertionError("overlapping HTTP requests")
        path = urllib.parse.urlsplit(request.full_url).path
        body = request.data or b""
        headers = dict(request.header_items()) | {"Content-Length": str(len(body))}
        raw = (request.get_method() + " " + path + " HTTP/1.0\r\n"
               + "".join(f"{k}: {v}\r\n" for k, v in headers.items()) + "\r\n").encode() + body
        sock = MemorySocket(raw)
        server = type("Server", (), {"server_name": "fixture", "server_port": 0})()
        handler(self.fixture)(sock, ("127.0.0.1", 0), server)
        response = http.client.HTTPResponse(MemorySocket(bytes(sock.outgoing)), method=request.get_method())
        response.begin()
        self.previous = response
        if response.status >= 400:
            raise urllib.error.HTTPError(request.full_url, response.status, "fixture error", response.headers, response)
        return response


@contextlib.contextmanager
def loopback(fixture):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler(fixture))
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

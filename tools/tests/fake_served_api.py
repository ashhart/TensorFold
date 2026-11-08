"""A fake served API for the conformance kit's tests: deterministic chat replies on loopback, with planted faults.

A clean fake passes every gate. Each fault breaks one property the kit gates, and only that one:

- drafted: a drafted reply to the copy prompt differs from its plain reply.
- repeat: a drafted reply to the sky prompt differs the second time that request arrives.
- concurrent: a reply differs while another request is in flight.
- resumed: the cached server reports no cached prompt tokens.
- refusals: an out-of-range presence_penalty is answered instead of refused.

Standard library only; no device, no network beyond 127.0.0.1.
"""
import contextlib
import hashlib
import http.server
import json
import threading

FAULTS = ("drafted", "repeat", "concurrent", "resumed", "refusals")
BATCH_WINDOW = 0.15  # seconds a lone request waits for a partner under the concurrent fault


def doc(sha="a" * 64, cached=0, accepted=2, drafted=3, text="hi"):
    """A chat completion body carrying the fields the kit reads."""
    return {"choices": [{"message": {"content": text}}], "tensorfold": {"token_sha256": sha},
            "speculative": {"accepted": accepted, "drafted": drafted},
            "usage": {"prompt_tokens": 9, "completion_tokens": 4, "prompt_tokens_details": {"cached_tokens": cached}}}


def refusal(body, fault=None):
    """The error a real server gives for the profile's refused requests, else None."""
    if body.get("n", 1) != 1:
        return "n must be 1; multiple choices are not supported"
    if "top_logprobs" in body and not body.get("logprobs"):
        return "top_logprobs requires logprobs: true"
    if abs(body.get("presence_penalty", 0)) > 2 and fault != "refusals":
        return "presence_penalty must be between -2 and 2, or null"
    if body.get("max_tokens", 0) > 100000:
        return "This server's maximum context length is 16384 tokens"
    return None


def text_of(body) -> str:
    return " ".join(m.get("content", "") for m in body.get("messages", []))


class FakeEngine(http.server.BaseHTTPRequestHandler):
    """The reply hash depends on the messages and, when sampling, the seed; never on draft, cache or batch."""

    def log_message(self, *_):
        pass

    def do_GET(self):
        self.reply(200, {"object": "list", "data": [{"id": "fake", "object": "model"}]})

    def do_POST(self):
        server = self.server
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        error = refusal(body, server.fault)
        if error:
            return self.reply(400, {"error": {"message": error, "type": "invalid_request_error"}})
        key = json.dumps(body, sort_keys=True)
        with server.lock:
            server.seen[key] = server.seen.get(key, 0) + 1
            arrival = server.seen[key]
            server.inflight += 1
        try:
            batched = self.batched() if server.fault == "concurrent" else False
            seed = body["seed"] if body.get("temperature") else None
            salt = self.salt(body, arrival, batched)
            sha = hashlib.sha256(json.dumps([body["messages"], seed, salt]).encode()).hexdigest()
            cached = 5 if server.cache and server.fault != "resumed" and len(body["messages"]) > 1 else 0
            accepted = 2 if body.get("draft", True) else 0
            self.reply(200, doc(sha=sha, cached=cached, accepted=accepted, drafted=accepted, text="r" + sha[:6]))
        finally:
            with server.lock:
                server.inflight -= 1

    def batched(self) -> bool:
        """True when another request shares the server with this one within the batch window."""
        server = self.server
        with server.cond:
            server.cond.notify_all()
            return server.cond.wait_for(lambda: server.inflight > 1, timeout=BATCH_WINDOW)

    def salt(self, body, arrival, batched):
        """A planted fault changes the reply by adding a salt to its hash; None leaves it correct."""
        fault, draft = self.server.fault, body.get("draft", True)
        if fault == "drafted" and draft and "Repeat exactly" in text_of(body):
            return "drafted"
        if fault == "repeat" and draft and "sky" in text_of(body) and arrival == 2:
            return "repeat"
        if fault == "concurrent" and batched:
            return "batched"
        return None

    def reply(self, status, payload):
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@contextlib.contextmanager
def fake(cache, fault=None):
    """A running fake on 127.0.0.1 port 0, yielding its base URL; cache=False is the cache-off server."""
    if fault not in (None, *FAULTS):
        raise ValueError(f"unknown fault {fault}")
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeEngine)
    server.cache, server.fault, server.seen, server.inflight = cache, fault, {}, 0
    server.lock = threading.Lock()
    server.cond = threading.Condition(server.lock)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()

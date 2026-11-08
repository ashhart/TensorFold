"""A fake served API for the tools tests: deterministic chat replies on loopback, with planted faults or a script.

A clean fake passes every gate. Each fault breaks one property the kit gates, and only that one:

- drafted: a drafted reply to the copy prompt differs from its plain reply.
- repeat: a drafted reply to the sky prompt differs the second time that request arrives.
- concurrent: a reply differs while another request is in flight.
- resumed: the cached server reports no cached prompt tokens.
- refusals: an out-of-range presence_penalty is answered instead of refused.

A request with ``"stream": true`` gets the same reply as SSE, as the server streams it with ``include_usage``: the
content in pieces, a finishing chunk carrying ``tensorfold`` and ``speculative``, a usage-only chunk, then ``[DONE]``.

``serve`` plays a scripted SSE reply instead, for the benchmark clients' tests. A reply is a list of tokens, each
``(choice, kind, text)`` for content and reasoning or ``(choice, "tool", index, field, text)`` for a tool call's
``name`` or ``arguments``. ``script`` cuts the same tokens into SSE events of the given sizes, so a test can serve one
reply chunked several ways. Headers go out before the script's first step.

Standard library only; no device, no network beyond 127.0.0.1.
"""
import contextlib
import hashlib
import http.server
import json
import threading
import time

FAULTS = ("drafted", "repeat", "concurrent", "resumed", "refusals")
BATCH_WINDOW = 0.15  # seconds a lone request waits for a partner under the concurrent fault
DONE = b"data: [DONE]\n\n"
HASHES = {"token_sha": "d" * 12, "token_sha256": "d" * 64}


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


def event(chunk: dict) -> bytes:
    return b"data: " + json.dumps(chunk).encode() + b"\n\n"


def streamed(payload: dict) -> list:
    """``payload`` (a ``doc``) as server-sent events: content two characters an event, the finish, usage, [DONE]."""
    text = payload["choices"][0]["message"]["content"]
    pieces = [{"choices": [{"index": 0, "delta": {"content": text[i:i + 2]}, "finish_reason": None}]}
              for i in range(0, len(text), 2)]
    last = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "tensorfold": payload["tensorfold"],
            "speculative": payload["speculative"]}
    return [event(c) for c in pieces + [last, {"choices": [], "usage": payload["usage"]}]] + [DONE]


def delta(group: list[tuple], label: str) -> list[dict]:
    """One event's choices: each choice's tokens merged into one delta; tool fragments merged by tool index."""
    choices: dict[int, dict] = {}
    for token in group:
        d = choices.setdefault(token[0], {})
        if token[1] == "tool":
            _, _, index, field, text = token
            calls = d.setdefault("tool_calls", [])
            call = next((c for c in calls if c["index"] == index), None)
            if call is None:
                call = {"index": index, "id": f"call_{label}_{index}", "type": "function", "function": {}}
                calls.append(call)
            call["function"][field] = call["function"].get(field, "") + text
        else:
            key = "content" if token[1] == "content" else "reasoning_content"
            d[key] = d.get(key, "") + token[2]
    return [{"index": c, "delta": d, "finish_reason": None} for c, d in choices.items()]


def script(tokens: list[tuple], sizes: list[int], *, finish: str | None = "stop", gap: float = 0.0,
           before: float = 0.0, usage_after: float = 0.0, error_after: int | None = None, done: bool = True) -> list:
    """Steps for the handler: bytes are written and flushed, a float sleeps that many seconds."""
    steps: list = [before] if before else []
    at, k = 0, 0
    while at < len(tokens):
        if error_after is not None and at >= error_after:
            return steps + [event({"error": {"message": "fixture failure", "type": "server_error"}})]
        group = tokens[at:at + sizes[k % len(sizes)]]
        steps += [event({"choices": delta(group, "-".join(map(str, sizes)))})] + ([gap] if gap else [])
        at, k = at + len(group), k + 1
    if finish is not None:
        ends = sorted({t[0] for t in tokens}) or [0]
        steps.append(event({"choices": [{"index": c, "delta": {}, "finish_reason": finish} for c in ends]}))
    if usage_after:
        steps.append(usage_after)
    steps.append(event({"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": len(tokens),
                                                 "prompt_tokens_details": {"cached_tokens": 0}}, "tensorfold": HASHES}))
    return steps + ([DONE] if done else [])


class FakeEngine(http.server.BaseHTTPRequestHandler):
    """The reply hash depends on the messages and, when sampling, the seed; never on draft, cache, batch or stream."""

    def log_message(self, *_):
        pass

    def do_GET(self):
        self.reply(200, {"object": "list", "data": [{"id": "fake", "object": "model"}]})

    def do_POST(self):
        server = self.server
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if server.steps is not None:
            return self.play(server.steps(body))
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
            payload = doc(sha=sha, cached=cached, accepted=accepted, drafted=accepted, text="r" + sha[:6])
        finally:
            with server.lock:
                server.inflight -= 1
        # Out of flight before the reply is written: the client's next request must not find this one still counted.
        if body.get("stream"):
            self.play(streamed(payload))
        else:
            self.reply(200, payload)
        time.sleep(server.linger)

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

    def play(self, steps):
        """An event stream: headers first, then each step (bytes written and flushed, a float slept)."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        try:
            for step in steps:
                if isinstance(step, bytes):
                    self.wfile.write(step)
                    self.wfile.flush()
                else:
                    time.sleep(step)
        except (BrokenPipeError, ConnectionResetError):
            pass


@contextlib.contextmanager
def running(cache=False, fault=None, linger=0.0, steps=None):
    """One fake on 127.0.0.1 port 0, yielding its base URL; stopped on exit."""
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeEngine)
    server.cache, server.fault, server.seen, server.inflight, server.linger = cache, fault, {}, 0, linger
    server.steps = steps
    server.lock = threading.Lock()
    server.cond = threading.Condition(server.lock)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@contextlib.contextmanager
def fake(cache, fault=None, linger=0.0):
    """A running fake on 127.0.0.1 port 0, yielding its base URL; cache=False is the cache-off server."""
    # linger: seconds each handler stays alive after its reply, widening any window between reply and exit.
    if fault not in (None, *FAULTS):
        raise ValueError(f"unknown fault {fault}")
    with running(cache, fault, linger) as url:
        yield url


def serve(steps):
    """A fake that answers each POST by playing ``steps(body)``; yields its base URL."""
    return running(steps=steps)

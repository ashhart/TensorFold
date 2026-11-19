"""A fake served API for the tools tests: deterministic chat replies on loopback, with planted faults or a script.

A clean fake passes every gate. Each fault breaks one property the kit gates, and only that one:

- drafted: a drafted reply to the copy prompt differs from its plain reply.
- repeat: a drafted reply to the sky prompt differs the second time that request arrives.
- concurrent: a reply differs while another request is in flight.
- resumed: a later chat turn reports no cached tokens unless its whole prompt was asked before.
- refusals: an out-of-range presence_penalty is answered instead of refused.
- cancel_lane: a streamed reply the client leaves keeps its lane busy.
- cancel_free: a streamed reply the client leaves ends, but its lane never returns to the free ones.
- cancel_reply: the request a client left answers differently the next time it arrives.
- cancel_health: a server a client left answers ``/health`` with an error.
- evict: a chat turn pushed out of the prompt cache reports it (0 cached) but answers from the stale state.
- undrafted: no request drafts, so a drafted reply runs the plain path.
- plain_drafts: a ``"draft": false`` request drafts anyway, so both arms run the same drafted path.
- sampled_drafts: a sampled request drafts nothing, so drafting is exercised only greedy.
- unsampled: temperature and seed never reach the sampler; a sampled request gets the greedy reply.
- serial: the server takes one request at a time, so requests sent together never overlap in it.
- empty: every reply is empty (no tokens), so every pair of replies hashes the same.

``running(..., warm=S)`` keeps the server busy (one running, more waiting) for ``S`` seconds after each reply, as a
server that prefills the chat's next turn in the background; ``lanes_free=False`` leaves ``capacity.lanes_free`` out
of ``GET /v1/load``, as a server without lanes.

A request is counted in ``GET /v1/load``'s running and active chat requests while open. One that finds another in
flight within ``PARTNER_WINDOW`` stays open ``HOLD`` seconds more; a request alone leaves after the window. While a
client polls the census (one read within ``WATCHED`` seconds), a pair stays open until a census counts two open at
once (at most ``WATCHED`` seconds), so however slow a census read is, one taken meanwhile sees them.

The cached server reports five cached tokens for a later chat turn (more than one message), and for a one-message
prompt it answered before. It keeps the prompts it
answered in an LRU of ``CAPACITY`` entries; a prompt pushed out reports 0 until it is asked again. ``GET /v1/load``,
``/metrics`` and ``/health`` show the lanes open streams hold, the disconnects counted and the server's health.

A request with ``"stream": true`` gets the same reply as SSE, as the server streams it with ``include_usage``: one
event a token, ``TOKEN_GAP`` apart (the content two characters an event, empty once the text is out), a finishing
chunk carrying ``tensorfold`` and ``speculative``, a usage-only chunk, then ``[DONE]``. A client that leaves a stream
is counted as a disconnect, and its lane is freed.

``serve`` plays a scripted SSE reply instead, for the benchmark clients' tests. A reply is a list of tokens, each
``(choice, kind, text)`` for content and reasoning or ``(choice, "tool", index, field, text)`` for a tool call's
``name`` or ``arguments``. ``script`` cuts the same tokens into SSE events of the given sizes, so a test can serve one
reply chunked several ways. Headers go out before the script's first step.

Standard library only; no device, no network beyond 127.0.0.1.
"""
import collections
import contextlib
import hashlib
import http.server
import json
import threading
import time

FAULTS = ("drafted", "repeat", "concurrent", "resumed", "refusals", "cancel_lane", "cancel_free", "cancel_reply",
          "cancel_health", "evict", "undrafted", "plain_drafts", "sampled_drafts", "unsampled", "serial", "empty")
BATCH_WINDOW = 0.15  # seconds a lone request waits for a partner under the concurrent fault
DONE = b"data: [DONE]\n\n"
HASHES = {"token_sha": "d" * 12, "token_sha256": "d" * 64}
CAPACITY = 4  # prompt cache entries the cached server keeps
LANES = 8
TOKEN_GAP = 0.0002  # seconds between streamed events
PARTNER_WINDOW = 0.01  # seconds a request waits for another to share the server with
HOLD = 0.025  # seconds a request that shares the server stays open, when no census is being read
WATCHED = 0.25  # seconds a census read counts as watching, and the most a pair then waits to be counted


def doc(sha="a" * 64, cached=0, accepted=2, drafted=3, text="hi", tokens=4, sampling="greedy", prompt=9):
    """A chat completion body carrying the fields the kit reads."""
    return {"choices": [{"message": {"content": text}}], "tensorfold": {"token_sha256": sha, "sampling": sampling},
            "speculative": {"rounds": int(drafted > 0), "accepted": accepted, "drafted": drafted},
            "usage": {"prompt_tokens": prompt, "completion_tokens": tokens,
                      "prompt_tokens_details": {"cached_tokens": cached}}}


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


def messages_of(body) -> list:
    """A chat request's messages; a completion request's prompt as one user message."""
    return body.get("messages") or [{"role": "user", "content": body.get("prompt", "")}]


def text_of(body) -> str:
    return " ".join(m.get("content", "") for m in messages_of(body))


def event(chunk: dict) -> bytes:
    return b"data: " + json.dumps(chunk).encode() + b"\n\n"


def streamed(payload: dict, tokens: int = 0, gap: float = 0.0) -> list:
    """``payload`` (a ``doc``) as ``play`` steps: ``tokens`` content events ``gap`` apart, the finish, usage, [DONE]."""
    text = payload["choices"][0]["message"]["content"]
    pieces = [{"choices": [{"index": 0, "delta": {"content": text[i:i + 2]}, "finish_reason": None}]}
              for i in range(0, max(len(text), 2 * tokens), 2)]
    last = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "tensorfold": payload["tensorfold"],
            "speculative": payload["speculative"]}
    steps = [s for c in pieces for s in (event(c), gap) if s]
    return steps + [event(c) for c in (last, {"choices": [], "usage": payload["usage"]})] + [DONE]


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


class FixtureHTTPServer(http.server.ThreadingHTTPServer):
    request_queue_size = 128  # Allow the full concurrent batch to connect while handler threads start.


class FakeEngine(http.server.BaseHTTPRequestHandler):
    """The reply hash depends on the messages and, when sampling, the seed; never on draft, cache, batch or stream."""

    def log_message(self, *_):
        pass

    def do_GET(self):
        server = self.server
        if self.path == "/v1/load":
            with server.lock:
                busy, warming = server.streams + server.leaked + server.inflight, time.monotonic() < server.warm_until
                server.census_at = time.monotonic()
                if busy > 1:
                    server.overlaps += 1
                    server.cond.notify_all()
            capacity = {"mode": "window", "lanes_free": LANES - busy - server.lost} if server.lanes_free else {}
            return self.reply(200, {"lanes": LANES, "running": busy + warming, "waiting": 19 * warming,
                                    "http_active": {"chat_completions": busy}, "capacity": capacity})
        if self.path == "/metrics":
            return self.reply(200, f"tensorfold:client_disconnections_total {server.disconnects}\n")
        if self.path == "/health":
            sick = server.fault == "cancel_health" and server.disconnects
            return self.reply(503 if sick else 200, {"status": "error" if sick else "ok"})
        self.reply(200, {"object": "list", "data": [{"id": "fake", "object": "model"}]})

    def do_POST(self):
        server = self.server
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if server.steps is not None:
            return self.play(server.steps(body))
        error = refusal(body, server.fault)
        if error:
            return self.reply(400, {"error": {"message": error, "type": "invalid_request_error"}})
        key = json.dumps({k: v for k, v in body.items() if k not in ("stream", "stream_options")}, sort_keys=True)
        with server.one if server.fault == "serial" else contextlib.nullcontext():
            payload = self.answer(body, key)
            # Out of flight before the reply is written: the client's next request must not find this one counted.
            if body.get("stream"):
                self.play(streamed(payload, body.get("max_tokens", 16), TOKEN_GAP), key)
            else:
                self.reply(200, payload)
        with server.lock:
            server.warm_until = time.monotonic() + server.warm
        time.sleep(server.linger)

    def answer(self, body, key) -> dict:
        """The reply's body, computed while the request is counted in flight."""
        server, messages = self.server, messages_of(body)
        with server.lock:
            server.seen[key] = server.seen.get(key, 0) + 1
            arrival = server.seen[key]
            server.inflight += 1
            after_cut = key in server.cut
            server.cut.discard(key)
        try:
            watched = time.monotonic() - server.census_at < WATCHED
            batched = self.batched(BATCH_WINDOW if server.fault == "concurrent" else PARTNER_WINDOW)
            if batched:
                self.counted() if watched else time.sleep(HOLD)
            sampled = bool(body.get("temperature")) and server.fault != "unsampled"
            cached, stale = self.lookup(messages)
            salt = "stale" if stale else "after a cut" if after_cut and server.fault == "cancel_reply" else None
            salt = salt or self.salt(body, arrival, batched and server.fault == "concurrent")
            seed = body.get("seed") if sampled else None
            sha = hashlib.sha256(json.dumps([messages, seed, salt]).encode()).hexdigest()
            self.keep(messages)
            drafts = (body.get("draft", True) or server.fault == "plain_drafts") and server.fault != "undrafted"
            accepted = 2 if drafts and not (sampled and server.fault == "sampled_drafts") else 0
            if server.fault == "empty":
                return doc(sha=hashlib.sha256(b"").hexdigest(), cached=cached, accepted=accepted, drafted=accepted,
                           text="", tokens=0)
            return doc(sha=sha, cached=cached, accepted=accepted, drafted=accepted, text="r" + sha[:6],
                       sampling="exact" if sampled else "greedy", prompt=max(9, len(text_of(body)) // 4))
        finally:
            with server.lock:
                server.inflight -= 1

    def lookup(self, messages):
        """Cached tokens for the prompt, and whether the evict fault answers from an evicted prompt."""
        server = self.server
        if not server.cache:
            return 0, False
        with server.lock:
            whole = json.dumps(messages)
            if len(messages) < 2:
                return (5 if whole in server.kept else 0), False
            if whole in server.evicted:
                return 0, server.fault == "evict"
            return (0 if server.fault == "resumed" and whole not in server.kept else 5), False

    def keep(self, messages):
        """The prompt enters the LRU; the oldest prompts leave it."""
        server = self.server
        if not server.cache:
            return
        with server.lock:
            whole = json.dumps(messages)
            server.kept[whole] = True
            server.kept.move_to_end(whole)
            server.evicted.discard(whole)
            while len(server.kept) > CAPACITY:
                server.evicted.add(server.kept.popitem(last=False)[0])

    def batched(self, window: float) -> bool:
        """True when another request shares the server with this one within ``window`` seconds."""
        server = self.server
        with server.cond:
            server.cond.notify_all()
            return server.cond.wait_for(lambda: server.inflight > 1, timeout=window)

    def counted(self):
        """Wait, at most ``WATCHED`` seconds, for a ``GET /v1/load`` that counts two chat requests open at once."""
        server = self.server
        with server.cond:
            mark = server.overlaps
            server.cond.wait_for(lambda: server.overlaps > mark, timeout=WATCHED)

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
        raw = payload.encode() if isinstance(payload, str) else json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def play(self, steps, key=None):
        """An event stream: headers first, then each step (bytes written and flushed, a float slept)."""
        server = self.server
        with server.lock:
            server.streams += 1
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for step in steps:
                if isinstance(step, bytes):
                    self.wfile.write(step)
                    self.wfile.flush()
                else:
                    time.sleep(step)
        except ConnectionError:  # Windows raises ConnectionAbortedError (10053) on a cut
            with server.lock:
                server.disconnects += 1
                server.cut.add(key)
                server.leaked += server.fault == "cancel_lane"
                server.lost += server.fault == "cancel_free"
        finally:
            with server.lock:
                server.streams -= 1
            self.close_connection = True


@contextlib.contextmanager
def running(cache=False, fault=None, linger=0.0, steps=None, warm=0.0, lanes_free=True):
    """One fake on 127.0.0.1 port 0, yielding its base URL; stopped on exit."""
    server = FixtureHTTPServer(("127.0.0.1", 0), FakeEngine)
    server.cache, server.fault, server.seen, server.inflight, server.linger = cache, fault, {}, 0, linger
    server.steps, server.kept, server.evicted, server.cut = steps, collections.OrderedDict(), set(), set()
    server.streams = server.leaked = server.lost = server.disconnects = server.overlaps = 0
    server.warm, server.warm_until, server.lanes_free, server.census_at = warm, 0.0, lanes_free, -1e9
    server.lock, server.one = threading.Lock(), threading.Lock()
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
def fake(cache, fault=None, linger=0.0, **shape):
    """A running fake on 127.0.0.1 port 0, yielding its base URL; cache=False is the cache-off server."""
    # linger: seconds each handler stays alive after its reply, widening any window between reply and exit.
    if fault not in (None, *FAULTS):
        raise ValueError(f"unknown fault {fault}")
    # shape: warm and lanes_free, see ``running``.
    with running(cache, fault, linger, **shape) as url:
        yield url


def serve(steps):
    """A fake that answers each POST by playing ``steps(body)``; yields its base URL."""
    return running(steps=steps)

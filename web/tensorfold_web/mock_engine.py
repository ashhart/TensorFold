"""A stand-in for `tensorfold serve`, for developing TensorFold Web on a machine that cannot run a model."""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar

MODEL = "Qwen3.8-27B"
REPLY = ("This is the mock engine that ships with TensorFold Web. It streams a canned reply at about 27 tokens "
         "a second so the Chat and Monitor screens can be developed without a Mac or a GPU. The real TensorFold "
         "answers through the same OpenAI-compatible API, so nothing in the page changes when you point it at one.")


class State:
    started = time.time()
    requests = 0
    tokens = 0
    decode_s = 0.0
    live = 0
    seen: ClassVar[set[str]] = set()              # prompt openings already read: the prompt cache
    lock = threading.Lock()
    # a hybrid model's template, for the tests: fields it refuses with a 400, and the fields that turn thinking off
    # (in chat_template_kwargs or at the top level); with `thinks` on, a request that turns nothing off reasons first
    thinks = False
    refuses: ClassVar[set[str]] = set()
    off_by: ClassVar[set[str]] = {"enable_thinking"}
    learn_seconds = 0.05                           # how long the mock "learns" each fact (the tests of a stop raise it)
    # another engine, for the tests: no `tensorfold` block on replies (so no prefill time), owned by someone else
    plain = False
    no_prompt_scores = False                       # an engine without `prompt_logprobs`, as on Metal: a 400
    # Living Weights, as TensorFold 1.0.7 serves it with --slide: facts learned in turn, a fact naming "unknowable"
    # missed; questions sharing a long word with a kept fact are answered with it
    facts: ClassVar[list[dict]] = []

WINDOW = 32768
PREFILL_TPS = 1420.0


def prompt_of(body: dict) -> str:
    return "\n".join(str(m.get("content", "")) for m in body.get("messages") or []) or str(body.get("prompt", ""))


def _off(value: object) -> bool:
    return value is False or value == "disabled" or (isinstance(value, dict) and value.get("type") == "disabled")


def thinking_of(body: dict) -> bool | None:
    """None when the template refuses the request's fields, else whether the reply reasons first."""

    kwargs = body.get("chat_template_kwargs") or {}
    if set(kwargs) & State.refuses:
        return None
    turned_off = any(_off(kwargs.get(k)) or _off(body.get(k)) for k in State.off_by)
    return State.thinks and not turned_off


def _handler(rate_tps: float, api_key: str | None = None):
    class H(BaseHTTPRequestHandler):
        def _authorized(self) -> bool:
            if not api_key:
                return True
            auth = self.headers.get("Authorization", "")
            return auth == f"Bearer {api_key}" or self.headers.get("x-api-key") == api_key

        def log_message(self, *a):  # quiet
            pass

        def _json(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            p = self.path.split("?")[0].rstrip("/")
            if p in ("", "/health", "/v1/health"):
                if api_key:                       # like TensorFold once a key is on: a bare status, nothing else
                    return self._json(200, {"status": "ok"})
                with State.lock:
                    live = State.live
                self._json(200, {"status": "ok", "model": MODEL, "model_ids": [MODEL], "max_batch_size": 8,
                                 "warming": time.time() - State.started < 3,
                                 "memory": {"active": int(23.4 * 1024**3), "cache": int(1.2 * 1024**3),
                                            "peak": int(24.6 * 1024**3)},
                                 "live": {"connections": live, "waiting": 0,
                                          "decode_tokens_per_second": rate_tps if live else 0.0,
                                          "prefill_tokens_per_second": 1420.0 if live else 0.0},
                                 "mock": True})
            elif p.endswith("/models"):
                if not self._authorized():
                    return self._json(401, {"error": {"message": "invalid API key"}})
                self._json(200, {"object": "list", "data": [{"id": MODEL, "object": "model", "owned_by": "library" if State.plain else "tensorfold"}]})
            elif p == "/slide":                         # TensorFold 1.0.7's page: absolute paths, a key it may ask
                body = (b'<html><script type="module">import * as data from "/slide/data.js"; data.start();</script>'
                        b'</html>')
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif p == "/slide/data.js":
                body = b'export function start() { fetch("/v1/slide/graph").then((r) => r.json()); }\n'
                self.send_response(200)
                self.send_header("Content-Type", "text/javascript")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif p == "/v1/slide/graph":
                self._json(200, {"facts": list(State.facts), "model": {"layers": 52}, "next": len(State.facts) + 1})
            elif p in ("/metrics", "/v1/metrics"):       # TensorFold's names; behind the key like the real page
                if not self._authorized():
                    return self._json(401, {"error": {"message": "invalid API key"}})
                body = (f"tensorfold:num_requests_running {State.live}\ntensorfold:num_requests_waiting 0\n"
                        f"tensorfold:requests_total{{key=\"none\",status=\"200\"}} {State.requests}\n"
                        f"tensorfold:prompt_tokens_total {200 * State.requests}\n"
                        f"tensorfold:generation_tokens_total {State.tokens}\n"
                        f"tensorfold:request_decode_seconds_sum {State.decode_s:.3f}\n"
                        f"tensorfold:mtp_drafted_total {State.tokens * 2}\ntensorfold:mtp_accepted_total {State.tokens}\n"
                        f"tensorfold:process_footprint_bytes {int(23.4 * 1024**3)}\n").encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self._json(404, {"error": {"message": f"unknown path {self.path}"}})

        def _learn(self, body: dict) -> None:
            text = str(body.get("text") or "")
            if not text:
                return self._json(400, {"error": {"message": "text is required"}})
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()

            def event(kind: str, data: dict) -> None:
                self.wfile.write(f"event: {kind}\ndata: {json.dumps(data)}\n\n".encode())
                self.wfile.flush()
            facts = [s.strip() + "." for s in text.split(".") if s.strip()]
            ids = []
            for f in facts:
                with State.lock:
                    State.facts.append({"id": len(State.facts) + 1, "text": f, "state": "queued", "neurons": [],
                                        "questions": [], "source": body.get("source")})
                    ids.append(State.facts[-1])
                event("fact", {"id": ids[-1]["id"], "text": f})
            # each fact's fate is settled before the events go out, so a client that leaves early finds the same graph:
            # like an engine that had already learned what it was learning when the request went away
            for fact in ids:
                kept = "unknowable" not in fact["text"].lower()
                fact["state"] = "learned" if kept else "missed"
                fact["neurons"] = [{"layer": 20 + k, "index": 0} for k in range(10)] if kept else []
            try:
                for fact in ids:
                    event("learning", {"id": fact["id"]})
                    time.sleep(State.learn_seconds)
                    kept = fact["state"] == "learned"
                    event("learned", {"id": fact["id"], "recalled": kept, "steps": 1,
                                      **({} if kept else {"message": "too few clean answers to learn from"})})
                self.wfile.write(b"data: [DONE]\n\n")
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _prompt_logprobs(self, path: str, body: dict) -> None:
            """vLLM's `prompt_logprobs`, as TensorFold 1.0.8 answers it on CUDA: the prompt split into words, each
            likely at 0.6 after the first, then the best others; refused where the real engine refuses it."""

            n = body["prompt_logprobs"]
            if State.no_prompt_scores:
                return self._json(400, {"error": {"message": "prompt_logprobs is not supported by this engine"}})
            if path != "/v1/completions" or body.get("stream"):
                return self._json(400, {"error": {"message": "prompt_logprobs are supported on /v1/completions "
                                                             "without streaming only"}})
            if isinstance(n, bool) or not isinstance(n, int) or not 0 <= n <= 20:
                return self._json(400, {"error": {"message": "prompt_logprobs must be an integer between 0 and 20, or null"}})
            words = str(body.get("prompt") or "").split(" ")
            toks = [w if i == 0 else " " + w for i, w in enumerate(words)]

            def tid(tok: str) -> str:
                return str(int(hashlib.sha256(tok.encode()).hexdigest()[:6], 16))

            def row(tok: str) -> dict:
                out = {tid(tok): {"logprob": math.log(0.6), "rank": 1, "decoded_token": tok}}
                for k in range(1, n):
                    out[str(k)] = {"logprob": math.log(0.3 / 2 ** k), "rank": k + 1, "decoded_token": f" alt{k}"}
                return out
            with State.lock:
                State.requests += 1
            return self._json(200, {"id": "cmpl-mock-prompt", "object": "text_completion", "model": MODEL,
                                    "choices": [{"index": 0, "text": " The", "finish_reason": "length", "logprobs": None,
                                                 "prompt_logprobs": [None, *(row(t) for t in toks[1:])]}],
                                    "usage": {"prompt_tokens": len(toks), "completion_tokens": 1,
                                              "total_tokens": len(toks) + 1},
                                    **({} if State.plain else {"tensorfold": {"engine": "mock",
                                                                              "drafts": body.get("draft") is not False}})})

        def _living_reply(self, body: dict) -> str | None:
            """TensorFold Web's preview and its checks, answered the way a model would."""

            msgs = body.get("messages") or []
            system = next((str(m.get("content")) for m in msgs if m.get("role") == "system"), "")
            user = str(msgs[-1].get("content") if msgs else "")
            if (body.get("response_format") or {}).get("json_schema", {}).get("name") == "facts":
                sentences = [s.strip() for s in user.split(".") if s.strip()]
                return json.dumps({"facts": [{"fact": s + ".", "subject": "the user" if s.lower().startswith(("my ", "i "))
                                              else s.split()[0], "question": f"What about {s.split()[-1]}?"}
                                             for s in sentences]})
            if system.startswith("Answer yes or no"):
                claim = user.rsplit("\"", 2)[-2] if user.count("\"") >= 2 else user
                return "no" if "founded by" in claim.lower() else "yes"
            kept = [f["text"] for f in State.facts if f["state"] == "learned"]
            words = {w.strip("?.,").lower() for w in user.split() if len(w.strip("?.,")) > 4}
            hit = next((f for f in kept if words & {w.strip("?.,").lower() for w in f.split()}), None)
            return hit if hit else None

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw or b"{}")
            except ValueError:
                return self._json(400, {"error": {"message": "bad json"}})
            p = self.path.split("?")[0]
            if p == "/tokenize":                         # TensorFold's: a rough count, and the served window
                count = len(prompt_of(body)) // 4
                return self._json(200, {"count": count, "tokens": [], "max_model_len": WINDOW})
            if p == "/v1/slide/learn":
                if not self._authorized():
                    return self._json(401, {"error": {"message": "invalid API key"}})
                return self._learn(body)
            if p not in ("/v1/chat/completions", "/v1/completions"):
                return self._json(404, {"error": {"message": f"unknown path {self.path}"}})
            if not self._authorized():
                return self._json(401, {"error": {"message": "invalid API key"}})
            if body.get("prompt_logprobs") is not None:
                return self._prompt_logprobs(p, body)
            special = self._living_reply(body)
            if special is not None:
                return self._json(200, {"id": "chatcmpl-mock-living", "object": "chat.completion", "model": MODEL,
                                        "choices": [{"index": 0, "finish_reason": "stop",
                                                     "message": {"role": "assistant", "content": special}}],
                                        "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}})
            thinks = thinking_of(body)
            if thinks is None:
                return self._json(400, {"error": {"message": "the chat template refuses these chat_template_kwargs"}})
            words = REPLY.split(" ")
            n = min(len(words), int(body.get("max_tokens") or 4096))
            drafts = body.get("draft") is not False      # like TensorFold: the same tokens, slower without drafts
            tps = rate_tps if drafts else rate_tps * 0.7
            rid = f"chatcmpl-mock-{int(time.time() * 1000)}"
            text = prompt_of(body)
            prompt_tokens = max(1, len(text) // 4)
            if prompt_tokens > WINDOW:
                return self._json(400, {"error": {"message": f"the prompt has {prompt_tokens} tokens; the window is {WINDOW}"}})
            with State.lock:
                cached = prompt_tokens - 1 if text[:64] in State.seen else 0
                State.seen.add(text[:64])
            prefill = (prompt_tokens - cached) / PREFILL_TPS * (1 if rate_tps < 100 else 0.01)
            with State.lock:
                State.live += 1
                State.requests += 1
            try:
                if body.get("stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    time.sleep(prefill)                          # the first token, after reading the prompt
                    if thinks:
                        chunk = {"id": rid, "object": "chat.completion.chunk", "model": MODEL,
                                 "choices": [{"index": 0, "delta": {"reasoning_content": "Let me think."},
                                              "finish_reason": None}]}
                        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    for i in range(n):
                        delta = {"content": ("" if i == 0 else " ") + words[i]}
                        chunk = {"id": rid, "object": "chat.completion.chunk", "model": MODEL,
                                 "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
                        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                        self.wfile.flush()
                        time.sleep(1.0 / tps)
                    final = {"id": rid, "object": "chat.completion.chunk", "model": MODEL,
                             "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                             "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": n,
                                       "total_tokens": prompt_tokens + n,
                                       "prompt_tokens_details": {"cached_tokens": cached}},
                             # the native engine's runtime block, as TensorFold 1.0 sends it on the last event
                             **({} if State.plain else {"tensorfold": {"engine": "mock", "tokens_per_second": float(tps),
                                            "time_to_first_token": prefill, "prefill_seconds": prefill, "sampling": "greedy",
                                            "drafts": drafts, "token_sha": hashlib.sha256(" ".join(words[:n]).encode())
                                            .hexdigest()[:12]},
                                            # and its draft counters, beside the block (TensorFold 1.0.7)
                                            "speculative": {"rounds": max(1, n // 3), "drafted": n if drafts else 0,
                                                            "accepted": n * 3 // 4 if drafts else 0,
                                                            "acceptance_rate": 0.75 if drafts else 0}})}
                    self.wfile.write(f"data: {json.dumps(final)}\n\ndata: [DONE]\n\n".encode())
                    self.wfile.flush()
                else:
                    time.sleep(prefill + n / rate_tps)
                    # TensorFold 1.0.5's log-probabilities: each word likely at 0.8, the rest spread over others;
                    # the same with drafts or without, as exact decoding gives
                    top_n = min(20, int(body.get("top_logprobs") or 0))

                    def entry(i: int) -> dict:
                        tok = ("" if i == 0 else " ") + words[i]
                        tops = [{"token": tok, "logprob": math.log(0.8), "bytes": list(tok.encode())}]
                        tops += [{"token": f" alt{k}", "logprob": math.log(0.2 / 2 ** k), "bytes": list(f" alt{k}".encode())}
                                 for k in range(1, top_n)]
                        return {"token": tok, "logprob": math.log(0.8), "bytes": list(tok.encode()), "top_logprobs": tops[:top_n]}

                    logprobs = {"content": [entry(i) for i in range(n)]} if body.get("logprobs") else None
                    self._json(200, {"id": rid, "object": "chat.completion", "model": MODEL,
                                     "choices": [{"index": 0, "finish_reason": "stop", "logprobs": logprobs,
                                                  "message": {"role": "assistant", "content": " ".join(words[:n])}}],
                                     "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": n,
                                               "total_tokens": prompt_tokens + n},
                                     # the runtime block on a whole reply too, so a reference asked for drafts off
                                     # can be seen to be off
                                     **({} if State.plain else {"tensorfold": {"engine": "mock", "drafts": drafts},
                                                                "speculative": {"drafted": n if drafts else 0}})})
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with State.lock:
                    State.live -= 1
                    State.tokens += n
                    State.decode_s += n / rate_tps

    return H


def serve(host: str = "127.0.0.1", port: int = 8080, rate_tps: float = 27.0,
          api_key: str | None = None) -> ThreadingHTTPServer:
    """Listen on ``port``, or on a free port when that one is taken; the caller reads ``server.server_address``."""

    try:
        server = ThreadingHTTPServer((host, port), _handler(rate_tps, api_key))
    except OSError:
        server = ThreadingHTTPServer((host, 0), _handler(rate_tps, api_key))
    threading.Thread(target=server.serve_forever, name="mock-engine", daemon=True).start()
    return server


if __name__ == "__main__":
    import sys

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    serve(port=port)
    print(f"mock engine on http://127.0.0.1:{port}")
    threading.Event().wait()

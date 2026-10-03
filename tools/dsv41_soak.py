#!/usr/bin/env python3
# Adapted from jayleaton/deepseek-v41-tensorfold-spark bench/soak.py, Apache License 2.0, Copyright 2026 Jay Leaton
# (https://x.com/jayleaton). Modified for tensorfold dsv41-cuda: the API key from TF_API_KEY, our defaults and /health
# fields, a token_sha gate over repeated greedy requests (return_token_ids), streamed tool-call fragments checked to
# concatenate to JSON with no DSML markup in the content, --kinds, a tool kind under tool_choice auto by default.
"""Soak test: mixed traffic against an OpenAI-compatible tensorfold server for --minutes (30), standard library only.

Four worker threads; a controller moves the number of active ones between 1 and 4 every 45-120 s. Each request draws a
kind: code / prose (T = 0 or 0.7), structured (response_format json_schema), tool (tools, --tool-choice: auto by
default, "required" needs TF_DSV41_TOOL_GRAMMAR on the server), thinking (effort low), long (an 8K-48K-token document
and a question), nonstream. Streamed requests are cancelled at a random chunk ~15% of the time (the client closes the
socket); ~5% are "disconnects" (non-streamed, a 2-5 s client timeout mid-reply). Both are intentional, not errors.

Checked: errors (HTTP >= 400, exceptions, empty replies of normal requests) = 0; content checks counted separately
(structured replies parse, tool calls present, streamed call fragments concatenate to JSON, no ｜DSML｜ markup in the
content); every repeated greedy (T = 0) body, streamed or not, gives one ``tensorfold.token_sha`` (batch invariance:
the same reply whatever else runs); latency (first token, total) per kind; /health every 10 s (requests_running,
fatal, stalled); at the end requests_running back to 0 within --drain-s (no slot leak) and 17*23 = 391 (a sanity
answer).

  TF_API_KEY=... python3 tools/dsv41_soak.py --base http://127.0.0.1:8888 --minutes 30 --out soak.json
The key is read from TF_API_KEY only (never a flag: a command line is visible to every user). Exit 0 on PASS (0
errors, drained, 391, no fatal, one token_sha per repeated greedy body).
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import random
import socket
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request

KINDS = [("code", 22), ("prose", 18), ("structured", 14), ("tool", 14), ("thinking", 10), ("long", 10),
         ("nonstream", 7), ("disconnect", 5)]
CODE = ["Write a Python function that merges two sorted lists, with a docstring and two doctests.",
        "Implement an LRU cache class in Python with get / put in O(1). Code only.",
        "Write a bash script that rotates log files older than 7 days into a tar.gz archive."]
PROSE = ["Write a short essay about the history of lighthouses.", "Describe a rainy morning in a harbor town.",
         "Explain to a ten-year-old why the sky is blue, in two paragraphs."]
SCHEMA = {"type": "object", "properties": {"title": {"type": "string"}, "year": {"type": "integer"},
                                           "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 4}},
          "required": ["title", "year", "tags"]}
TOOL = [{"type": "function", "function": {"name": "get_weather", "description": "Weather for a city", "parameters": {
    "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
WORDS = "amber basin cobalt drift ember fable grove harbor island jetty kelp lumen marsh north orbit pine quay ridge".split()
MARKUP = "｜DSML｜"


def headers() -> dict[str, str]:
    key = os.environ.get("TF_API_KEY", "")
    return {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}


def body_for(kind: str, model: str, rng: random.Random, tool_choice: str = "auto") -> tuple[dict, bool]:
    """(request body, streamed)."""

    off = {"chat_template_kwargs": {"enable_thinking": False}}
    t = rng.choice([0.0, 0.7])
    if kind == "code":
        b = dict(off, messages=[{"role": "user", "content": rng.choice(CODE)}], max_tokens=384, temperature=t)
    elif kind in ("prose", "nonstream", "disconnect"):
        b = dict(off, messages=[{"role": "user", "content": rng.choice(PROSE)}], max_tokens=384, temperature=t)
    elif kind == "structured":
        b = dict(off, messages=[{"role": "user", "content": "Describe a classic film as JSON."}], max_tokens=256,
                 temperature=0, response_format={"type": "json_schema", "json_schema": {"name": "film", "schema": SCHEMA}})
    elif kind == "tool":
        b = dict(off, messages=[{"role": "user", "content": f"Weather in {rng.choice(['Oslo', 'Lima', 'Kyiv'])}? "
                                                            "Use the tool."}],
                 tools=TOOL, tool_choice=tool_choice, max_tokens=256, temperature=0)
    elif kind == "thinking":
        b = {"messages": [{"role": "user", "content": "How many weekdays are there in March 2027? Think it through."}],
             "reasoning_effort": "low", "max_tokens": 2048, "temperature": t}
    else:                                           # long: an 8K-48K-token document, one question
        n = rng.choice([8, 16, 32, 48]) * 1024
        doc = " ".join(f"Entry {i}: the {rng.choice(WORDS)} {rng.choice(WORDS)} met the {rng.choice(WORDS)}."
                       for i in range(n // 11))
        b = dict(off, messages=[{"role": "user", "content": doc + "\n\nHow many entries are there? One number."}],
                 max_tokens=32, temperature=0)
    b["model"] = model
    b["return_token_ids"] = True
    return b, kind not in ("nonstream", "disconnect")


def same_key(body: dict) -> str | None:
    """Greedy bodies that must give one reply, streamed or not (the long kind's random documents rarely repeat)."""

    if body.get("temperature") != 0:
        return None
    return json.dumps({k: v for k, v in body.items() if k != "stream"}, sort_keys=True)


class Soak:
    def __init__(self, a) -> None:
        self.a, self.lock, self.stop = a, threading.Lock(), threading.Event()
        self.active = 1
        self.rec: list[dict] = []
        self.health: list[dict] = []
        self.shas: dict[str, set[str]] = {}
        names = [k for k, _ in KINDS]
        wanted = [k.strip() for k in a.kinds.split(",") if k.strip()] if a.kinds else names
        unknown = sorted(set(wanted) - set(names))
        if unknown:
            raise SystemExit(f"unknown --kinds {unknown}: choose from {names}")
        self.kinds = [(k, w) for k, w in KINDS if k in wanted]

    def one(self, kind: str, rng: random.Random) -> dict:
        body, streamed = body_for(kind, self.a.model, rng, self.a.tool_choice)
        r = {"kind": kind, "t": round(time.time() - self.t0, 1), "error": None, "check": None, "cancel": False}
        cancel_at = rng.randint(1, 20) if streamed and rng.random() < self.a.cancel_p else None
        timeout = rng.uniform(2, 5) if kind == "disconnect" else self.a.timeout
        if streamed:
            body["stream"] = True
        req = urllib.request.Request(self.a.base + "/v1/chat/completions", json.dumps(body).encode(), headers())
        t0, text, content, args, chunks, sha = time.time(), "", "", {}, 0, None
        calls = 0
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if not streamed:
                    payload = json.loads(resp.read())
                    m = payload["choices"][0]["message"]
                    content = m.get("content") or ""
                    text = content + (m.get("reasoning_content") or "")
                    calls = len(m.get("tool_calls") or [])
                    args = {i: c["function"]["arguments"] for i, c in enumerate(m.get("tool_calls") or [])}
                    sha = (payload.get("tensorfold") or {}).get("token_sha")
                else:
                    for raw in resp:
                        line = raw.decode().strip()
                        if not line.startswith("data:") or line == "data: [DONE]":
                            continue
                        c = json.loads(line[5:])
                        sha = (c.get("tensorfold") or {}).get("token_sha") or sha
                        for ch in c.get("choices") or []:
                            d = ch.get("delta") or {}
                            if r.get("ttft_s") is None and (d.get("content") or d.get("reasoning_content")
                                                             or d.get("reasoning") or d.get("tool_calls")):
                                r["ttft_s"] = round(time.time() - t0, 3)
                            content += d.get("content") or ""
                            text += (d.get("content") or "") + (d.get("reasoning_content") or d.get("reasoning") or "")
                            for tc in d.get("tool_calls") or []:
                                if tc.get("id"):
                                    calls += 1
                                at = tc.get("index", 0)
                                args[at] = args.get(at, "") + ((tc.get("function") or {}).get("arguments") or "")
                        chunks += 1
                        if cancel_at is not None and chunks >= cancel_at:
                            r["cancel"] = True
                            break                   # leaving the with-block closes the socket mid-stream
        except urllib.error.HTTPError as exc:
            r["error"] = f"HTTP {exc.code}: {exc.read()[:200]!r}"
        except (socket.timeout, TimeoutError) as exc:
            if kind == "disconnect":
                r["cancel"] = True
            else:
                r["error"] = f"timeout: {exc}"
        except (OSError, http.client.HTTPException, ValueError) as exc:
            r["error"] = f"{type(exc).__name__}: {exc}"[:200]
        r["total_s"] = round(time.time() - t0, 3)
        if r["error"] is None and not r["cancel"]:
            r["check"] = self.check(kind, streamed, text, content, calls, args)
            if r["check"] == "empty reply":
                r["error"], r["check"] = r["check"], None
            key = same_key(body)
            if key is not None and sha:
                with self.lock:
                    self.shas.setdefault(key, set()).add(sha)
        return r

    @staticmethod
    def check(kind: str, streamed: bool, text: str, content: str, calls: int, args: dict) -> str | None:
        if MARKUP in content:
            return f"DSML markup in the content: {content[:80]!r}"
        for at, raw in args.items():
            try:
                json.loads(raw)
            except ValueError:
                return f"call {at}'s arguments are not JSON: {raw[:80]!r}"
        if kind == "structured":
            try:
                json.loads(content)
            except ValueError:
                return f"structured reply not JSON: {content[:80]!r}"
        elif kind == "tool":
            return None if calls else "no tool call"
        elif not text.strip():
            return "empty reply"
        return None

    def worker(self, i: int) -> None:
        rng = random.Random(self.a.seed * 100 + i)
        names, weights = zip(*self.kinds)
        while not self.stop.is_set():
            if i >= self.active:
                time.sleep(1)
                continue
            r = self.one(rng.choices(names, weights)[0], rng)
            with self.lock:
                self.rec.append(r)
            if r["error"]:
                print(f"[soak] ERROR {r['kind']} at {r['t']} s: {r['error']}", flush=True)

    def control(self) -> None:
        rng = random.Random(self.a.seed)
        while not self.stop.wait(rng.uniform(45, 120)):
            self.active = rng.randint(1, 4)
            print(f"[soak] {round(time.time() - self.t0)} s: {self.active} concurrent; {len(self.rec)} done", flush=True)

    def poll(self) -> None:
        while not self.stop.wait(10):
            self.health.append(dict(health(self.a.base), t=round(time.time() - self.t0)))

    def run(self) -> dict:
        self.t0 = time.time()
        th = [threading.Thread(target=self.worker, args=(i,), daemon=True) for i in range(4)]
        th += [threading.Thread(target=self.control, daemon=True), threading.Thread(target=self.poll, daemon=True)]
        for t in th:
            t.start()
        time.sleep(self.a.minutes * 60)
        self.stop.set()
        for t in th[:4]:
            t.join(self.a.timeout + 30)
        drained, t1 = False, time.time()
        while time.time() - t1 < self.a.drain_s:
            if health(self.a.base).get("running") == 0:
                drained = True
                break
            time.sleep(2)
        return self.report(drained, round(time.time() - t1, 1))

    def report(self, drained: bool, drain_s: float) -> dict:
        rec = self.rec
        by: dict[str, dict] = {}
        for k, _ in self.kinds:
            xs = [r for r in rec if r["kind"] == k]
            ok = [r for r in xs if not r["error"] and not r["cancel"]]
            tt = sorted(r["ttft_s"] for r in ok if r.get("ttft_s") is not None)
            tot = sorted(r["total_s"] for r in ok)
            pct = (lambda v, p: v[min(len(v) - 1, int(p * len(v)))] if v else None)    # noqa: E731
            by[k] = {"n": len(xs), "ok": len(ok), "cancelled": sum(r["cancel"] for r in xs),
                     "errors": sum(bool(r["error"]) for r in xs), "check_fails": sum(bool(r["check"]) for r in xs),
                     "ttft_p50": pct(tt, 0.5), "ttft_p95": pct(tt, 0.95),
                     "total_p50": statistics.median(tot) if tot else None, "total_p95": pct(tot, 0.95)}
        ans = ""
        try:
            req = urllib.request.Request(self.a.base + "/v1/chat/completions", json.dumps({
                "model": self.a.model, "messages": [{"role": "user", "content": "What is 17*23? Answer with the number only."}],
                "max_tokens": 64, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}).encode(),
                headers())
            with urllib.request.urlopen(req, timeout=300) as r:
                ans = json.loads(r.read())["choices"][0]["message"].get("content") or ""
        except Exception as exc:                    # noqa: BLE001
            ans = f"ERROR {exc}"
        errors = sum(bool(r["error"]) for r in rec)
        fatal = any(h.get("fatal") for h in self.health)
        split = {k: sorted(v) for k, v in self.shas.items() if len(v) > 1}
        repeated = sum(1 for v in self.shas.values() if v)
        out = {"minutes": self.a.minutes, "requests": len(rec), "errors": errors,
               "check_fails": sum(bool(r["check"]) for r in rec), "cancelled": sum(r["cancel"] for r in rec),
               "drained": drained, "drain_s": drain_s, "answer_17x23": ans[:40], "fatal_seen": fatal,
               "stalled_seen": any(h.get("stalled") for h in self.health),
               "running_max": max((h.get("running") or 0 for h in self.health), default=None),
               "greedy_bodies": repeated, "sha_splits": len(split),
               "by_kind": by, "health_end": health(self.a.base),
               "error_samples": [r for r in rec if r["error"]][:20], "check_samples": [r for r in rec if r["check"]][:10],
               "sha_split_samples": [{"body": k[:200], "shas": v} for k, v in list(split.items())[:5]]}
        out["pass"] = errors == 0 and drained and "391" in ans and not fatal and not split
        return out


def health(base: str) -> dict:
    try:
        with urllib.request.urlopen(base + "/health", timeout=10) as r:
            h = json.loads(r.read())
    except urllib.error.HTTPError as exc:          # TF_HEALTH=strict: 503 still carries the body
        try:
            h = json.loads(exc.read())
        except ValueError:
            return {"error": f"HTTP {exc.code}"}
    except Exception as exc:                        # noqa: BLE001
        return {"error": str(exc)[:120]}
    return {"running": h.get("requests_running"), "fatal": h.get("fatal"), "stalled": h.get("stalled"),
            "call_age_s": h.get("call_age_s")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:8888")
    ap.add_argument("--model", default="DeepSeek-v4.1-Flash-EXL3")
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--kinds", default="", help="comma list of request kinds (default: all)")
    ap.add_argument("--tool-choice", default="auto", help='the tool kind\'s tool_choice ("required" needs '
                                                         "TF_DSV41_TOOL_GRAMMAR on the server)")
    ap.add_argument("--cancel-p", type=float, default=0.15)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--drain-s", type=float, default=120)
    ap.add_argument("--seed", type=int, default=6)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    a.base = a.base.rstrip("/")
    out = Soak(a).run()
    print(json.dumps({k: v for k, v in out.items() if k not in ("by_kind", "error_samples", "check_samples")}), flush=True)
    for k, v in out["by_kind"].items():
        print(f"[soak] {k}: {v}", flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())

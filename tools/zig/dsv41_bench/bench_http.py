#!/usr/bin/env python3
"""Decode speed over HTTP, one client for any server: code/prose cells at T0 and T0.7, 1 and 4 streams, long, mixwin."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
import urllib.request

CODE = """Write a Python class `LRUCache` with `get(key)` and `put(key, value)` in O(1) using a dict and a doubly linked
list, with docstrings and type hints, then three unit tests with pytest."""
PROSE = """Write a 400-word essay on why lighthouses were built where they were, how their keepers lived, and what
replaced them. Plain prose, no lists or headings."""
CODE_MORE = (
    """Write a Python class `Trie` with `insert(word)`, `search(word)` and `starts_with(prefix)`, with docstrings and
type hints, then three unit tests with pytest.""",
    """Write a Python class `TokenBucket` rate limiter with `allow(now: float) -> bool`, refill rate and burst size,
with docstrings and type hints, then three unit tests with pytest.""",
    """Write a Python function `merge_intervals(intervals)` that merges overlapping [start, end] pairs in
O(n log n), with a docstring and type hints, then three unit tests with pytest.""")
PROSE_MORE = (
    """Write a 400-word essay on why canals were dug where they were, how the people who worked them lived, and what
replaced them. Plain prose, no lists or headings.""",
    """Write a 400-word essay on why early observatories were built where they were, how their astronomers lived, and
what replaced them. Plain prose, no lists or headings.""",
    """Write a 400-word essay on why windmills were built where they were, how their millers lived, and what replaced
them. Plain prose, no lists or headings.""")
PROMPTS = {"code": (CODE, *CODE_MORE), "prose": (PROSE, *PROSE_MORE)}
SALT = ""  # --salt


NO_DRAFT = False     # --no-draft: every request with "draft": false (both servers: the reply without DSpark)
TIMES = False        # --times: each stream's content-chunk arrival times (ms from its first token) in the cell


def one(url: str, model: str, msg: str, temp: float, max_tokens: int, out: dict) -> None:
    try:
        stream_one(url, model, msg, temp, max_tokens, out)
    except Exception as e:      # a refused / dropped connection: the stream failed (the cell records it, never averages it)
        out["error"] = f"{type(e).__name__}: {e}"


def stream_one(url: str, model: str, msg: str, temp: float, max_tokens: int, out: dict) -> None:
    body = {"model": model, "messages": [{"role": "user", "content": msg}], "max_tokens": max_tokens,
            "temperature": temp, "stream": True, "stream_options": {"include_usage": True},
            "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}}
    if temp > 0:
        body.update(top_p=0.95, seed=1234)
    if NO_DRAFT:
        body["draft"] = False
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    times, usage, t0, text, tf = [], None, time.perf_counter(), [], None
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            if d.get("tensorfold"):
                tf = d["tensorfold"]
            if d.get("error"):
                out["error"] = str(d["error"])[:300]
            for c in d.get("choices") or []:
                delta = c.get("delta") or {}
                if delta.get("content") or delta.get("reasoning_content"):
                    times.append(time.perf_counter())
                    out["_first_seen"] = True
                    text.append(delta.get("content") or delta.get("reasoning_content") or "")
    n = (usage or {}).get("completion_tokens") or len(times)
    out.update(t0=t0, first=times[0] if times else None, last=times[-1] if times else None, tokens=n,
               prompt_tokens=(usage or {}).get("prompt_tokens"),
               sha=hashlib.sha256("".join(text).encode()).hexdigest()[:16],
               decode_tok_s=round((n - 1) / (times[-1] - times[0]), 2) if len(times) > 1 and times[-1] > times[0] else None,
               ttft_s=round(times[0] - t0, 3) if times else None, tpr=tokens_per_round(tf), times=times)
    if TIMES and times:
        out["times_ms"] = [round((t - times[0]) * 1e3, 2) for t in times]


def tokens_per_round(tf: dict | None) -> float | None:
    """Tokens a decode round of one reply (1 = no draft kept), from either server's stats block."""
    tf = tf or {}
    dr = tf.get("drafting") or {}
    if dr.get("tokens_per_round") is not None:
        return dr["tokens_per_round"]
    if not dr and tf.get("rounds") and "accepted" in tf:     # the reference server's top-level "rounds" is the server's total
        return round((tf["rounds"] + tf["accepted"]) / tf["rounds"], 3)
    return None


def mean(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def cell(url: str, model: str, work: str, temp: float, streams: int, max_tokens: int) -> dict:
    res = [dict() for _ in range(streams)]
    th = [threading.Thread(target=one, args=(url, model, SALT + PROMPTS[work][i % 4], temp, max_tokens, res[i]))
          for i in range(streams)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    errors = [r.get("error") or "no tokens" for r in res if r.get("error") or not r.get("first")]
    if errors:
        print(f"cell {work} T{temp} x{streams}: {len(errors)} stream(s) failed: {errors[:2]}", file=sys.stderr)
    if len(errors) == streams:
        raise RuntimeError(f"every stream failed: {errors[0]}")
    rates = [r["decode_tok_s"] for r in res if r.get("decode_tok_s")]
    lo = max(r["first"] for r in res if r.get("first"))
    hi = min(r["last"] for r in res if r.get("last"))
    agg = None
    if streams > 1 and hi > lo:     # tokens inside the all-live span, apportioned at each stream's own rate
        agg = round(sum(r["decode_tok_s"] * (hi - lo) for r in res if r.get("decode_tok_s")) / (hi - lo), 2)
    return {"work": work, "temp": temp, "streams": streams, "per_stream": rates,
            "mean_tok_s": round(sum(rates) / len(rates), 2) if rates else None, "aggregate_tok_s": agg,
            "ttft_s": [r.get("ttft_s") for r in res], "tokens": [r.get("tokens") for r in res],
            "sha": [r.get("sha") for r in res], "tpr": [r.get("tpr") for r in res],
            "mean_tpr": mean([r.get("tpr") for r in res]), "failed_streams": len(errors), "errors": errors[:4],
            **({"times_ms": [r.get("times_ms") for r in res]} if TIMES else {})}


def document(n: int) -> str:
    """About n tokens of distinct text (numbered prose paragraphs, ~0.75 words a token) and a question."""
    paras, words, i = [], 0, 0
    base = [p.replace("Write a 400-word essay on ", "").replace("\n", " ") for p in PROMPTS["prose"] + PROMPTS["code"]]
    while words < 0.75 * n:
        t = f"Section {i}: note {i * 7919 % 100003} on {base[i % len(base)]}"
        paras.append(t)
        words += len(t.split())
        i += 1
    return "\n\n".join(paras) + f"\n\nWhich note number does section {i // 2} carry? Then summarize the document in 100 words."


def long_cell(url: str, model: str, n: int, max_tokens: int) -> dict:
    r: dict = {}
    one(url, model, document(n), 0.0, max_tokens, r)
    pt = r.get("prompt_tokens")
    return {"work": f"long-{n}", "temp": 0.0, "streams": 1, "prompt_tokens": pt, "ttft_s": r.get("ttft_s"),
            "prompt_tok_s": round(pt / r["ttft_s"], 1) if pt and r.get("ttft_s") else None,
            "mean_tok_s": r.get("decode_tok_s"), "aggregate_tok_s": None, "per_stream": [r.get("decode_tok_s")],
            "tokens": [r.get("tokens")], "sha": [r.get("sha")], "tpr": [r.get("tpr")], "mean_tpr": r.get("tpr"),
            "failed_streams": int(bool(r.get("error") or not r.get("first"))), "errors": [r["error"]] if r.get("error") else []}


def mixwin_cell(url: str, model: str, n: int) -> dict:
    """A long prompt-cached stream decoding beside 3 code streams started at its first token (mixed windows)."""
    doc = document(n)
    warm: dict = {}
    one(url, model, doc, 0.0, 8, warm)                                      # fills the prompt cache; not timed
    res = [dict() for _ in range(4)]
    th = threading.Thread(target=one, args=(url, model, doc, 0.0, 512, res[0]))
    th.start()
    t_wait = time.perf_counter()
    while not res[0].get("first") and th.is_alive() and time.perf_counter() - t_wait < 900:
        # stream_one fills `times` only at the end: watch the request's first chunk through a side flag instead
        if res[0].get("_first_seen"):
            break
        time.sleep(0.01)
    shorts = [threading.Thread(target=one, args=(url, model, SALT + PROMPTS["code"][i], 0.0, 384, res[i + 1]))
              for i in range(3)]
    for t in shorts:
        t.start()
    for t in [th, *shorts]:
        t.join()
    errors = [r.get("error") or "no tokens" for r in res if r.get("error") or not r.get("first")]
    rates = [r.get("decode_tok_s") for r in res]
    lo = max((r["first"] for r in res if r.get("first")), default=0.0)
    hi = min((r["last"] for r in res if r.get("last")), default=0.0)
    agg = live = live_chunks = None
    if hi > lo and not errors:
        agg = round(sum(r["decode_tok_s"] * (hi - lo) for r in res if r.get("decode_tok_s")) / (hi - lo), 2)
        # the long stream's tokens in the all-live span: chunks inside x tokens a chunk (a chunk is a round)
        times0 = res[0].get("times") or []
        inside = [t for t in times0 if lo <= t <= hi]
        per_chunk = (res[0]["tokens"] - 1) / (len(times0) - 1) if len(times0) > 1 and res[0].get("tokens") else 1.0
        ok = len(inside) > 1 and inside[-1] > inside[0]
        live = round((len(inside) - 1) * per_chunk / (inside[-1] - inside[0]), 2) if ok else None
        live_chunks = round((len(inside) - 1) / (inside[-1] - inside[0]), 2) if ok else None
    return {"work": f"mixwin-{n}", "temp": 0.0, "streams": 4, "per_stream": rates,
            "mean_tok_s": round(sum(x for x in rates if x) / max(1, len([x for x in rates if x])), 2),
            "aggregate_tok_s": agg, "long_live_tok_s": live, "long_live_chunks_s": live_chunks, "live_span_s": round(hi - lo, 3) if hi > lo else None,
            "prompt_tokens": res[0].get("prompt_tokens"), "ttft_s": [r.get("ttft_s") for r in res],
            "tokens": [r.get("tokens") for r in res], "sha": [r.get("sha") for r in res], "tpr": [r.get("tpr") for r in res],
            "mean_tpr": mean([r.get("tpr") for r in res]), "failed_streams": len(errors), "errors": errors[:4], "rep": 0}


def compare(a_path: str, b_path: str) -> None:
    try:
        a, b = json.load(open(a_path)), json.load(open(b_path))
    except (OSError, ValueError) as e:     # a part that did not run (zig-bench.sh PARTS): no table, not a crash
        print(f"no comparison: {e}")
        return

    def key(c):
        return (c["work"], c["temp"], c["streams"])

    best = lambda cs: {k: max(v) for k, v in _group(cs, key).items()}       # noqa: E731 (best of the reps)
    pa, pb = best(a["cells"]), best(b["cells"])
    print(f"| workload | T | streams | {a['label']} tok/s | {b['label']} tok/s | {b['label']} vs {a['label']} |")
    print("|---|---|---|---:|---:|---:|")
    for k in sorted(pa):
        if k in pb and pa[k] and pb[k]:
            print(f"| {k[0]} | {k[1]} | {k[2]} | {pa[k]:.1f} | {pb[k]:.1f} | {100 * (pb[k] / pa[k] - 1):+.1f} % |")
    # greedy (T0) replies: the same text on both sides, prompt by prompt (the same tokens)
    eq = tot = 0
    for ca in a["cells"]:
        for cb in b["cells"]:
            if key(ca) == key(cb) and ca["temp"] == 0 and ca.get("sha") and cb.get("sha") and ca.get("rep") == cb.get("rep"):
                for x, y in zip(ca["sha"], cb["sha"]):
                    tot += 1
                    eq += x == y
    if tot:
        print(f"\nT0 replies equal ({b['label']} vs {a['label']}): {eq}/{tot}")
    la = {c["work"]: c for c in a["cells"] if c["work"].startswith("long-")}
    lb = {c["work"]: c for c in b["cells"] if c["work"].startswith("long-")}
    if la and lb:
        print(f"\n| long context | prompt tokens | {a['label']} prompt tok/s | {b['label']} prompt tok/s | "
              f"{a['label']} TTFT s | {b['label']} TTFT s |")
        print("|---|---:|---:|---:|---:|---:|")
        for w in sorted(la, key=lambda x: int(x[5:])):
            if w in lb:
                print(f"| {w} | {la[w]['prompt_tokens']} | {la[w]['prompt_tok_s']} | {lb[w]['prompt_tok_s']} | "
                      f"{la[w]['ttft_s']} | {lb[w]['ttft_s']} |")
    # drafting's acceptance: mean tokens a round a cell (all reps), where both sides report it
    ta, tb = _tprs(a["cells"], key), _tprs(b["cells"], key)
    rows = [k for k in sorted(ta) if k in tb and ta[k] and tb[k]]
    if rows:
        print(f"\n| workload | T | streams | {a['label']} tokens/round | {b['label']} tokens/round | {b['label']} vs {a['label']} |")
        print("|---|---|---|---:|---:|---:|")
        for k in rows:
            print(f"| {k[0]} | {k[1]} | {k[2]} | {ta[k]:.3f} | {tb[k]:.3f} | {100 * (tb[k] / ta[k] - 1):+.1f} % |")


def _tprs(cs, key):
    out = {}
    for c in cs:
        out.setdefault(key(c), []).extend(x for x in c.get("tpr") or [] if x is not None)
    return {k: sum(v) / len(v) if v else None for k, v in out.items()}


def _group(cs, key):
    out = {}
    for c in cs:
        v = c["mean_tok_s"] if c["streams"] == 1 else (c["aggregate_tok_s"] or c["mean_tok_s"])
        out.setdefault(key(c), []).append(v or 0.0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="dsv41")
    ap.add_argument("--label", default="")
    ap.add_argument("--streams", default="1,4")
    ap.add_argument("--temps", default="0,0.7")
    ap.add_argument("--work", default="code,prose")
    ap.add_argument("--salt", default="", help="text put before every cell prompt: new prompts of the same shape (cold-start studies)")
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out", default="")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--long", default="", help="long-context lengths (tokens, comma separated): one stream each, T0")
    ap.add_argument("--mixwin", type=int, default=0, help="a long stream of this many tokens decoding beside 3 code streams (mixwin_cell)")
    ap.add_argument("--no-draft", action="store_true", help='every request with "draft": false (drafts-off replies)')
    ap.add_argument("--times", action="store_true", help="each stream's content-chunk arrival times in the cell (ms)")
    a = ap.parse_args()
    global NO_DRAFT, SALT, TIMES
    NO_DRAFT = a.no_draft
    TIMES = a.times
    SALT = a.salt + "\n\n" if a.salt else ""
    if a.compare:
        compare(*a.compare)
        return 0
    cell(a.url, a.model, "code", 0.0, 1, 32)                                 # warm-up, not recorded
    cells = []
    if a.mixwin:
        c = mixwin_cell(a.url, a.model, a.mixwin)
        c.pop("times", None)
        print(json.dumps(c), flush=True)
        if a.out:
            json.dump({"label": a.label or a.url, "cells": [c]}, open(a.out, "w"), indent=1)
        return 0 if not c["failed_streams"] else 1
    if a.long:
        for n in [int(x) for x in a.long.split(",")]:
            c = long_cell(a.url, a.model, n, 128)
            cells.append(c)
            print(json.dumps(c), flush=True)
        if a.out:
            json.dump({"label": a.label or a.url, "cells": cells}, open(a.out, "w"), indent=1)
        return 0
    for rep in range(a.reps):
        for work in a.work.split(","):
            for temp in [float(t) for t in a.temps.split(",")]:
                for s in [int(x) for x in a.streams.split(",")]:
                    c = cell(a.url, a.model, work, temp, s, a.max_tokens)
                    c["rep"] = rep
                    cells.append(c)
                    print(json.dumps(c), flush=True)
    if a.out:
        json.dump({"label": a.label or a.url, "cells": cells}, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

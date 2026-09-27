#!/usr/bin/env python3
"""Concurrency bench for an OpenAI-compatible server (``tensorfold serve --lanes N``).

Fires N simultaneous streaming chat requests (temperature 0, distinct prompts by default) and reports, per level:
wall time, aggregate tok/s, each stream's decode tok/s and time to first token, and whether every concurrent answer
is byte-identical to the same prompt answered alone (N = 1). Standard library only.

    tools/bench_concurrency.py --url http://127.0.0.1:8080 --model MODEL --levels 1,2,4,8 --max-tokens 160 \
        [--tokenizer MODEL_DIR] [--prompts same|distinct] [--out results.json] [--keep-text]

Tokens are counted from the stream's deltas unless ``--tokenizer`` names a directory with a ``tokenizer.json``
(the ``tokenizers`` package), which counts them exactly whatever the server puts in each SSE event.
"""
import argparse, json, statistics, sys, threading, time, urllib.request
TOK = None  # a tokenizers.Tokenizer when --tokenizer is given; else the stream's deltas are counted

PROMPTS = [
    "Write a Python function that parses ISO-8601 dates and returns a Unix timestamp. Include three tests.",
    "Explain, in plain English, how speculative decoding can be exact. Then give one concrete example.",
    "List the steps to safely resize a GPT partition on Linux, with the commands.",
    "Write a short bash script that tails a log, counts ERROR lines per minute, and prints a table.",
    "Describe the difference between logical and physical Postgres replication with one use case each.",
    "Write a SQL query for the top 5 customers by 90-day revenue with their first order date.",
    "Explain what a Metal kernel launch costs and why fusing kernels helps decode speed.",
    "Give a recipe for cornbread that serves eight, with times and temperatures.",
    "Summarize how Tailscale NAT traversal works in five bullets.",
    "Write a haiku sequence (three haiku) about a laptop that carries a family's memory.",
    "What is the Gumbel-max trick and why does it make sampling reproducible?",
    "Draft a polite two-paragraph email declining a meeting and proposing two alternatives.",
    "Explain KV cache prefix sharing across concurrent requests.",
    "Write a Python asyncio example that limits concurrency to 4 with a semaphore.",
    "Give three ways to detect a stuck Thunderbolt link on macOS from the command line.",
    "Describe how to verify two model outputs are byte-identical, including a shell one-liner.",
]

def one(url, model, prompt, max_tokens, out, idx, t_launch):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "stream": True}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); first = None; text = []; ntok = 0; err = None
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            for line in r:
                if not line.startswith(b"data:"): continue
                d = line[5:].strip()
                if d == b"[DONE]": break
                try: j = json.loads(d)
                except Exception: continue
                ch = j.get("choices") or []
                if not ch: continue
                delta = ch[0].get("delta", {})
                piece = delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning") or ""
                if piece:
                    if first is None: first = time.perf_counter()
                    text.append(piece); ntok += 1
                u = j.get("usage")
                if u and u.get("completion_tokens"): ntok = u["completion_tokens"]
    except Exception as e:
        err = repr(e)
    t1 = time.perf_counter()
    full = "".join(text)
    if TOK is not None and full:
        ntok = len(TOK.encode(full, add_special_tokens=False).ids)   # exact count, independent of how the server chunks its stream
    out[idx] = {"prompt_i": idx, "ttft": (first - t0) if first else None, "wall": t1 - t0, "tokens": ntok,
                "decode_tps": (ntok - 1) / (t1 - first) if first and ntok > 1 and t1 > first else None,
                "text": full, "err": err, "start_offset": t0 - t_launch}

def run_level(url, model, n, max_tokens, distinct):
    out = [None] * n; th = []; t_launch = time.perf_counter()
    for i in range(n):
        p = PROMPTS[i % len(PROMPTS)] if distinct else PROMPTS[0]
        t = threading.Thread(target=one, args=(url, model, p, max_tokens, out, i, t_launch)); t.start(); th.append(t)
    for t in th: t.join()
    wall = time.perf_counter() - t_launch
    ok = [o for o in out if o and not o["err"] and o["tokens"] > 0]
    tot = sum(o["tokens"] for o in ok)
    return {"n": n, "wall": wall, "ok": len(ok), "errors": [o["err"] for o in out if o and o["err"]],
            "tokens_total": tot, "agg_tps": tot / wall if wall else 0,
            "ttft_med": statistics.median([o["ttft"] for o in ok if o["ttft"]]) if ok else None,
            "ttft_max": max([o["ttft"] for o in ok if o["ttft"]]) if ok else None,
            "per_stream_tps_med": statistics.median([o["decode_tps"] for o in ok if o["decode_tps"]]) if ok else None,
            "per_stream_tps_min": min([o["decode_tps"] for o in ok if o["decode_tps"]]) if ok else None,
            "streams": out}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--url", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--levels", default="1,2,4,8"); ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--prompts", default="distinct", choices=["same", "distinct"]); ap.add_argument("--out")
    ap.add_argument("--label", default=""); ap.add_argument("--tokenizer", help="dir with tokenizer.json: count tokens exactly"); ap.add_argument("--keep-text", action="store_true")
    a = ap.parse_args(); levels = [int(x) for x in a.levels.split(",")]
    global TOK
    if a.tokenizer:
        from tokenizers import Tokenizer; TOK = Tokenizer.from_file(a.tokenizer.rstrip("/") + "/tokenizer.json")
    # warm-up + serial reference answers (temperature 0) for the identity check
    print(f"# {a.label} url={a.url} model={a.model} max_tokens={a.max_tokens} prompts={a.prompts} count={'tokenizer' if a.tokenizer else 'sse-deltas'}", flush=True)
    run_level(a.url, a.model, 1, 16, False)
    ref = {}
    for i in range(max(levels)):
        p = PROMPTS[i % len(PROMPTS)] if a.prompts == "distinct" else PROMPTS[0]
        if p in ref: continue
        o = [None]; one(a.url, a.model, p, a.max_tokens, o, 0, time.perf_counter()); ref[p] = o[0]["text"]
    results = []
    print(f"{'N':>3} {'wall s':>7} {'ok':>3} {'tok':>6} {'agg tok/s':>10} {'stream tok/s med/min':>21} {'TTFT med/max s':>15} {'identical':>9}", flush=True)
    for n in levels:
        r = run_level(a.url, a.model, n, a.max_tokens, a.prompts == "distinct")
        ident = 0
        for o in r["streams"]:
            if o and not o["err"]:
                p = PROMPTS[o["prompt_i"] % len(PROMPTS)] if a.prompts == "distinct" else PROMPTS[0]
                if o["text"] == ref.get(p): ident += 1
        r["identical_to_serial"] = f"{ident}/{r['ok']}"
        f = lambda v, w=6: (f"{v:{w}.2f}" if isinstance(v, (int, float)) and v is not None else " " * (w - 1) + "-")
        print(f"{n:>3} {f(r['wall'],7)} {r['ok']:>3} {r['tokens_total']:>6} {f(r['agg_tps'],10)} {f(r['per_stream_tps_med'],10)}/{f(r['per_stream_tps_min'],6)} {f(r['ttft_med'],7)}/{f(r['ttft_max'],6)} {r['identical_to_serial']:>9}", flush=True)
        if r["errors"]: print("   errors:", r["errors"][:3], flush=True)
        if not a.keep_text:
            for o in r["streams"]: o.pop("text", None)
        results.append(r)
    if a.out: json.dump({"label": a.label, "url": a.url, "model": a.model, "max_tokens": a.max_tokens, "levels": results}, open(a.out, "w"), indent=1)

if __name__ == "__main__": main()

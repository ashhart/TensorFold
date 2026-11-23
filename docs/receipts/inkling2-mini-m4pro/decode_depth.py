#!/usr/bin/env python3
"""Decode rate after a cold prompt: exact-token-count prompt, then NEW tokens streamed, one SSE chunk a token.

Prompts are the ones `cold_prefill.py` builds (same builder, same seeds: seed = N * 10 + rep + --rep0), so the arrays
match the receipt's cold-prompt SHA-256 table. Each chunk's arrival time gives an inter-token latency; the first
`skipped` intervals after the first token are dropped and the rest averaged. `--sizes 65344:192,65536:16` gives a size its
own NEW (a native server at `--context 65552` takes at most 16 reply tokens after a 65,536-token prompt).
`--tokens-only` writes the token arrays and prints their SHA-256 without a server.
"""
import argparse, hashlib, json, random, re, sys, time, urllib.error, urllib.request

WORDS = ("residual stream attention head rotary position cache window sliding global expert router token gate norm layer "
         "weight scale kernel matrix vector memory bandwidth latency throughput batch prompt prefill decode sample logit "
         "softmax sink bias loop visit pass message query key value fold shard tensor graph compile fuse tile group").split()


def build(tok, n, seed):
    rng = random.Random(seed)
    text = f"Request {seed}:"
    chunk = []
    while True:
        chunk.append(rng.choice(WORDS))
        if len(chunk) % 64 == 0:
            cand = text + " " + " ".join(chunk)
            if len(tok(cand, add_special_tokens=False)["input_ids"]) >= n:
                break
            text, chunk = cand, []
    # add words one at a time until the count reaches n, then trim by tokens
    while True:
        cand = text + " " + " ".join(chunk[:1])
        chunk = chunk[1:] or [rng.choice(WORDS)]
        ids = tok(cand, add_special_tokens=False)["input_ids"]
        text = cand
        if len(ids) >= n:
            break
    ids = ids[:n]
    return {"text": tok.decode(ids), "ids": ids}


def stream(base, model, prompt, new):
    body = json.dumps({"model": model, "prompt": prompt, "max_tokens": new, "temperature": 0, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(base + "/completions", data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    stamps, usage, pieces = [], None, []
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            d = line[5:].strip()
            if d == b"[DONE]":
                break
            j = json.loads(d)
            usage = j.get("usage") or usage
            ch = j.get("choices") or []
            if ch and ch[0].get("text"):
                stamps.append(time.perf_counter() - t0)
                pieces.append(ch[0]["text"])
    return stamps, usage, "".join(pieces)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base", nargs="?", help="server base URL ending in /v1; not needed with --tokens-only")
    ap.add_argument("--model", default="m")
    ap.add_argument("--ck", required=True, help="checkpoint directory (its tokenizer builds the prompts)")
    ap.add_argument("--sizes", default="2048,8192,32768,65536", help="N or N:NEW, comma separated")
    ap.add_argument("--tokens", type=int, default=192, help="NEW for a size without its own")
    ap.add_argument("--rep0", type=int, default=0, help="added to every seed, so a repeat gets a prompt no cache has seen")
    ap.add_argument("--label", default="run")
    ap.add_argument("--output", required=True)
    ap.add_argument("--tokens-only", action="store_true")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.ck)
    if not a.tokens_only:
        stream(a.base, a.model, "Hello", 4)  # warm
    rows = []
    for spec in a.sizes.split(","):
        n, _, new = spec.partition(":")
        n, new = int(n), int(new or a.tokens)
        p = build(tok, n, seed=n * 10 + a.rep0)
        if a.tokens_only:
            blob = json.dumps({"ids": p["ids"]}).encode()
            open(f"{a.output}.tokens-{n}-{a.rep0}.json", "wb").write(blob)
            print(f"{hashlib.sha256(blob).hexdigest()}  tokens-{n}-{a.rep0}.json", flush=True)
            continue
        try:
            ts, usage, text = stream(a.base, a.model, p["text"], new)
        except urllib.error.HTTPError as e:
            cap = re.search(r"at most (\d+) reply", e.read().decode())
            if e.code != 400 or not cap:
                raise
            new = int(cap.group(1))
            ts, usage, text = stream(a.base, a.model, p["text"], new)
        itl = [ts[i] - ts[i - 1] for i in range(1, len(ts))]
        skipped = min(8, max(1, len(itl) // 4))
        tail = sorted(itl[skipped:])
        mean = sum(tail) / len(tail)
        rows.append({"asked": n, "rep0": a.rep0, "prompt_tokens": (usage or {}).get("prompt_tokens"), "new": new,
                     "completion_tokens": (usage or {}).get("completion_tokens"), "chunks": len(ts), "skipped": skipped,
                     "first_token_s": round(ts[0], 3), "mean_ms": round(mean * 1e3, 2),
                     "median_ms": round(tail[len(tail) // 2] * 1e3, 2), "p90_ms": round(tail[int(len(tail) * .9)] * 1e3, 2),
                     "tok_s": round(1 / mean, 2), "text_sha": hashlib.sha256(text.encode()).hexdigest()[:12]})
        print(json.dumps(rows[-1]), flush=True)
    if not a.tokens_only:
        json.dump({"label": a.label, "rows": rows}, open(a.output, "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())

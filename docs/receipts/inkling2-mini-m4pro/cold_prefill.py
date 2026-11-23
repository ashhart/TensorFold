#!/usr/bin/env python3
"""Cold prefill client: exact-token-count prompts through /v1/completions, one token out, cache state reported.

Each prompt is built from the checkpoint's tokenizer (transformers; tokenizers only) so it has exactly N tokens, from a
seeded word list, with a unique leading word per request so no prefix cache can hit. `--tokens-only` writes the token
arrays (`PREFIX.tokens-N-REP.json`) and prints their SHA-256 without a server, which is how the receipt's arrays are
rebuilt: the same checkpoint tokenizer gives the same bytes.
"""
import argparse, hashlib, json, random, sys, time, urllib.request

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


def post(base, model, prompt, n_out):
    body = json.dumps({"model": model, "prompt": prompt, "max_tokens": n_out, "temperature": 0, "stream": False}).encode()
    req = urllib.request.Request(base + "/completions", data=body, headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=3600) as r:
        raw = json.loads(r.read())
    return time.perf_counter() - t, raw, body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base", nargs="?", help="server base URL ending in /v1; not needed with --tokens-only")
    ap.add_argument("--model", default="m")
    ap.add_argument("--ck", required=True, help="checkpoint directory (its tokenizer builds the prompts)")
    ap.add_argument("--sizes", default="2048,8192,32768,65536")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--label", default="run")
    ap.add_argument("--output", required=True)
    ap.add_argument("--tokens-only", action="store_true")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.ck)
    if not a.tokens_only:
        post(a.base, a.model, "Hello", 1)  # warm
    rows = []
    for n in [int(x) for x in a.sizes.split(",")]:
        for rep in range(a.reps):
            p = build(tok, n, seed=n * 10 + rep)
            path = f"{a.output}.tokens-{n}-{rep}.json"
            with open(path, "w") as f:
                json.dump({"ids": p["ids"]}, f)
            sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
            if a.tokens_only:
                print(f"{sha}  tokens-{n}-{rep}.json", flush=True)
                continue
            dt, raw, body = post(a.base, a.model, p["text"], 1)
            u = raw.get("usage", {})
            row = {"asked": n, "rep": rep, "prompt_tokens": u.get("prompt_tokens"),
                   "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens", u.get("cached_tokens")),
                   "first_token_s": round(dt, 3), "prefill_tps": round((u.get("prompt_tokens") or n) / dt, 1),
                   "body_bytes": len(body)}
            rows.append(row)
            print(json.dumps(row), flush=True)
    if not a.tokens_only:
        json.dump({"label": a.label, "rows": rows}, open(a.output, "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())

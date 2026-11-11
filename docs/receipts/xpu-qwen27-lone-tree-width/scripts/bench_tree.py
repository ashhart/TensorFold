#!/usr/bin/env python3
"""Served decode by prompt depth, one JSON line per request. usage: bench_tree.py URL LABEL OUTDIR [reps]

Per (depth, mode) cell: 1 warm-up request and `reps` measured ones. Modes: draft (greedy, the default), plain
("draft": false) and sampled (temperature 1.0, top_k 20, top_p 0.95, fixed seed). A request without a
tensorfold.token_sha256 is a failure. The prompts are the first lines of inputs/corp_code.txt (build_corp_code.sh).
Env: CELLS="0:code;32768:code;65536:code;131072:code" (depth:kind), MODES="draft,plain", CORP=inputs/corp_code.txt.
"""
import hashlib, json, os, sys, time, urllib.request

URL, LABEL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 3
CELLS = [(int(c.split(":")[0]), c.split(":")[1]) for c in
         os.environ.get("CELLS", "0:code;32768:code;65536:code;131072:code").split(";")]
H = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)
CORP = open(os.environ.get("CORP", f"{H}/../inputs/corp_code.txt"), errors="ignore").read()
ASK = "Review the code above in detail, then write one new, complete Python function in the same style."
SHORT = ("Write a complete Python function that parses a CSV file and prints the average of each numeric column, "
         "with comments.")
SEED = 7


def post(content, mode):
    body = {"model": "m", "messages": [{"role": "user", "content": content}], "max_tokens": 256, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    if mode == "plain":
        body["draft"] = False
    if mode == "sampled":
        body.update(temperature=1.0, top_k=20, top_p=0.95, seed=SEED)
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0, first, text, usage, tsha, spec = time.time(), None, [], None, None, None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: {"):
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                usage = d["usage"]
            tf = d.get("tensorfold") or (d.get("usage") or {}).get("tensorfold") or {}
            if d.get("speculative") is not None:
                spec = d["speculative"]
            elif (d.get("usage") or {}).get("speculative") is not None:
                spec = d["usage"]["speculative"]
            if tf.get("token_sha256"):
                tsha = tf["token_sha256"]
            c = (d.get("choices") or [{}])[0].get("delta", {}).get("content")
            if c:
                first = first or time.time()
                text.append(c)
    t1, s = time.time(), "".join(text)
    n = (usage or {}).get("completion_tokens", 0)
    ok = bool(text and usage and tsha) and (mode == "plain" or spec is not None)
    return {"ok": ok, "speculative": spec, "token_sha256": tsha, "text_sha256": hashlib.sha256(s.encode()).hexdigest(),
            "prompt_tokens": (usage or {}).get("prompt_tokens"), "completion_tokens": n,
            "ttft_s": (first or t1) - t0,
            "decode_tok_s": (n - 1) / (t1 - first) if first and t1 > first and n > 1 else None}


tokens_per_char = []


def prompt(depth):
    if depth == 0:
        return SHORT
    if not tokens_per_char:
        r = post(CORP[:20000], "plain")
        tokens_per_char.append(20000 / (r["prompt_tokens"] - 15))
    return "Here is the material:\n\n" + CORP[:int((depth - 300) * tokens_per_char[0])] + "\n\n" + ASK


failed = 0
with open(f"{OUT}/{LABEL}.jsonl", "a") as out:
    for depth, kind in CELLS:
        content = prompt(depth)
        psha = hashlib.sha256(content.encode()).hexdigest()
        for mode in os.environ.get("MODES", "draft,plain").split(","):
            for rep in range(REPS + 1):
                r = post(content, mode)
                r.update(label=LABEL, depth=depth, kind=kind, mode=mode, rep=rep, warmup=rep == 0, prompt_sha256=psha)
                out.write(json.dumps(r) + "\n")
                out.flush()
                failed += not r["ok"]
                print(f"{LABEL} d={depth} {kind} {mode} rep{rep}: pt={r['prompt_tokens']} n={r['completion_tokens']} "
                      f"{r['decode_tok_s'] or 0:.2f} tok/s tsha={(r['token_sha256'] or 'NONE')[:16]}", flush=True)
print("FINI failed=%d" % failed, flush=True)
sys.exit(1 if failed else 0)

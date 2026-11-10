#!/usr/bin/env python3
"""Served decode by depth, one JSON line per request. usage: bench_merge.py URL LABEL OUTDIR [reps]

Per (depth, kind, mode) cell: 1 warm-up request then `reps` measured ones. A request without a
tensorfold.token_sha256 fingerprint is a failure. Inputs are the two corpora next to this script.
Env: CELLS="0:code,text;8192:code,text;32768:code,text;131072:code" (depth:kinds).
"""
import hashlib, json, os, sys, time, urllib.request

URL, LABEL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 3
CELLS = [(int(c.split(":")[0]), c.split(":")[1].split(",")) for c in
         os.environ.get("CELLS", "0:code,text;8192:code,text;32768:code,text;131072:code").split(";")]
H = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)
CORP = {k: open(f"{H}/inputs/corp_{k}.txt", errors="ignore").read() for k in ("code", "text")}
ASK = {"code": "Review the code above in detail, then write one new, complete Python function in the same style.",
       "text": "Write a detailed, well structured summary of the documentation above."}
SHORT = {"code": "Write a complete Python function that parses a CSV file and prints the average of each numeric column, "
                 "with comments.",
         "text": "Write a detailed essay of about two hundred words on why local inference matters for privacy."}


def post(content, draft):
    body = {"model": "m", "messages": [{"role": "user", "content": content}], "max_tokens": 256, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    if not draft:
        body["draft"] = False
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0, first, text, usage, tsha = time.time(), None, [], None, None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: {"):
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                usage = d["usage"]
            tf = d.get("tensorfold") or (d.get("usage") or {}).get("tensorfold") or {}
            if tf.get("token_sha256"):
                tsha = tf["token_sha256"]
            c = (d.get("choices") or [{}])[0].get("delta", {}).get("content")
            if c:
                first = first or time.time()
                text.append(c)
    t1, s = time.time(), "".join(text)
    n = (usage or {}).get("completion_tokens", 0)
    ok = bool(text and usage and tsha)
    return {"ok": ok, "token_sha256": tsha, "text_sha256": hashlib.sha256(s.encode()).hexdigest(),
            "prompt_tokens": (usage or {}).get("prompt_tokens"), "completion_tokens": n,
            "ttft_s": (first or t1) - t0,
            "decode_tok_s": (n - 1) / (t1 - first) if first and t1 > first and n > 1 else None}


tokens_per_char = {}


def calib(kind):
    if kind not in tokens_per_char:
        r = post(CORP[kind][:20000], False)
        tokens_per_char[kind] = 20000 / (r["prompt_tokens"] - 15)
    return tokens_per_char[kind]


def prompt(kind, depth):
    if depth == 0:
        return SHORT[kind]
    scale = 0.88 if (kind == "text" and depth >= 131072) else 1.0
    return "Here is the material:\n\n" + CORP[kind][:int((depth - 300) * calib(kind) * scale)] + "\n\n" + ASK[kind]


failed = 0
with open(f"{OUT}/{LABEL}.jsonl", "a") as out:
    for depth, kinds in CELLS:
        for kind in kinds:
            content = prompt(kind, depth)
            psha = hashlib.sha256(content.encode()).hexdigest()
            for mode in ("draft", "plain"):
                for rep in range(REPS + 1):
                    r = post(content, mode == "draft")
                    r.update(label=LABEL, depth=depth, kind=kind, mode=mode, rep=rep, warmup=rep == 0, prompt_sha256=psha)
                    out.write(json.dumps(r) + "\n")
                    out.flush()
                    failed += not r["ok"]
                    print(f"{LABEL} d={depth} {kind} {mode} rep{rep}: pt={r['prompt_tokens']} n={r['completion_tokens']} "
                          f"{r['decode_tok_s'] or 0:.2f} tok/s tsha={(r['token_sha256'] or 'NONE')[:16]}", flush=True)
print("FINI failed=%d" % failed, flush=True)
sys.exit(1 if failed else 0)

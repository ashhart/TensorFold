#!/usr/bin/env python3
"""Sampled served decode, P streams a wave; env CELLS, GREEDY, SEED0. usage: bench_topk.py URL LABEL OUTDIR P [reps]"""
import hashlib, json, os, sys, threading, time, urllib.request

URL, LABEL, OUT, P = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
REPS = int(sys.argv[5]) if len(sys.argv) > 5 else 3
CELLS = [(int(c.split(":")[0]), c.split(":")[1].split(",")) for c in os.environ.get("CELLS", "0:code,text;8192:code,text").split(";")]
GREEDY = os.environ.get("GREEDY") == "1"
SEED0 = int(os.environ.get("SEED0", "1000"))
H = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)
CORP = {k: open(f"{H}/inputs/corp_{k}.txt", errors="ignore").read() for k in ("code", "text")}
ASK = {"code": "Review the code above in detail, then write one new, complete Python function in the same style.",
       "text": "Write a detailed, well structured summary of the documentation above."}
SHORT = {"code": "Write a complete Python function that parses a CSV file and prints the average of each numeric column, "
                 "with comments.",
         "text": "Write a detailed essay of about two hundred words on why local inference matters for privacy."}


def post(content, seed):
    body = {"model": "m", "messages": [{"role": "user", "content": content}], "max_tokens": 256, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    if GREEDY:
        body["temperature"] = 0
    else:
        body["seed"] = seed
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
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
    return {"ok": bool(text and usage and tsha), "token_sha256": tsha, "text_sha256": hashlib.sha256(s.encode()).hexdigest(),
            "prompt_tokens": (usage or {}).get("prompt_tokens"), "completion_tokens": n, "t0": t0, "t_first": first or t1, "t1": t1,
            "ttft_s": (first or t1) - t0, "decode_tok_s": (n - 1) / (t1 - first) if first and t1 > first and n > 1 else None}


tpc = {}


def prompt(kind, depth):
    if depth == 0:
        return SHORT[kind]
    if kind not in tpc:
        r = post(CORP[kind][:20000], 1)
        tpc[kind] = 20000 / (r["prompt_tokens"] - 15)
    return "Here is the material:\n\n" + CORP[kind][:int((depth - 300) * tpc[kind])] + "\n\n" + ASK[kind]


failed = 0
with open(f"{OUT}/{LABEL}.jsonl", "a") as out:
    for depth, kinds in CELLS:
        for kind in kinds:
            content = prompt(kind, depth)
            psha = hashlib.sha256(content.encode()).hexdigest()
            for rep in range(REPS + 1):
                res = [None] * P

                def go(i):
                    try:
                        res[i] = post(content, SEED0 + i)
                    except Exception as e:
                        res[i] = {"ok": False, "error": repr(e)}
                ts = [threading.Thread(target=go, args=(i,)) for i in range(P)]
                w0 = time.time()
                [t.start() for t in ts]
                [t.join() for t in ts]
                wall = time.time() - w0
                for i, r in enumerate(res):
                    r.update(label=LABEL, parallel=P, depth=depth, kind=kind, rep=rep, warmup=rep == 0, stream=i,
                             seed=None if GREEDY else SEED0 + i, greedy=GREEDY, wave_wall_s=wall, prompt_sha256=psha)
                    out.write(json.dumps(r) + "\n")
                    failed += not r["ok"]
                out.flush()
                oks = [r for r in res if r["ok"]]
                tot = sum(r["completion_tokens"] for r in oks)
                mean = sum(r["decode_tok_s"] or 0 for r in oks) / max(1, len(oks))
                print(f"{LABEL} P={P} d={depth} {kind} rep{rep}: n={tot} wall={wall:.1f}s per-stream={mean:.2f} tok/s "
                      f"tsha0={(res[0].get('token_sha256') or 'NONE')[:16]}", flush=True)
print("FINI failed=%d" % failed, flush=True)
sys.exit(1 if failed else 0)

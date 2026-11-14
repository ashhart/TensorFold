#!/usr/bin/env python3
"""Four simultaneous streams, one JSON line per request. usage: par4.py URL LABEL OUTDIR [reps]

Same prompts as bench_merge.py (same corpora, same calibration, same wording), at depths 0 (code), 8192 (code),
32768 (code) and 32768 (text), sent at the same moment from four threads. Per mode (draft, plain): one warm-up wave
and `reps` measured waves. A request without a tensorfold.token_sha256 fingerprint is a failure.
"""
import hashlib, json, os, sys, threading, time, urllib.request

URL, LABEL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 3
H = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)
CORP = {k: open(f"{H}/inputs/corp_{k}.txt", errors="ignore").read() for k in ("code", "text")}
ASK = {"code": "Review the code above in detail, then write one new, complete Python function in the same style.",
       "text": "Write a detailed, well structured summary of the documentation above."}
SHORT = {"code": "Write a complete Python function that parses a CSV file and prints the average of each numeric column, "
                 "with comments.",
         "text": "Write a detailed essay of about two hundred words on why local inference matters for privacy."}
STREAMS = [(0, "code"), (8192, "code"), (32768, "code"), (32768, "text")]


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
            "ttft_s": (first or t1) - t0, "t_start": t0, "t_first": first, "t_end": t1,
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
    return "Here is the material:\n\n" + CORP[kind][:int((depth - 300) * calib(kind) * 1.0)] + "\n\n" + ASK[kind]


contents = [prompt(k, d) for d, k in STREAMS]
failed = 0
with open(f"{OUT}/{LABEL}.jsonl", "a") as out:
    for mode in ("draft", "plain"):
        for rep in range(REPS + 1):
            res = [None] * len(STREAMS)
            go = threading.Barrier(len(STREAMS))

            def work(i):
                go.wait()
                res[i] = post(contents[i], mode == "draft")

            th = [threading.Thread(target=work, args=(i,)) for i in range(len(STREAMS))]
            for t in th:
                t.start()
            for t in th:
                t.join()
            wall = max(r["t_end"] for r in res) - min(r["t_start"] for r in res)
            for i, r in enumerate(res):
                r.update(label=LABEL, depth=STREAMS[i][0], kind=STREAMS[i][1], mode=mode, rep=rep, warmup=rep == 0,
                         wave_wall_s=wall, prompt_sha256=hashlib.sha256(contents[i].encode()).hexdigest())
                out.write(json.dumps(r) + "\n")
                failed += not r["ok"]
                print(f"{LABEL} {mode} rep{rep} d={STREAMS[i][0]} {STREAMS[i][1]}: pt={r['prompt_tokens']} "
                      f"n={r['completion_tokens']} {r['decode_tok_s'] or 0:.2f} tok/s "
                      f"tsha={(r['token_sha256'] or 'NONE')[:16]}", flush=True)
            out.flush()
print("FINI failed=%d" % failed, flush=True)
sys.exit(1 if failed else 0)

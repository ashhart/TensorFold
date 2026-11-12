#!/usr/bin/env python3
"""Served decode tok/s with and without logprobs (top_logprobs 5), alternating, same server. usage: CORP=FILE bench.py PORT OUT.json [reps]"""
import json, os, statistics, sys, threading, time, urllib.request
PORT, OUT = sys.argv[1], sys.argv[2]
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 5
CORP = open(os.environ["CORP"], errors="ignore").read()  # any large source-code text, at least 30,000 characters
SHORT = "Write a long, detailed Python program with many functions and comments that implements a small library management system."
LONG = "Here is the material:\n\n" + CORP[:30000] + "\n\nReview the code above in detail, then write one new, complete Python function in the same style."
def post(content, draft, sampled, lp, n=256):
    b = {"model": "m", "messages": [{"role": "user", "content": content}], "max_tokens": n, "stream": True,
         "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    b.update(dict(temperature=1.0, top_k=20, top_p=0.95, seed=7) if sampled else {"temperature": 0})
    if not draft: b["draft"] = False
    if lp: b.update(logprobs=True, top_logprobs=5)
    r = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(b).encode(), headers={"Content-Type": "application/json"})
    first = last = None; n_tok = 0; usage = None
    with urllib.request.urlopen(r, timeout=3600) as f:
        for line in f:
            line = line.decode().strip()
            if not line.startswith("data: {"): continue
            d = json.loads(line[6:])
            if d.get("usage"): usage = d["usage"]
            if (d.get("choices") or [{}])[0].get("delta", {}).get("content"):
                last = time.time(); first = first or last
    n = usage["completion_tokens"]
    return {"n": n, "prompt": usage["prompt_tokens"], "tps": (n - 1) / (last - first) if last > first else None}
def cell(content, draft, sampled, lp, conc):
    out = [None] * conc
    def go(k): out[k] = post(content, draft, sampled, lp)
    ts = [threading.Thread(target=go, args=(k,)) for k in range(conc)]
    t0 = time.time(); [t.start() for t in ts]; [t.join() for t in ts]; dt = time.time() - t0
    return sum(o["n"] for o in out) / dt if conc > 1 else out[0]["tps"], out[0]["prompt"], out[0]["n"]
CELLS = [("short drafted greedy", SHORT, True, False, 1), ("short drafted sampled", SHORT, True, True, 1),
         ("short plain greedy", SHORT, False, False, 1), ("8k drafted greedy", LONG, True, False, 1),
         ("short drafted greedy x2 concurrent (aggregate incl. prefill)", SHORT, True, False, 2)]
res = []
for name, content, draft, sampled, conc in CELLS:
    cell(content, draft, sampled, False, conc); cell(content, draft, sampled, True, conc)  # warm-up
    t = {False: [], True: []}
    for rep in range(REPS):
        for lp in ((False, True) if rep % 2 == 0 else (True, False)):
            v, pt, n = cell(content, draft, sampled, lp, conc); t[lp].append(v)
        print(name, rep, t[False][-1], t[True][-1], flush=True)
    a, b = statistics.median(t[False]), statistics.median(t[True])
    res.append({"cell": name, "prompt_tokens": pt, "completion_tokens": n, "tok_s_without": a, "tok_s_with": b, "cost_pct": (a - b) / a * 100, "runs_without": t[False], "runs_with": t[True]})
    print(res[-1], flush=True)
json.dump(res, open(OUT, "w"), indent=1)

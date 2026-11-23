#!/usr/bin/env python3
"""3 greedy decoding streams, then a cold 4096-token prompt at t=8 s: prompt first token and the decode streams' worst gap."""
import json, subprocess, sys, threading, time, urllib.request

base, ck, label = sys.argv[1], sys.argv[2], sys.argv[3]
t0 = time.time()
stamps = [[] for _ in range(3)]
prompt_res = {}


def stream(i):
    body = json.dumps({"model": "m", "stream": True, "max_tokens": 700, "temperature": 0,
                       "messages": [{"role": "user", "content": f"Write a long Python module number {i} with many small functions and docstrings."}]})
    req = urllib.request.Request(base + "/chat/completions", body.encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            if line.startswith(b"data: ") and b"[DONE]" not in line:
                stamps[i].append(time.time() - t0)


def prompt():
    time.sleep(8)
    s = time.time() - t0
    out = subprocess.run([sys.executable, "cold_prefill.py", base, "--model", "m", "--ck", ck, "--sizes", "4096",
                          "--label", label, "--output", f"/tmp/mixed-{label}.json"], capture_output=True, text=True)
    j = json.load(open(f"/tmp/mixed-{label}.json"))
    prompt_res["start"] = s
    prompt_res["end"] = time.time() - t0
    prompt_res["json"] = j


ths = [threading.Thread(target=stream, args=(i,)) for i in range(3)] + [threading.Thread(target=prompt)]
for t in ths:
    t.start()
for t in ths:
    t.join()
s, e = prompt_res["start"], prompt_res["end"]
gaps = []
chunks_in = 0
for st in stamps:
    g = [b - a for a, b in zip(st, st[1:]) if b > s]
    gaps.append(max(g) if g else None)
    chunks_in += sum(1 for x in st if s <= x <= e)
print(json.dumps({"label": label, "prompt_window_s": round(e - s, 2), "decode_chunks_during_prompt": chunks_in,
                  "decode_max_gap_s": [round(g, 2) if g else None for g in gaps],
                  "decode_end_s": [round(st[-1], 1) for st in stamps], "decode_chunks": [len(st) for st in stamps]}))

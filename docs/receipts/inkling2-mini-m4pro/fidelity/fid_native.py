#!/usr/bin/env python3
"""Native side, teacher forced by restarts (the native Inkling backend has no prompt_logprobs). Per prompt: greedy
completion with logprobs=5 from prompt + MLX's ids[:p]; at the first token that differs from MLX's, record both,
force MLX's token (p moves past it) and request again. Raw replies are saved.
python fid_native.py BASE_URL CKPT prompts.json mlx.json out.json"""
import json, sys, time, urllib.request
base, ck, pf, mf, out = sys.argv[1:6]
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(ck)
P, M = json.load(open(pf)), json.load(open(mf))

def post(prompt, n):
    body = {"model": "m", "prompt": prompt, "max_tokens": n, "temperature": 0, "logprobs": 5}
    r = urllib.request.Request(base + "/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(r, timeout=7200) as f:
        d = json.loads(f.read())
    return d, time.perf_counter() - t

R = {}
for name, text in P.items():
    m = M[name]; ids = m["ids"]; pids = m["prompt_ids"]
    strs = [tok.decode([i]) for i in ids]
    steps = []  # one per position: native token string, native logprob, top5 dict, forced flag
    free = None; p = 0; calls = []
    while p < 256:
        prompt = text if p == 0 else tok.decode(pids + ids[:p])
        d, sec = post(prompt, 256 - p)
        c = d["choices"][0]; lp = c["logprobs"]
        calls.append({"p": p, "seconds": sec, "prompt_tokens": d["usage"]["prompt_tokens"], "finish": c["finish_reason"],
                      "n": len(lp["tokens"])})
        if p == 0: free = c
        q = 0
        while q < len(lp["tokens"]) and p + q < 256 and lp["tokens"][q] == strs[p + q]:
            steps.append({"j": p + q, "agree": True, "lp": lp["token_logprobs"][q], "top": lp["top_logprobs"][q]}); q += 1
        if p + q >= 256 or q >= len(lp["tokens"]):
            p += q
            if q >= len(lp["tokens"]) and p < 256:  # native stopped early (EOS): record and force past it
                steps.append({"j": p, "agree": False, "native": None, "lp": None, "top": {}}); p += 1
            continue
        steps.append({"j": p + q, "agree": False, "native": lp["tokens"][q], "native_lp": lp["token_logprobs"][q],
                      "lp": lp["top_logprobs"][q].get(strs[p + q]), "top": lp["top_logprobs"][q]})
        p += q + 1
    R[name] = {"free": free, "calls": calls, "steps": steps, "strs": strs}
    print(name, [(c["p"], c["prompt_tokens"], round(c["seconds"], 1)) for c in calls], flush=True)
    json.dump(R, open(out, "w"))

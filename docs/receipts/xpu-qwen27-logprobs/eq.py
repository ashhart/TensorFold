#!/usr/bin/env python3
"""Tokens and logprobs with and without logprobs, drafted and plain, greedy and sampled (seed 7). usage: CORP=FILE eq.py PORT OUT.json"""
import json, os, sys, urllib.request
PORT, OUT = sys.argv[1], sys.argv[2]
CORP = open(os.environ["CORP"], errors="ignore").read()  # any large source-code text, at least 30,000 characters
PROMPTS = [
    "Write a complete Python function that parses a CSV file and prints the average of each numeric column, with comments.",
    "Explain in three short paragraphs why the sky is blue and why sunsets are red.",
    "List the first twelve prime numbers, then write a Rust function that tests primality.",
    "Translate to French: The quick brown fox jumps over the lazy dog, and then goes back to sleep under the old oak tree.",
    "Here is some code:\n\n" + CORP[:6000] + "\n\nReview it and write one new function in the same style.",
    "Write a haiku about debugging, then a limerick about memory leaks.",
]
SAMPLED = dict(temperature=1.0, top_k=20, top_p=0.95, seed=7)
def post(content, sampled, draft, lp, n):
    b = {"model": "m", "messages": [{"role": "user", "content": content}], "max_tokens": n, "stream": True,
         "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    b.update(SAMPLED if sampled else {"temperature": 0})
    if not draft: b["draft"] = False
    if lp: b.update(logprobs=True, top_logprobs=5)
    r = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(b).encode(), headers={"Content-Type": "application/json"})
    sha, rows, text = None, [], []
    with urllib.request.urlopen(r, timeout=3600) as f:
        for line in f:
            line = line.decode().strip()
            if not line.startswith("data: {"): continue
            d = json.loads(line[6:])
            tf = d.get("tensorfold") or (d.get("usage") or {}).get("tensorfold") or {}
            if tf.get("token_sha256"): sha = tf["token_sha256"]
            for c in d.get("choices") or []:
                lpc = (c.get("logprobs") or {}).get("content") or []
                rows += lpc
                t = (c.get("delta") or {}).get("content")
                if t: text.append(t)
    return {"sha": sha, "rows": rows, "text": "".join(text)}
res, bad = [], 0
for pi, p in enumerate(PROMPTS):
    for sampled in (False, True):
        cell = {}
        for draft in (True, False):
            for lp in (False, True):
                cell[(draft, lp)] = post(p, sampled, draft, lp, 160)
        for draft in (True, False):
            a, b = cell[(draft, False)], cell[(draft, True)]
            same = a["sha"] is not None and a["sha"] == b["sha"] and a["text"] == b["text"]
            bad += not same
            res.append({"prompt": pi, "sampled": sampled, "draft": draft, "check": "sha_with_eq_without", "ok": same, "sha": a["sha"], "n_rows": len(b["rows"])})
        d, q = cell[(True, True)]["rows"], cell[(False, True)]["rows"]
        toks_eq = cell[(True, True)]["sha"] == cell[(False, True)]["sha"]
        same_len = len(d) == len(q) and len(d) > 0
        mx, topdiff, tokdiff = 0.0, 0, 0
        if same_len:
            for x, y in zip(d, q):
                tokdiff += x["token"] != y["token"]
                mx = max(mx, abs(x["logprob"] - y["logprob"]))
                tx, ty = x["top_logprobs"], y["top_logprobs"]
                topdiff += [t["token"] for t in tx] != [t["token"] for t in ty]
                for s, t in zip(tx, ty): mx = max(mx, abs(s["logprob"] - t["logprob"]))
        ok = same_len and tokdiff == 0 and topdiff == 0 and mx == 0.0
        bad += not ok
        res.append({"prompt": pi, "sampled": sampled, "check": "drafted_vs_plain_logprobs", "ok": ok, "tokens_eq": toks_eq, "rows": len(d), "max_abs_diff": mx, "top_order_diff": topdiff, "token_diff": tokdiff})
        print(res[-2], res[-1], flush=True)
json.dump(res, open(OUT, "w"), indent=1)
print("FAILED" if bad else "ALL OK", bad)

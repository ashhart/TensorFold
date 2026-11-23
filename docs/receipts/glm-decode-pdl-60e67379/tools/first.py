#!/usr/bin/env python3
"""first.py PORT PROMPT_JSON N REPS: REPS cold greedy completions of N tokens, drafts off; prints the reply-sha histogram."""
import collections
import hashlib
import json
import sys
import urllib.request

port, prompt, n, reps = sys.argv[1], json.load(open(sys.argv[2])), int(sys.argv[3]), int(sys.argv[4])
seen = collections.Counter()
for _ in range(reps):
    body = {"model": "local-model", "prompt": prompt, "max_tokens": n, "temperature": 0, "ignore_eos": True, "draft": False}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=3600))
    seen[hashlib.sha256(r["choices"][0]["text"].encode()).hexdigest()[:8]] += 1
print(dict(seen))

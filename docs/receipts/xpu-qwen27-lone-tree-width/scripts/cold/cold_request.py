#!/usr/bin/env python3
"""One chat request, the first of a fresh server, as one JSON line. usage: cold_request.py URL PROMPTFILE OUT.jsonl KEY=VALUE...

The request is bench_tree.py's draft mode: greedy, 256 new tokens, thinking off, streamed. decode_tok_s is its definition,
(tokens - 1) / (last chunk - first token); ttft_s is the time to the first content chunk. The KEY=VALUE pairs (arm, prompt,
server, ...) are copied into the record. Exit 1 when the reply has no token_sha256 or no speculative object.
"""
import hashlib, json, sys, time, urllib.request

URL, PROMPT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
tags = dict(a.split("=", 1) for a in sys.argv[4:])
content = open(PROMPT).read()
body = {"model": "m", "messages": [{"role": "user", "content": content}], "max_tokens": 256, "temperature": 0,
        "stream": True, "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
t0, first, text, usage, tsha, spec = time.time(), None, [], None, None, None
with urllib.request.urlopen(req, timeout=900) as r:
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
ok = bool(text and usage and tsha and spec is not None)
rec = {"ok": ok, "speculative": spec, "token_sha256": tsha, "text_sha256": hashlib.sha256(s.encode()).hexdigest(),
       "prompt_tokens": (usage or {}).get("prompt_tokens"), "completion_tokens": n,
       "ttft_s": (first or t1) - t0,
       "decode_tok_s": (n - 1) / (t1 - first) if first and t1 > first and n > 1 else None,
       "nodes_per_round": (spec["drafted"] / spec["rounds"]) if spec and spec.get("rounds") else None,
       "prompt_sha256": hashlib.sha256(content.encode()).hexdigest(), **tags}
with open(OUT, "a") as f:
    f.write(json.dumps(rec) + "\n")
print(f"{tags} pt={rec['prompt_tokens']} n={n} {rec['decode_tok_s'] or 0:.2f} tok/s ttft={rec['ttft_s']:.2f}s "
      f"nodes/round={rec['nodes_per_round'] or 0:.2f} tsha={(tsha or 'NONE')[:16]}", flush=True)
sys.exit(0 if ok else 1)

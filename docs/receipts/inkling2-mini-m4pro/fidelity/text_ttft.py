import json, sys, time, urllib.request
base, model = sys.argv[1:3]
SENT = "Explain how a residual stream mixes attention outputs. "
def ask(prompt, n=1):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": n, "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    r = urllib.request.Request(base + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.perf_counter(); first = None; usage = None
    with urllib.request.urlopen(r, timeout=1800) as f:
        for line in f:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]": continue
            j = json.loads(line[5:])
            if j.get("usage"): usage = j["usage"]
            for c in j.get("choices", []):
                if (c.get("delta", {}).get("content") or "") and first is None: first = time.perf_counter() - t0
    return first, usage and usage.get("prompt_tokens")
ask("Say hi.", 4)
for k in (59, 171):
    print("text", k, [ask(SENT * k + str(i)) for i in range(3)], flush=True)

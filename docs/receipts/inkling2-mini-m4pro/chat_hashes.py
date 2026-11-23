import sys, json, urllib.request
base = sys.argv[1]
SENT = "Explain how a residual stream mixes attention outputs. "
def chat(prompt, k):
    body = json.dumps({"model": "m", "max_tokens": k, "temperature": 0, "stream": False, "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(base + "/chat/completions", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r: raw = json.loads(r.read())
    return raw["usage"]["prompt_tokens"], raw["choices"][0]["message"].get("content", "")[:40]
print(chat("Write a Python function that returns the n-th Fibonacci number.", 8))
print(chat("Write a Python function that returns the n-th Fibonacci number.", 64))
print(chat(SENT * 40, 16))
print(chat(SENT * 102, 24))

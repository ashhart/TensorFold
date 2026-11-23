#!/usr/bin/env python3
"""Image requests through /v1/chat/completions image_url parts, greedy, streamed to time the first token.
python img_client.py BASE_URL MODEL IMG_DIR OUT.json [--no-warm]"""
import base64, json, sys, time, urllib.request
base, model, d, out = sys.argv[1:5]
CASES = [
    ("text-small.png", "What text is written in this image? Answer with the text only."),
    ("shapes.png", "List the shapes in this image and their colors."),
    ("chart.png", "Which weekday has the highest sales, and what is its value?"),
    ("chart.jpg", "Which weekday has the highest sales, and what is its value?"),
    ("scene.jpg", "Describe this image in two sentences."),
    ("invoice-large.jpg", "What is the total due and what is the invoice number?"),
    ("invoice-large.png", "What is the total due and what is the invoice number?"),
]

CASES = [c for c in CASES if c[0] in ("scene.jpg", "invoice-large.jpg", "invoice-large.png")]

def ask(content, n=160):
    body = {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": n, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True}}
    r = urllib.request.Request(base + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.perf_counter(); first = None; text = ""; usage = None; err = None
    try:
        with urllib.request.urlopen(r, timeout=1800) as f:
            for line in f:
                line = line.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]": continue
                j = json.loads(line[5:])
                if j.get("usage"): usage = j["usage"]
                for c in j.get("choices", []):
                    piece = c.get("delta", {}).get("content") or ""
                    if piece and first is None: first = time.perf_counter() - t0
                    text += piece
    except urllib.error.HTTPError as e:
        err = f"HTTP {e.code}: {e.read().decode()[:300]}"
    return {"ttft": first, "total": time.perf_counter() - t0, "text": text, "usage": usage, "error": err}

R = []
if "--no-warm" not in sys.argv:
    ask([{"type": "text", "text": "Say hi."}], 8)
for name, q in CASES:
    mime = "image/jpeg" if name.endswith("jpg") else "image/png"
    url = f"data:{mime};base64," + base64.b64encode(open(f"{d}/{name}", "rb").read()).decode()
    content = [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": q}]
    runs = [ask(content) for _ in range(2)]
    R.append({"image": name, "question": q, "runs": runs})
    print(name, [(round(x["ttft"], 2) if x["ttft"] else None, x["usage"] and x["usage"].get("prompt_tokens"), x["error"]) for x in runs],
          repr(runs[0]["text"][:150]), flush=True)
    json.dump(R, open(out, "w"), indent=1)

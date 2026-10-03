"""Client streaming throughput of an OpenAI-compatible server on three fixed prompts (greedy, 256 tokens, thinking off).

    python3 tools/stream_bench.py PORT TAG MODEL [EXTRA_JSON] [--out DIR]

rate = (completion_tokens - 1) / (time of last content chunk - time of first content chunk); completion_tokens comes
from the server's ``usage``. A chunk may carry several drafted tokens, so this is client streaming throughput, not
GPU decode time. Replies are written to DIR/stream_TAG.json.
"""
import json, os, sys, time, urllib.request
argv = [a for a in sys.argv[1:] if a != "--out"]
out_dir = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "."
if out_dir in argv:
    argv.remove(out_dir)
port, tag, model = argv[0], argv[1], argv[2]
extra = json.loads(argv[3]) if len(argv) > 3 else {}
prompts = {
 "rdma": "Explain in two paragraphs what RDMA is and why it lowers latency compared with TCP.",
 "code": "Write a Python function that merges two sorted lists into one sorted list, with a docstring and three doctests.",
 "story": "Write a short story of about 200 words about a lighthouse keeper who finds a message in a bottle.",
}
out = {}
for name, p in prompts.items():
    body = json.dumps({"model": model, **extra, "stream_options": {"include_usage": True}, "messages": [{"role": "user", "content": p}], "max_tokens": 256,
        "temperature": 0, "stream": True, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions", body, {"Content-Type": "application/json"})
    t0 = time.time(); first = None; times = []; text = ""; last = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data: ") or line == b"data: [DONE]": continue
            d = json.loads(line[6:]); last = d
            ch = d.get("choices") or []
            c = (ch[0].get("delta") or {}).get("content") if ch else None
            if c:
                now = time.time(); first = first or now; times.append(now); text += c
    usage = (last or {}).get("usage") or {}
    ntok = usage.get("completion_tokens")
    dec = (ntok - 1) / (times[-1] - first) if ntok and len(times) > 1 else None
    print(json.dumps({"tag": tag, "prompt": name, "ttft_s": round(first - t0, 3), "completion_tokens": ntok,
        "decode_tok_s": round(dec, 2) if dec else None, "chunks": len(times), "timings": (last or {}).get("timings")}))
    out[name] = text
os.makedirs(out_dir, exist_ok=True)
json.dump(out, open(os.path.join(out_dir, f"stream_{tag}.json"), "w"))

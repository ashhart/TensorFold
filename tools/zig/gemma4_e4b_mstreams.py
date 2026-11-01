"""Streams in parallel against a Bolt server: each reply equal to its solo reply, aggregate tokens/s, TTFT."""
import argparse, hashlib, json, threading, time, urllib.request
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("prompts", type=Path, help="the reference prompts (gemma4_e4b_prompts.py)")
ap.add_argument("french", type=Path, help="the French robot scenarios (gemma4_e4b_prompts.py --french)")
ap.add_argument("--port", default="8081")
ap.add_argument("--sampled", action="store_true", help="temperature 1, a seed per stream (greedy otherwise)")
ap.add_argument("--nodraft", action="store_true")
ap.add_argument("--max-tokens", type=int, default=128)
args = ap.parse_args()
PORT, SAMPLED, DRAFT, MAXT = args.port, args.sampled, not args.nodraft, args.max_tokens
P = {p["name"]: p for p in json.loads(args.prompts.read_text())}
P.update({p["name"]: p for p in json.loads(args.french.read_text())})
NAMES = ["code", "chat_short", "duck_wall", "french", "think", "fr_danse", "multiturn", "fr_porte_assis"]


def request(name, idx):
    p = P[name]
    b = {"model": "x", "messages": p["messages"], "temperature": 0, "max_tokens": MAXT, "stream": True, "stream_options": {"include_usage": True},
         "chat_template_kwargs": {"enable_thinking": p.get("enable_thinking", False)}, "draft": DRAFT}
    if p.get("tools"):
        b["tools"] = p["tools"]
    if SAMPLED:
        b.update(temperature=1.0, top_p=0.95, top_k=64, seed=100 + idx)
    t0 = time.time()
    r = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(b).encode(), {"Content-Type": "application/json"}))
    text, first, usage = [], None, 0
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        d = json.loads(line[6:])
        if d.get("usage"):
            usage = d["usage"]["completion_tokens"]
        for c in d.get("choices", []):
            dl = c.get("delta", {})
            tc = dl.get("tool_calls")
            if tc:  # the arguments arrive in pieces: only their concatenation is comparable
                for t in tc:
                    f = t.get("function", {})
                    text.append("\u0001" + str(t.get("index", 0)) + (f.get("name") or "") + (f.get("arguments") or ""))
            piece = (dl.get("content") or "") + (dl.get("reasoning_content") or "")
            if piece:
                if first is None:
                    first = time.time() - t0
                text.append(piece)
    return hashlib.sha256("".join(text).encode()).hexdigest()[:12], usage, first, time.time() - t0


solo = {}
for i, n in enumerate(NAMES):
    request(n, i)  # warm (prefix cache)
    solo[n] = request(n, i)
print("solo:", " ".join(f"{n}:{solo[n][1]}tok/{solo[n][3]*1000:.0f}ms" for n in NAMES))
solo_rate = sum(solo[n][1] for n in NAMES) / sum(solo[n][3] for n in NAMES)
print(f"solo aggregate {solo_rate:.1f} tok/s ({'sampled' if SAMPLED else 'greedy'}, drafts {'on' if DRAFT else 'off'})")
for N in (1, 2, 4, 8):
    res = [None] * N

    def go(i):
        res[i] = request(NAMES[i], i)

    th = [threading.Thread(target=go, args=(i,)) for i in range(N)]
    t0 = time.time()
    [t.start() for t in th]
    [t.join() for t in th]
    wall = time.time() - t0
    toks = sum(r[1] for r in res)
    eq = sum(res[i][0] == solo[NAMES[i]][0] for i in range(N))
    ttft = sorted(r[2] for r in res if r[2] is not None)
    print(f"{N} streams: {toks} tokens in {wall*1000:.0f} ms = {toks/wall:.1f} tok/s aggregate ({toks/wall/solo_rate:.2f}x solo), equal to solo {eq}/{N}, TTFT median {ttft[len(ttft)//2]*1000:.0f} ms max {ttft[-1]*1000:.0f} ms")

"""A long prompt arriving while streams decode; usage: serve_conc.py URL MODEL [DECODERS] [PROMPT_WORDS] [TOKENS]."""

import http.client
import json
import sys
import threading
import time
import urllib.parse

URL, MODEL = sys.argv[1], sys.argv[2]
DECODERS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
WORDS = int(sys.argv[4]) if len(sys.argv) > 4 else 2400
TOKENS = int(sys.argv[5]) if len(sys.argv) > 5 else 400
HOST = urllib.parse.urlparse(URL)
SHORT = ["Write a long story about a lighthouse keeper.", "Explain how a transformer works, in detail.",
         "List one hundred animals, one per line.", "Describe the history of Rome at length."]


def stream(text, max_tokens, stamps, temperature=0.0):
    """One streamed chat request: the time of every content chunk goes to `stamps`; returns the text."""
    body = {"model": MODEL, "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens,
            "temperature": temperature, "stream": True, "chat_template_kwargs": {"enable_thinking": False}}
    conn = http.client.HTTPConnection(HOST.hostname, HOST.port, timeout=900)
    conn.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    reply = conn.getresponse()
    out = []
    for raw in reply:
        line = raw.decode().strip()
        if not line.startswith("data:") or line.endswith("[DONE]"):
            continue
        delta = json.loads(line[5:])["choices"][0].get("delta", {})
        if delta.get("content"):
            stamps.append(time.perf_counter())
            out.append(delta["content"])
    conn.close()
    return "".join(out)


def long_prompt(nonce):
    words = [f"{nonce}"] + [f"item{i % 97}x{(i * 7919) % 1013}" for i in range(WORDS)]
    return "Summarise the following list in one short sentence.\n" + " ".join(words)


def run(label):
    stamps = [[] for _ in range(DECODERS)]
    threads = [threading.Thread(target=stream, args=(SHORT[i % len(SHORT)], TOKENS, stamps[i])) for i in range(DECODERS)]
    for t in threads:
        t.start()
    while min(len(s) for s in stamps) < 16:
        time.sleep(0.05)
    sent = time.perf_counter()
    first = []
    text = long_prompt(f"{label}-{time.time_ns()}")
    t = threading.Thread(target=stream, args=(text, 16, first))
    t.start()
    t.join()
    done = time.perf_counter()
    for th in threads:
        th.join()
    ttft = first[0] - sent
    gaps = []
    tokens = 0
    for s in stamps:
        for a, b in zip(s, s[1:]):
            if b > sent and a < first[0]:
                gaps.append(b - a)
        tokens += sum(1 for x in s if sent <= x <= first[0])
    gaps.sort()
    window = first[0] - sent
    print(f"{label}: long prompt first token {ttft:.2f} s, decoding streams: longest gap {gaps[-1]:.2f} s, "
          f"median gap {gaps[len(gaps) // 2] * 1000:.0f} ms, {tokens / window:.1f} tok/s over the prompt's window "
          f"(the prompt ended at {done - sent:.2f} s)", flush=True)


for i in range(2):
    run(f"run{i}")

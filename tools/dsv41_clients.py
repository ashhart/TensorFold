"""N concurrent chat clients for T seconds against a running server: aggregate completion tok/s.

  TF_API_KEY=... python3 tools/dsv41_clients.py 16 120 [--base http://127.0.0.1:8888]

Each client sends one request after another (varied prompts, 256 tokens, thinking off) until the time is up; requests
still running then finish and count, and the rate is over the wall time to the last reply.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import urllib.request

PROMPTS = [
    "Write a Python function that merges overlapping intervals, with a short docstring.",
    "Explain how a B-tree keeps itself balanced during inserts.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "List ten practical tips for reducing memory use in a long-running Python service.",
    "Describe the differences between TCP congestion control algorithms Reno, CUBIC and BBR.",
    "Write a Rust function that parses a CSV line with quoted fields.",
    "Summarize the causes and consequences of the 1929 stock market crash.",
    "Write a bash script that rotates log files older than seven days.",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clients", type=int)
    ap.add_argument("seconds", type=float)
    ap.add_argument("--base", default="http://127.0.0.1:8888")
    ap.add_argument("--model", default="DeepSeek-v4.1-Flash-EXL3")
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()
    key = os.environ.get("TF_API_KEY", "")
    end = time.monotonic() + args.seconds
    lock = threading.Lock()
    done: list[tuple[int, float]] = []          # (completion tokens, seconds) a request
    errors = [0]

    def client(c: int) -> None:
        n = 0
        while time.monotonic() < end:
            body = {"model": args.model, "max_tokens": args.max_tokens, "temperature": 0.7,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "messages": [{"role": "user", "content": f"{PROMPTS[(c + n) % len(PROMPTS)]} (variant {c}.{n})"}]}
            req = urllib.request.Request(f"{args.base}/v1/chat/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
            t = time.monotonic()
            try:
                with urllib.request.urlopen(req, timeout=600) as r:
                    out = json.loads(r.read())
                with lock:
                    done.append((out["usage"]["completion_tokens"], time.monotonic() - t))
            except Exception as e:  # noqa: BLE001 - a failed request is counted, not fatal
                with lock:
                    errors[0] += 1
                print(f"client {c}: {e}", flush=True)
            n += 1

    t0 = time.monotonic()
    threads = [threading.Thread(target=client, args=(c,)) for c in range(args.clients)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.monotonic() - t0
    tokens = sum(n for n, _ in done)
    mean = sum(n / s for n, s in done) / max(len(done), 1)
    print(f"{args.clients} clients: {tokens} tokens in {wall:.0f}s = {tokens / wall:.1f} tok/s aggregate, "
          f"{len(done)} requests, per-request mean {mean:.1f} tok/s, {errors[0]} errors", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Concurrent replies == solo runs through the server (token_sha), greedy and seeded, with aggregate tok/s."""
# usage: xpu_concurrent_receipt.py BASE_URL MODEL [--streams N] [--tokens N] [--rounds R]
import argparse
import json
import threading
import time
import urllib.request

PROMPTS = [
    "Write a Python function that parses an ISO 8601 date string and returns a datetime, with error handling and tests.",
    "Describe a quiet harbour town at dawn in about two hundred words.",
    "List the planets of the solar system in order, one per line, with one fact about each.",
    "Explain how a hash map handles collisions, with a short example in C.",
    "Write a haiku sequence about autumn rain, five haiku.",
    "Summarize the causes of the French Revolution in five bullet points.",
    "Repeat this sentence three times: the quick brown fox jumps over the lazy dog.",
    "Give a step-by-step recipe for a simple tomato soup.",
]
SAMPLINGS = [{"temperature": 0}, {"temperature": 0.8, "top_p": 0.95, "seed": 7}]


def ask(base, model, prompt, tokens, extra):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": tokens, "ignore_eos": True,
            "chat_template_kwargs": {"enable_thinking": False}, **extra}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = json.load(r)
    return out["tensorfold"]["token_sha"], out["usage"]["completion_tokens"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("model")
    ap.add_argument("--streams", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=192)
    ap.add_argument("--rounds", type=int, default=2)
    a = ap.parse_args()
    jobs = [(PROMPTS[i % len(PROMPTS)], SAMPLINGS[i % 2]) for i in range(a.streams)]
    solo = []
    t0 = time.time()
    for p, s in jobs:
        solo.append(ask(a.base, a.model, p, a.tokens, s)[0])
    solo_s = time.time() - t0
    bad = 0
    for rnd in range(a.rounds):
        got = [None] * len(jobs)
        toks = [0] * len(jobs)

        def run(i):
            got[i], toks[i] = ask(a.base, a.model, jobs[i][0], a.tokens, jobs[i][1])

        t0 = time.time()
        th = [threading.Thread(target=run, args=(i,)) for i in range(len(jobs))]
        for t in th:
            t.start()
        for t in th:
            t.join()
        wall = time.time() - t0
        same = sum(g == s for g, s in zip(got, solo))
        bad += len(jobs) - same
        print(f"round {rnd + 1}: {same}/{len(jobs)} concurrent replies == solo; {sum(toks)} tokens in {wall:.1f} s = "
              f"{sum(toks) / wall:.1f} tok/s aggregate (solo, one after another: {sum(toks) / solo_s:.1f})", flush=True)
    print("CONCURRENT == SOLO:", "PASS" if bad == 0 else f"FAIL ({bad})")


if __name__ == "__main__":
    main()

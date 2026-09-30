"""Collect teacher-forced prompt logprobs from a running vLLM DeepSeek-V4.1 server: goldens for the reference forward.

    python tools/dsv41_golden.py http://127.0.0.1:8888 notes/dsv41/golden.json

Each prompt is tokenized by the server (so ids match its tokenizer), then sent back as token ids with
``prompt_logprobs=5`` and one greedy completion token. The file keeps ids, per-position top-5 and the
logprob of the actual next token.
"""

from __future__ import annotations

import json
import sys
import urllib.request

PROMPTS = [
    "The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n    if n < 2:\n        return n\n    return",
    "In 1905, Albert Einstein published four papers that changed physics. The first explained the photoelectric "
    "effect, the second Brownian motion, the third special relativity, and the fourth showed that mass and energy "
    "are equivalent, expressed by the famous equation",
    "Q: A train leaves at 3 pm travelling 60 km/h. Another leaves the same station at 4 pm travelling 90 km/h on the "
    "same track. At what time does the second train catch up?\nA: Let t be hours after 3 pm. 60t = 90(t - 1), so",
    "Die Donau ist der zweitlängste Fluss Europas. Sie entspringt im Schwarzwald und mündet ins",
    "Once upon a time, in a small village at the edge of a vast forest, there lived an old clockmaker named Elias. "
    "Every morning he opened his shop at dawn, wound every clock on the wall, and listened. One day, one of the "
    "clocks did not tick. He opened its case and found, instead of gears, a tiny folded note that read:",
]


def post(url: str, body: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def main() -> None:
    base, out = sys.argv[1].rstrip("/"), sys.argv[2]
    model = json.loads(urllib.request.urlopen(base + "/v1/models").read())["data"][0]["id"]
    goldens = []
    for text in PROMPTS:
        ids = post(base + "/tokenize", {"model": model, "prompt": text, "add_special_tokens": True})["tokens"]
        r = post(base + "/v1/completions", {"model": model, "prompt": ids, "max_tokens": 1, "temperature": 0,
                                            "prompt_logprobs": 5, "logprobs": 5})
        choice = r["choices"][0]
        rows = choice.get("prompt_logprobs") or []
        top, actual = [], []
        for i, row in enumerate(rows):
            if row is None:
                top.append(None)
                actual.append(None)
                continue
            entries = {int(k): v for k, v in row.items()}
            top.append(sorted(((tid, e["logprob"]) for tid, e in entries.items()), key=lambda t: -t[1])[:5])
            actual.append(entries.get(ids[i], {}).get("logprob"))
        last = choice["logprobs"]["top_logprobs"][0] if choice.get("logprobs") else {}
        goldens.append({"text": text, "ids": ids, "prompt_top5": top, "prompt_actual": actual,
                        "next_text": choice["text"], "next_top5": last})
        print(f"{len(ids):4d} tokens -> {choice['text']!r}", flush=True)
    with open(out, "w") as f:
        json.dump({"model": model, "goldens": goldens}, f)


if __name__ == "__main__":
    main()

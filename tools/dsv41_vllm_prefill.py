"""vLLM prefill rate on the long document: python dsv41_vllm_prefill.py URL doc.txt 2048,8192,16000

Each prompt starts with a fresh random token so the prefix cache cannot serve it; max_tokens 1, time to the
response (prefill + one step). VLLM_API_KEY from the environment.
"""

import json
import os
import random
import sys
import time
import urllib.request
from pathlib import Path

AUTH = {"Authorization": "Bearer " + os.environ["VLLM_API_KEY"]} if os.environ.get("VLLM_API_KEY") else {}


def post(url: str, body: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", **AUTH})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.loads(r.read())


def main() -> None:
    base, doc, lengths = sys.argv[1].rstrip("/"), Path(sys.argv[2]).read_text(), sys.argv[3]
    model = json.loads(urllib.request.urlopen(urllib.request.Request(base + "/v1/models", headers=AUTH)).read())
    model = model["data"][0]["id"]
    ids = post(base + "/tokenize", {"model": model, "prompt": doc, "add_special_tokens": True})["tokens"]
    post(base + "/v1/completions", {"model": model, "prompt": [random.randrange(1000, 100000)] + ids[:512],
                                    "max_tokens": 1})
    for n in (int(x) for x in lengths.split(",")):
        rates = []
        for _ in range(2):
            prompt = [random.randrange(1000, 100000)] + ids[:n - 1]
            t = time.perf_counter()
            post(base + "/v1/completions", {"model": model, "prompt": prompt, "max_tokens": 1, "temperature": 0})
            rates.append(n / (time.perf_counter() - t))
        print(f"{n} tokens: {max(rates):.0f} tok/s (runs {', '.join(f'{r:.0f}' for r in rates)})", flush=True)


if __name__ == "__main__":
    main()

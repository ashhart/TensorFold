"""vLLM's DSpark acceptance and speed on given prompts, from its /metrics counters (run inside the vLLM container).

    python dsv41_vllm_accept.py http://127.0.0.1:8888 prompts.json out.json

prompts.json: {"prompts": [{"name": ..., "ids": [...]} | {"name": ..., "messages": [...]}], "max_tokens": N}.
Chat prompts are tokenized by the server (chat template, thinking on) so another engine can replay the same ids.
"""

from __future__ import annotations

import json
from pathlib import Path
import os
import re
import sys
import time
import urllib.request

AUTH = {"Authorization": "Bearer " + os.environ["VLLM_API_KEY"]} if os.environ.get("VLLM_API_KEY") else {}


def call(url: str, body: dict | None = None) -> dict | str:
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **AUTH})
    with urllib.request.urlopen(req, timeout=1200) as r:
        raw = r.read().decode()
    return json.loads(raw) if raw.lstrip().startswith("{") else raw


def counters(base: str) -> tuple[float, float]:
    text = call(base + "/metrics")

    def get(name: str) -> float:
        return sum(float(v) for v in re.findall(rf"^{name}{{[^}}]*}} ([0-9.e+]+)$", text, re.MULTILINE))

    return get("vllm:spec_decode_num_drafts_total"), get("vllm:spec_decode_num_accepted_tokens_total")


def main() -> None:
    base, spec, out = sys.argv[1].rstrip("/"), json.loads(Path(sys.argv[2]).read_text()), sys.argv[3]
    model = call(base + "/v1/models")["data"][0]["id"]
    results = []
    for p in spec["prompts"]:
        ids = p.get("ids") or call(base + "/tokenize", {"model": model, "messages": p["messages"],
                                                        "add_generation_prompt": True})["tokens"]
        d0, a0 = counters(base)
        t = time.perf_counter()
        r = call(base + "/v1/completions", {"model": model, "prompt": ids, "max_tokens": spec["max_tokens"],
                                            "temperature": 0, "return_token_ids": True})
        dt = time.perf_counter() - t
        d1, a1 = counters(base)
        ch = r["choices"][0]
        n = r["usage"]["completion_tokens"]
        res = {"name": p["name"], "ids": ids, "out_ids": ch.get("token_ids"), "text": ch["text"], "tokens": n,
               "rounds": d1 - d0, "accepted_per_round": (a1 - a0) / max(d1 - d0, 1), "tok_s": n / dt}
        results.append(res)
        print(f"{p['name']}: {n} tokens in {dt:.1f} s ({n / dt:.1f} tok/s incl. prefill), "
              f"{res['accepted_per_round']:.2f} accepted a round over {d1 - d0:.0f} rounds", flush=True)
    Path(out).write_text(json.dumps({"results": results}))


if __name__ == "__main__":
    main()

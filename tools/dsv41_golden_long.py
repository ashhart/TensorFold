"""Long-context goldens from a running vLLM DeepSeek-V4.1 server: prompt logprobs of token prefixes of one document.

    python dsv41_golden_long.py http://127.0.0.1:8888 doc.txt out.json 1024,2048,4096,8192

Per prefix: the ids, and per position the actual token's logprob and the top-1 id (prompt_logprobs=1).
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

AUTH = {"Authorization": "Bearer " + os.environ["VLLM_API_KEY"]} if os.environ.get("VLLM_API_KEY") else {}


def post(url: str, body: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", **AUTH})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.loads(r.read())


def main() -> None:
    base, doc, out, lengths = sys.argv[1].rstrip("/"), Path(sys.argv[2]).read_text(), sys.argv[3], sys.argv[4]
    model = json.loads(urllib.request.urlopen(urllib.request.Request(base + "/v1/models", headers=AUTH)).read())
    model = model["data"][0]["id"]
    ids_all = post(base + "/tokenize", {"model": model, "prompt": doc, "add_special_tokens": True})["tokens"]
    goldens = []
    for n in (int(x) for x in lengths.split(",")):
        ids = ids_all[:n]
        r = post(base + "/v1/completions", {"model": model, "prompt": ids, "max_tokens": 1, "temperature": 0,
                                            "prompt_logprobs": 1})
        rows = r["choices"][0]["prompt_logprobs"]
        actual, top1 = [None], [None]
        for i in range(1, len(ids)):
            entries = {int(k): v for k, v in rows[i].items()}
            actual.append(entries.get(ids[i], {}).get("logprob"))
            top1.append(min(entries.items(), key=lambda kv: kv[1].get("rank", 99))[0])
        goldens.append({"ids": ids, "prompt_actual": actual, "prompt_top1": top1})
        print(f"{n} tokens: done", flush=True)
    Path(out).write_text(json.dumps({"model": model, "goldens": goldens}))


if __name__ == "__main__":
    main()

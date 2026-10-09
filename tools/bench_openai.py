"""Single-stream decode speed of an OpenAI-compatible server, measured from the stream.

Decode tok/s = (completion tokens - 1) / (last token time - first token time), the same span the
CUDA engine's bench times (the first sampled token excluded). Standard library only, so it runs on
a bare host.

  python3 tools/bench_openai.py http://127.0.0.1:8080 MODEL --tokens 64 --reps 5 --output out.json
"""

import argparse
import json
import statistics
import sys
import time
import urllib.request

from bench_evidence import Evidence

if sys.version_info < (3, 11):
    sys.exit(f"Python 3.11+ is required (this is {sys.version.split()[0]}); "
             "TensorFold's tools use PEP 604 unions. Try: python3.12 tools/bench_openai.py ...")

PROMPTS = [
    {"name": "fibonacci-raw", "kind": "completion",
     "prompt": "Write a short Python function that computes the Fibonacci sequence and explain it."},
    {"name": "gpu-chat-no-think", "kind": "chat",
     "prompt": "Explain how matrix multiplication uses a GPU in plain English, then give a small numerical example."},
]


def stream(base: str, model: str, item: dict, tokens: int, temperature: float, seed: int | None) -> dict:
    body = {"model": model, "max_tokens": tokens, "temperature": temperature, "stream": True,
            "stream_options": {"include_usage": True}, "ignore_eos": True}
    if seed is not None:
        body["seed"] = seed
    if temperature > 0:
        body.update(top_k=20, top_p=0.95)
    if item["kind"] == "chat":
        url = base + "/v1/chat/completions"
        body["messages"] = [{"role": "user", "content": item["prompt"]}]
        body["chat_template_kwargs"] = {"enable_thinking": False}
    else:
        url = base + "/v1/completions"
        body["prompt"] = item["prompt"]
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    first = last = None
    usage = None
    text = []
    evidence = Evidence()
    runtime = {}
    complete = False
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if line == "data: [DONE]":
                complete = True
                break
            if not line.startswith("data:"):
                continue
            chunk = json.loads(line[5:])
            if "error" in chunk:
                raise RuntimeError("server_stream_error")
            runtime = chunk.get("tensorfold") or runtime
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                evidence.add(choice)
                delta = choice.get("delta") or {}
                content = choice.get("text") or delta.get("content") or ""
                piece = content or delta.get("reasoning_content") or ""
                if piece:
                    now = time.perf_counter()
                    first = first if first is not None else now
                    last = now
                    text.append(content)
    observed = evidence.finish(usage, runtime, complete)
    n = observed["tokens"]
    span = last - first if first is not None else None
    return {"ttft_s": first - start if complete and first is not None else None, "decode_s": span, **observed,
            "decode_tps": (n - 1) / span if complete and n is not None and n > 1 and span else None,
            "text": "".join(text)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("base")
    p.add_argument("model")
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--temperatures", default="1.0,0")
    p.add_argument("--label", default="")
    p.add_argument("--output")
    p.add_argument("--seed-from-prompt", action="store_true",
                   help="send no seed: cuda_server then seeds from the prompt, as the engine benches do, "
                        "so the reply equals the bench's and the timings compare directly")
    args = p.parse_args()
    results = []
    for temp in [float(t) for t in args.temperatures.split(",")]:
        for item in PROMPTS:
            seeds = [None] * args.reps if args.seed_from_prompt else [1234 + i for i in range(args.reps)]
            stream(args.base, args.model, item, args.tokens, temp, seeds[0])          # warm-up
            runs = [stream(args.base, args.model, item, args.tokens, temp, seed) for seed in seeds]
            tps = [r["decode_tps"] for r in runs if r["decode_tps"]]
            ttfts = [r["ttft_s"] for r in runs if r["ttft_s"] is not None]
            row = {"label": args.label, "prompt": item["name"], "temperature": temp, "tokens": args.tokens,
                   "decode_tps_median": statistics.median(tps) if tps else None, "decode_tps_all": [round(x, 2) for x in tps],
                   "ttft_s_median": statistics.median(ttfts) if ttfts else None,
                   "evidence": [{k: r[k] for k in ("prompt_tokens", "cached_tokens", "tokens", "cache_state",
                                "complete", "token_sha", "output_sha256")} for r in runs],
                   "sample": runs[0]["text"][:160]}
            print(json.dumps({k: row[k] for k in ("label", "prompt", "temperature", "decode_tps_median",
                                                   "decode_tps_all", "ttft_s_median")}), flush=True)
            results.append(row)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()

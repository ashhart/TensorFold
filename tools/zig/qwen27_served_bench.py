"""Measure one owned drafted server on fixed token IDs with prefix caching disabled."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import socket
import statistics
import subprocess
import time
from urllib.request import Request, urlopen


def request(base, ids, count, wanted=None):
    body = {"model": "qwen27", "prompt": ids, "temperature": 0, "draft": True,
            "max_tokens": count, "ignore_eos": True, "stream": True,
            "stream_options": {"include_usage": True}}
    call = Request(base + "/v1/completions", data=json.dumps(body).encode(),
                   headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    first, finished, usage, runtime, speculative = None, None, None, None, None
    done = False
    with urlopen(call, timeout=600) as response:
        for raw in response:
            line = raw.decode().strip()
            if line == "data: [DONE]":
                done = True
                continue
            if not line.startswith("data:"):
                continue
            event = json.loads(line[5:])
            if event.get("error"):
                raise RuntimeError("server returned an SSE error")
            now = time.perf_counter()
            for choice in event.get("choices", []):
                if choice.get("text") or (choice.get("delta") or {}).get("content"):
                    first = first if first is not None else now
            if event.get("usage"):
                usage, finished = event["usage"], now
            runtime = event.get("tensorfold", runtime)
            speculative = event.get("speculative", speculative)
    if not done or usage is None or runtime is None or finished is None or first is None:
        raise RuntimeError("incomplete served timing receipt")
    if usage["prompt_tokens"] != len(ids) or usage["completion_tokens"] != count:
        raise RuntimeError("served token lengths differ")
    if usage.get("prompt_tokens_details", {}).get("cached_tokens") != 0:
        raise RuntimeError("cold-prefix request reused cached tokens")
    if not runtime["drafts"] or speculative is None:
        raise RuntimeError("drafted server mode not recorded")
    if count > 1 and (speculative["rounds"] <= 0 or speculative["drafted"] <= 0):
        raise RuntimeError("draft verification was not exercised")
    if wanted is not None:
        digest = hashlib.sha256(",".join(map(str, wanted)).encode()).hexdigest()[:12]
        if runtime["token_sha"] != digest:
            raise RuntimeError("drafted served tokens differ from the engine's plain reference")
    span = finished - first
    return {"prompt_tokens": len(ids), "completion_tokens": count,
            "first_content_s": first - start, "first_content_to_usage_s": span,
            "client_decode_tok_s": (count - 1) / span if count > 1 and span > 0 else None,
            "reported_decode_tok_s": runtime["tokens_per_second"],
            "prefill_s": runtime.get("prefill_seconds"), "runtime": runtime,
            "speculative": speculative, "cached_tokens": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("native", "python066"), required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--drafter", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18436)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--stage", choices=("all", "decode", "prefill"), default="all")
    args = parser.parse_args()
    inputs = json.loads(args.inputs.read_text())
    names = ("copy", "arithmetic", "story", "code")
    if not 1024 <= args.port <= 65535 or not 1 <= args.repeats <= 5:
        raise ValueError("invalid bounded benchmark options")
    if set(inputs["prompts"]) != set(names) or set(inputs["targets"]) != set(names):
        raise ValueError("expected four fixed prompts and references")
    for name in names:
        if not 1 <= len(inputs["prompts"][name]) <= 128 or len(inputs["targets"][name]) != 256:
            raise ValueError("invalid short prompt/reference length")
    if set(inputs["cold"]) != {"2048", "8192", "32768"}:
        raise ValueError("expected exact 2k/8k/32k cold prompts")
    for size, ids in inputs["cold"].items():
        if len(ids) != int(size):
            raise ValueError("cold prompt length differs")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    args.out.mkdir(parents=True, exist_ok=False)
    command = ([str(args.binary)] if args.engine == "native" else
               [str(args.binary), "-B", "-m", "tensorfold"])
    command += ["serve", str(args.model), "--drafter", str(args.drafter), "--drafter-bits", "4",
                "--context", "65536", "--name", "qwen27", "--host", "127.0.0.1",
                "--port", str(args.port), "--no-thinking", "--parallel", "1",
                "--prompt-cache-gib", "0", "--snapshot-dir", "none", "--max-snapshots", "0",
                "--temperature", "0", "--no-update-check"]
    base, results = f"http://127.0.0.1:{args.port}", {"decode": {}, "cold_prefill": {}}
    def sample(key, ids, count, wanted=None):
        value = request(base, ids, count, wanted)
        with (args.out / f"{key}.json").open("x") as output:
            output.write(json.dumps(value, indent=2) + "\n")
        return value
    with (args.out / "server.log").open("x") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        print(f"START {args.engine} served benchmark owned PID {child.pid}", flush=True)
        try:
            deadline = time.monotonic() + 300
            while True:
                if child.poll() is not None:
                    raise RuntimeError("owned server exited during startup")
                try:
                    with urlopen(base + "/health", timeout=2) as response:
                        json.load(response)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("owned server startup deadline") from None
                    time.sleep(0.2)
            for name in names if args.stage != "prefill" else ():
                ids, target = inputs["prompts"][name], inputs["targets"][name]
                warm = sample(f"{name}-warmup", ids, 256, target)
                samples = [sample(f"{name}-{repeat}", ids, 256, target) for repeat in range(args.repeats)]
                results["decode"][name] = {"warmup": warm, "samples": samples,
                    "client_median_tok_s": statistics.median(s["client_decode_tok_s"] for s in samples),
                    "reported_median_tok_s": statistics.median(s["reported_decode_tok_s"] for s in samples)}
                print(json.dumps({"prompt": name, "samples": args.repeats,
                                  "client_median_tok_s": results["decode"][name]["client_median_tok_s"],
                                  "reported_median_tok_s": results["decode"][name]["reported_median_tok_s"]}), flush=True)
            for size, ids in inputs["cold"].items() if args.stage != "decode" else ():
                samples = [sample(f"cold-{size}-{repeat}", ids, 1) for repeat in range(args.repeats)]
                results["cold_prefill"][size] = {"samples": samples,
                    "ttft_median_s": statistics.median(s["first_content_s"] for s in samples)}
                print(f"PASS cold {size} {args.repeats} fresh-prefix samples", flush=True)
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=10)
            print(f"END {args.engine} served benchmark owned PID {child.pid} rc {child.returncode}", flush=True)
    result = {"engine": args.engine, "server_command": command, "repeats": args.repeats, "stage": args.stage,
              "inputs_sha256": hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
              "cold_definition": "fresh KV and prefix cache disabled; startup compilation may be warm",
              "client_decode_span": "first nonempty content event to complete usage event; first token excluded",
              **results}
    with (args.out / "bench.json").open("x") as out:
        out.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

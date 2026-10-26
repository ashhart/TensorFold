"""Decode speed of an OpenAI-compatible server under concurrent requests (bench_openai's prompts and settings).

Default: four cells (code and chat prompts, sampled and greedy). In each cell N requests of the same prompt start
at once, stream i with seed ``seed + i`` (greedy: all ``seed``), each ``--tokens`` long with ignore_eos. Per stream: decode tok/s =
(tokens - 1) / (last token - first token) and time to first token. Aggregate tok/s = the streams' tokens after
their first / (last token of any stream - first token of any stream). Medians over ``--reps``.

``--mixed``: request i takes prompt i % 2 and seed seed + i // 2; also reports the steady aggregate, over the time
every stream is live (tokens split by the characters each stream delivered inside that window).
``--alone``: runs every request alone first and checks each concurrent reply's token-id hash (``token_sha256``)
against it; ``--serial`` checks the alone runs against ``"draft": false``. ``--no-seed``: sampled requests carry
no seed (mlx_lm's server batches only those), for measuring a standard; no hash checks then. Run a standard with its
prompt cache off (mlx_lm: ``--prompt-cache-size 0``): a cached reply can arrive as one text piece. Tokens come from
the reply's usage. A request that fails to connect or errors counts in ``failed``; one whose text came in fewer than
``MIN_PIECES`` pieces counts in ``unmeasured``; neither enters the rates. ``--stagger-ms`` spaces the starts (a
server with a small listen backlog resets simultaneous connections; those resets are reported, not hidden).
Each request also carries bench_metrics' figures (prefill tok/s, ITL, round gaps; docs/cache-benchmarks.md defines
them); each rep pools them over its streams, and each cell gives medians ("request" and "pooled").
Standard library only.

  python3 tools/bench_concurrent.py http://127.0.0.1:8080 MODEL --levels 1,2,4,8 --alone --output out.json
"""

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.request

from bench_evidence import Evidence, token_equal
from bench_auth import authorize, read_api_key, redact
import bench_metrics
import bench_timing

if sys.version_info < (3, 11):
    sys.exit(f"Python 3.11+ is required (this is {sys.version.split()[0]}); "
             "TensorFold's tools use PEP 604 unions. Try: python3.12 tools/bench_concurrent.py ...")

MIN_PIECES = 4           # text pieces a reply needs before its span is timed

PROMPTS = [
    {"name": "code", "kind": "completion",
     "prompt": "Write a short Python function that computes the Fibonacci sequence and explain it."},
    {"name": "chat", "kind": "chat",
     "prompt": "Explain how matrix multiplication uses a GPU in plain English, then give a small numerical example."},
]


def stream(base: str, model: str, item: dict, tokens: int, temperature: float, seed: int | None,
           draft: bool = True, gate: threading.Barrier | None = None, *, api_key: str | None = None) -> dict:
    try:
        return _stream(base, model, item, tokens, temperature, seed, draft, gate, api_key)
    except Exception as exc:  # noqa: BLE001 - a failed request is reported, not fatal
        if gate is not None:
            gate.abort()
        return failed_run(item, seed, type(exc).__name__)


def failed_run(item: dict, seed: int | None, error: str) -> dict:
    return {"prompt": item["name"], "seed": seed, "sent": None, "first": None, "last": None, "tokens": None,
            "pieces": [], "ttft_s": None, "decode_tps": None, "token_sha": None, "token_sha256": None,
            "completed": None, "error": error, "complete": False, "prompt_tokens": None,
            "cached_tokens": None, "cache_state": "unverified", "output_sha256": None}


def _stream(base: str, model: str, item: dict, tokens: int, temperature: float, seed: int | None, draft: bool,
            gate: threading.Barrier | None, api_key: str | None = None) -> dict:
    body = {"model": model, "max_tokens": tokens, "temperature": temperature, "stream": True,
            "stream_options": {"include_usage": True}, "ignore_eos": True}
    if seed is not None:
        body["seed"] = seed
    if not draft:
        body["draft"] = False
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
    authorize(req, api_key)
    if gate is not None:
        gate.wait(timeout=30)
    sent = time.perf_counter()
    pieces: list[tuple[float, int]] = []           # (arrival, characters) of every text chunk
    usage: dict = {}
    runtime: dict = {}
    evidence = Evidence()
    complete = False
    with urllib.request.urlopen(req, timeout=1800) as resp:
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
            usage = chunk.get("usage") or usage
            runtime = chunk.get("tensorfold") or runtime
            for choice in chunk.get("choices", []):
                evidence.add(choice)
                delta = choice.get("delta") or {}
                # reasoning counts: its tokens are in completion_tokens, so the clock starts with them
                piece = choice.get("text") or delta.get("content") or delta.get("reasoning_content") or ""
                if piece:
                    pieces.append((time.perf_counter(), len(piece)))
    completed = time.perf_counter()
    observed = evidence.finish(usage, runtime, complete)
    n = observed["tokens"]
    out = {"prompt": item["name"], "seed": seed, "sent": sent, "first": None, "last": None, "tokens": n,
           "pieces": pieces, "completed": completed, "ttft_s": None, "decode_tps": None, **observed}
    if not complete:
        out["error"] = "incomplete_stream"
    if pieces:
        first, last = pieces[0][0], pieces[-1][0]
        out.update(first=first, last=last, ttft_s=first - sent if complete else None)
        if complete and n is not None and n > 1 and last > first and len(pieces) >= MIN_PIECES:
            out["decode_tps"] = (n - 1) / (last - first)
        else:
            out["unmeasured"] = True
    else:
        out["unmeasured"] = True
    return {**out, **bench_timing.request(out), **bench_metrics.request(out)}


def together(base: str, model: str, specs: list[tuple[dict, int | None]], tokens: int, temperature: float,
             stagger_ms: float = 0.0, *, api_key: str | None = None) -> list[dict]:
    if not bench_timing.timestamp(stagger_ms):
        raise ValueError("stagger_ms must be finite and nonnegative")
    if not specs:
        return []
    gate = threading.Barrier(len(specs))
    out = [failed_run(item, seed, "worker_incomplete") for item, seed in specs]

    def run(i: int) -> None:
        item, seed = specs[i]
        try:
            if stagger_ms:
                gate.wait(timeout=30)
                time.sleep(i * stagger_ms / 1e3)
                out[i] = stream(base, model, item, tokens, temperature, seed, api_key=api_key)
            else:
                out[i] = stream(base, model, item, tokens, temperature, seed, gate=gate, api_key=api_key)
        except Exception as exc:  # noqa: BLE001 - retain failures outside the HTTP request too
            gate.abort()
            out[i] = failed_run(item, seed, type(exc).__name__)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(specs))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return out


def aggregates(runs: list[dict]) -> dict:
    done = [r for r in runs if r["decode_tps"] is not None]
    out: dict = {"failed": sum(1 for r in runs if r.get("error")), "unmeasured": sum(1 for r in runs if r.get("unmeasured")),
                 "aggregate_tps": 0.0, **bench_timing.workload(runs)}
    if not done:
        return out
    first = min(r["first"] for r in done)
    last = max(r["last"] for r in done)
    if last > first:
        out["aggregate_tps"] = sum(r["tokens"] - 1 for r in done) / (last - first)
    lo, hi = max(r["first"] for r in done), min(r["last"] for r in done)       # every stream live
    if len(done) > 1 and hi > lo:
        share = 0.0
        for r in done:
            chars = sum(c for _, c in r["pieces"]) or 1
            share += r["tokens"] * sum(c for t, c in r["pieces"] if lo <= t <= hi) / chars
        out["steady_tps"] = share / (hi - lo)
    return out


METRICS = ("prefill_tps", "itl_mean_ms", "itl_p50_ms", "itl_p95_ms", "round_gap_p50_ms", "round_gap_p95_ms",
           "tokens_per_chunk")
POOLED = ("aggregate_prefill_tps", "itl_mean_ms", "itl_p50_ms", "itl_p95_ms", "round_gap_p50_ms", "round_gap_p95_ms",
          "tokens_per_chunk")


def median(values) -> float | None:
    """The median of the values that are not None, rounded to 3 places; None when there are none."""
    got = [v for v in values if v is not None]
    return round(statistics.median(got), 3) if got else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("base")
    p.add_argument("model")
    p.add_argument("--levels", default="4", help="concurrent requests, a comma list")
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--temperatures", default="1.0,0")
    p.add_argument("--mixed", action="store_true", help="request i takes prompt i %% 2 (sampled only)")
    p.add_argument("--alone", action="store_true",
                   help="check each reply's full token_sha256 against its request alone")
    p.add_argument("--serial", action="store_true", help="check the alone runs against \"draft\": false")
    p.add_argument("--no-seed", action="store_true", help="sampled requests without a seed (standards; no checks)")
    p.add_argument("--stagger-ms", type=float, default=0.0, help="start request i this many ms after request i - 1")
    p.add_argument("--label", default="")
    p.add_argument("--output")
    p.add_argument("--api-key-env", metavar="NAME", help="read the bearer API key from this environment variable")
    args = p.parse_args()
    if not bench_timing.timestamp(args.stagger_ms):
        p.error("--stagger-ms must be finite and nonnegative")
    api_key = read_api_key(p, args.api_key_env)
    if args.no_seed:
        args.alone = args.serial = False
    levels = [int(x) for x in args.levels.split(",")]
    temps = [1.0] if args.mixed else [float(t) for t in args.temperatures.split(",")]
    cells = [None] if args.mixed else PROMPTS
    report: dict = {"label": args.label, "tokens": args.tokens, "mixed": args.mixed, "cells": []}
    for temp in temps:
        for item in cells:
            specs: list[tuple[dict, int | None]]
            if args.mixed or item is None:
                specs = [(PROMPTS[i % 2], args.seed + i // 2) for i in range(max(levels))]
            else:
                specs = [(item, args.seed + i if temp > 0 else args.seed) for i in range(max(levels))]
            if args.no_seed:
                specs = [(spec[0], None) for spec in specs]
            together(args.base, args.model, specs[:1], 16, temp, api_key=api_key)  # warm-up
            alone: dict[tuple[str, int | None], dict] = {}
            if args.alone or args.serial:
                for spec in specs:
                    key = (spec[0]["name"], spec[1])
                    if key not in alone:
                        alone[key] = stream(args.base, args.model, spec[0], args.tokens, temp, spec[1], api_key=api_key)
            if args.serial:
                for key, ref in alone.items():
                    serial = stream(args.base, args.model, next(q for q in PROMPTS if q["name"] == key[0]),
                                    args.tokens, temp, key[1], draft=False, api_key=api_key)
                    ref["serial_equal"] = token_equal(serial, ref)
                    ref["serial_run"] = serial
            for n in levels:
                reps = []
                for _ in range(args.reps):
                    runs = together(args.base, args.model, specs[:n], args.tokens, temp, args.stagger_ms, api_key=api_key)
                    rep = aggregates(runs)
                    rep["per_stream_tps"] = [round(r["decode_tps"], 1) for r in runs if r["decode_tps"] is not None]
                    rep["ttft_s"] = [round(r["ttft_s"], 2) if r["ttft_s"] is not None else None for r in runs]
                    rep["errors"] = [r["error"] for r in runs if r.get("error")]
                    rep["requests"] = [{k: r.get(k) for k in ("prompt_tokens", "ttft_s", "decode_tps", *METRICS)}
                                       for r in runs]
                    rep["pooled"] = bench_metrics.pooled(runs)
                    rep["evidence"] = [{k: r.get(k) for k in ("prompt_tokens", "cached_tokens", "tokens",
                                       "cache_state", "complete", "token_sha", "token_sha256", "output_sha256")}
                                       for r in runs]
                    rep["client_intervals"] = [{"start": r["sent"], "first": r.get("first"),
                                                "last": r.get("last"), "end": r.get("completed"),
                                                **bench_timing.request(r)} for r in runs]
                    if alone:
                        # None: the request or its solo run failed (not an exactness result)
                        rep["equal_alone"] = [token_equal(alone[(r["prompt"], r["seed"])], r) for r in runs]
                        fields = ("tokens", "token_sha256", "output_sha256", "complete", "error")
                        rep["comparisons"] = []
                        for r in runs:
                            ref = alone[(r["prompt"], r["seed"])]
                            pair = {"concurrent": r, "alone": ref, "serial": ref.get("serial_run", {})}
                            rep["comparisons"].append({k: {f: v.get(f) for f in fields} for k, v in pair.items()})
                    reps.append(rep)
                cell = {"prompt": "mixed" if args.mixed or item is None else item["name"], "temperature": temp,
                        "streams": n, "failed": sum(r["failed"] for r in reps),
                        "unmeasured": sum(r["unmeasured"] for r in reps),
                        "aggregate_tps": round(statistics.median(r["aggregate_tps"] for r in reps), 1),
                        "per_stream_tps": round(statistics.median([v for r in reps for v in r["per_stream_tps"]] or [0.0]), 1),
                        "ttft_s_max": max((t for r in reps for t in r["ttft_s"] if t is not None), default=None)}
                # request: medians over every request of every rep; pooled: medians of the reps' pooled figures
                asked = [q for r in reps for q in r["requests"]]
                cell["request"] = {k: median(q.get(k) for q in asked) for k in ("ttft_s", "decode_tps", *METRICS)}
                cell["pooled"] = {k: median(r["pooled"].get(k) for r in reps) for k in POOLED}
                valid = all(r["timing_valid"] for r in reps)
                cell["timing_valid"] = valid
                for key in ("end_to_end_tps", "makespan_s"):
                    cell[key] = statistics.median(r[key] for r in reps) if valid else None
                if all("steady_tps" in r for r in reps):
                    cell["steady_tps"] = round(statistics.median(r["steady_tps"] for r in reps), 1)
                if alone:
                    checks = [v for r in reps for v in r["equal_alone"]]
                    cell["alone"] = {"equal": checks.count(True), "unequal": checks.count(False),
                                     "failed": checks.count(None)}
                if args.serial:
                    checks = [v.get("serial_equal") for v in alone.values()]
                    cell["serial"] = {"equal": checks.count(True), "unequal": checks.count(False),
                                      "failed": checks.count(None)}
                errors = sorted({e for r in reps for e in r["errors"]})
                if errors:
                    cell["errors"] = errors[:3]
                print(json.dumps(redact({"label": args.label, **cell}, api_key)), flush=True)
                report["cells"].append({**cell, "reps": reps})
    if args.output:
        with open(args.output, "w") as f:
            json.dump(redact(report, api_key), f, indent=1)


if __name__ == "__main__":
    main()

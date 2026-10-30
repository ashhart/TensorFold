"""Concurrent two-request HTTP bench (repeated-ID prompts) used for the EXL3 ABBA runs: PORT IDS.json OUT.json LENGTH DRAFT."""
import concurrent.futures
import json
import statistics
import sys
import time
import urllib.request


def one(port, prompt, tokens=128, draft=False):
    body = {
        "prompt": prompt,
        "max_tokens": tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "draft": draft,
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first = last = None
    usage = {}
    runtime = {}
    text_events = 0
    complete = False
    with urllib.request.urlopen(request, timeout=1800) as response:
        for raw in response:
            line = raw.decode().strip()
            if line == "data: [DONE]":
                complete = True
                break
            if not line.startswith("data:"):
                continue
            event = json.loads(line[5:])
            usage = event.get("usage") or usage
            runtime = event.get("tensorfold") or runtime
            for choice in event.get("choices", []):
                piece = choice.get("text") or (choice.get("delta") or {}).get("content") or ""
                if piece:
                    now = time.perf_counter()
                    first = first or now
                    last = now
                    text_events += 1
    generated = int(usage.get("completion_tokens", 0))
    span = (last - first) if first is not None and last is not None else 0.0
    return {
        "complete": complete,
        "prompt_tokens": usage.get("prompt_tokens"),
        "tokens": generated,
        "ttft_s": (first - started) if first is not None else None,
        "first_token_at": first,
        "last_token_at": last,
        "prefill_s": runtime.get("prefill_seconds"),
        "decode_tps_server": runtime.get("tokens_per_second"),
        "decode_tps_client": ((generated - 1) / span) if generated > 1 and span > 0 else None,
        "itl_ms_client": (1000 * span / (generated - 1)) if generated > 1 and span > 0 else None,
        "token_sha256": runtime.get("token_sha256"),
        "text_events": text_events,
    }


def batch(port, prompt, tokens, draft=False):
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: one(port, prompt, tokens, draft), range(2)))
    end = time.perf_counter()
    total = sum(row["tokens"] for row in results)
    first_times = [row["first_token_at"] for row in results if row["first_token_at"] is not None]
    last_times = [row["last_token_at"] for row in results if row["last_token_at"] is not None]
    decode_span = max(last_times) - min(first_times) if first_times and last_times else 0.0
    return {
        "requests": results,
        "batch_wall_s": end - start,
        "aggregate_decode_tps": total / decode_span if decode_span > 0 else None,
        "end_to_end_aggregate_tps": total / (end - start) if end > start else None,
        "mean_prefill_s": statistics.mean(row["prefill_s"] for row in results if row["prefill_s"] is not None),
    }


if __name__ == "__main__":
    port = int(sys.argv[1])
    with open(sys.argv[2], encoding="utf-8") as source:
        base = json.load(source)
    length = int(sys.argv[4]); draft = sys.argv[5] == "1"
    ids = (base * (length // len(base) + 1))[:length]
    report = {"prompt_tokens": len(ids), "tokens": 128, "drafts": draft, "batches": []}
    for phase in ("warmup-1", "measure-1", "measure-2"):
        result = batch(port, ids, int(__import__("os").environ.get("TOK", "128")), draft)
        report["batches"].append({"phase": phase, **result})
        print(json.dumps(report["batches"][-1]), flush=True)
    with open(sys.argv[3], "w", encoding="utf-8") as target:
        json.dump(report, target, indent=2)

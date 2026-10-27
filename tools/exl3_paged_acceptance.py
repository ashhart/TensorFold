#!/usr/bin/env python3
r"""HTTP acceptance receipts for two BF16-paged TP2 requests (standard library only).

Start with short snapshot/cache branches:
  python3 tools/exl3_paged_acceptance.py a.bin b.bin --context-length 4096 \
      --base-url http://127.0.0.1:18400 --model MODEL --completion-tokens 256 \
      --warm-prefix-tokens 3072 --output receipt.json

The server must already enable BF16 paging, TP2 and two lanes. Repeat at increasing bounded lengths after snapshot/cache checks pass;
200k x2 is a final later stage, never an automatic validation step.

Fixtures are little-endian u32 IDs. Exactly the first --context-length IDs are
sent; longer fixtures are explicitly recorded, shorter/misaligned files fail.
With --warm-prefix-tokens, warm that common prefix once before the concurrent
branches and require positive cached_tokens on each branch. Without warming,
--require-cache-reuse can qualify an externally warmed cache. No cache reset is
attempted. Existing cache entries can satisfy reuse: these receipts do not prove
that a particular snapshot was restored, immutable, or equivalent to a cold run.

Non-streaming responses must report exact prompt and completion counts, a final
finish_reason and a full token_sha256. HTTP overlap alone never proves server
lane residency, TP2, BF16 storage, paging or snapshot correctness.

Exit 0 qualifies the requested checks only; exit 1 fails them, exit 2 is bad input.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid


LIMITATION = "HTTP overlap alone does not prove two admitted, live server lanes or TP2 residency."
CORRELATION = [
    "Match client_request_id (X-Request-ID), server_response_id, response request ID and UTC intervals to logs.",
    "Record binary/commit identity and BF16 paging, TP2, parallel=2 and sequence-memory reservation settings.",
    "On BOTH ranks, correlate distinct sequence/lane IDs, prompt lengths, positions and overlapping live lifetimes.",
    "Correlate /metrics running-request samples with per-rank lane traces; queued HTTP requests are insufficient.",
    "Record pager misses/writebacks/rebases/overflow, per-rank VRAM peaks, host/pinned bytes and expert-pool lines.",
    "For cache reuse, record snapshot save/restore IDs, prefix lengths, admission/eviction bytes and source lanes.",
    "Verify snapshot branching against independent cold continuations; record MTP rollback if enabled.",
]


def count(value):
    return value if type(value) is int and value >= 0 else None


def reject_constant(value):
    raise ValueError(f"invalid JSON constant: {value}")


def stamp():
    return {"utc": datetime.now(timezone.utc).isoformat(), "monotonic_ns": time.monotonic_ns()}


def load_fixture(path, length):
    with Path(path).open("rb") as fixture:
        size = fixture.seek(0, 2)
        if size % 4:
            raise ValueError(f"{path}: fixture byte length must be divisible by four")
        if size // 4 < length:
            raise ValueError(f"{path}: {size // 4} tokens available, {length} required")
        fixture.seek(0)
        raw = fixture.read(length * 4)
    if len(raw) != length * 4:
        raise ValueError(f"{path}: fixture changed while reading")
    return list(struct.unpack(f"<{length}I", raw)), {
        "path": str(path), "source_tokens": size // 4, "submitted_tokens": length,
        "selected_prefix": size // 4 > length, "prompt_u32le_sha256": hashlib.sha256(raw).hexdigest(),
    }


def inspect_response(row, payload, expected_prompt, expected_completion, require_reuse):
    if not isinstance(payload, dict):
        row["errors"].append("response must be a JSON object")
        return
    row["server_response_id"] = payload.get("id")
    runtime = payload.get("tensorfold")
    row["tensorfold"] = runtime
    runtime = runtime if isinstance(runtime, dict) else {}
    if "error" in payload:
        row["errors"].append(f"server error: {json.dumps(payload['error'])}")
    usage = payload.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    row["usage"] = usage
    row["prompt_tokens"] = count(usage.get("prompt_tokens"))
    row["completion_tokens"] = count(usage.get("completion_tokens"))
    details = usage.get("prompt_tokens_details")
    cached = count(details.get("cached_tokens")) if isinstance(details, dict) else None
    if cached is not None and (row["prompt_tokens"] is None or cached > row["prompt_tokens"]):
        cached = None
    row["cached_tokens"] = cached
    row["cache_reuse_observed"] = cached is not None and cached > 0
    choices = payload.get("choices")
    choice = choices[0] if isinstance(choices, list) and len(choices) == 1 else None
    reason = choice.get("finish_reason") if isinstance(choice, dict) else None
    row["finish_reason"] = reason
    row["complete"] = (
        reason in ("stop", "length") and "error" not in payload
        and payload.get("complete", True) is True and runtime.get("complete", True) is True
        and payload.get("status", "completed") in ("completed", "complete")
    )
    if not row["complete"]:
        row["errors"].append("incomplete or invalid completion state")
    if row["prompt_tokens"] != expected_prompt:
        row["errors"].append(f"prompt count must be exactly {expected_prompt}, got {row['prompt_tokens']}")
    if row["completion_tokens"] != expected_completion:
        row["errors"].append(f"completion count must be exactly {expected_completion}, got {row['completion_tokens']}")
    digest = runtime.get("token_sha256")
    short = runtime.get("token_sha")
    valid_hash = isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) is not None
    if short is not None:
        valid_hash = (valid_hash and isinstance(short, str)
                      and re.fullmatch(r"(?:[0-9a-f]{12}|[0-9a-f]{64})", short) is not None
                      and digest.startswith(short))
    row["token_sha256"] = digest if valid_hash else None
    if not valid_hash:
        row["errors"].append("missing, invalid or inconsistent TensorFold token_sha256 evidence")
    if require_reuse and not row["cache_reuse_observed"]:
        row["errors"].append("required positive cache reuse evidence is absent")


def request_completion(endpoint, model, prompt, completion_tokens, timeout, draft,
                       run_id, label, require_reuse=False, barrier=None):
    row = {
        "label": label, "client_request_id": f"{run_id}-{label}",
        "submitted_prompt_tokens": len(prompt), "requested_completion_tokens": completion_tokens,
        "start": None, "end": None, "http_status": None, "server_response_id": None,
        "response_request_id": None, "prompt_tokens": None, "completion_tokens": None,
        "cached_tokens": None, "cache_reuse_observed": False, "complete": False,
        "token_sha256": None, "tensorfold": None, "errors": [],
    }
    try:
        body = json.dumps({"model": model, "prompt": prompt, "max_tokens": completion_tokens,
                           "temperature": 0, "seed": 1234, "ignore_eos": True,
                           "draft": draft, "stream": False, "n": 1}).encode("utf-8")
        request = urllib.request.Request(endpoint, body, {
            "Content-Type": "application/json", "Accept": "application/json",
            "X-Request-ID": row["client_request_id"],
        })
        if barrier is not None:
            barrier.wait(timeout=min(timeout, 30))
        row["start"] = stamp()
        with urllib.request.urlopen(request, timeout=timeout) as response:
            row["http_status"] = response.status
            row["response_request_id"] = response.headers.get("X-Request-ID")
            raw = response.read()
        row["end"] = stamp()
        if row["http_status"] != 200:
            row["errors"].append(f"unexpected HTTP status {row['http_status']}")
        inspect_response(row, json.loads(raw, parse_constant=reject_constant),
                         len(prompt), completion_tokens, require_reuse)
    except urllib.error.HTTPError as error:
        row["http_status"] = error.code
        row["errors"].append(f"HTTPError: {error.code} {error.reason}")
        error.close()
    except Exception as error:
        row["errors"].append(f"{type(error).__name__}: {error}")
    finally:
        if row["end"] is None:
            row["end"] = stamp()
    row["qualified"] = not row["errors"]
    return row


def overlap(rows):
    if any(row["start"] is None or row["end"] is None for row in rows):
        return {"seconds": 0, "start_utc": None, "end_utc": None, "observed": False}
    latest = max(rows, key=lambda row: row["start"]["monotonic_ns"])["start"]
    earliest = min(rows, key=lambda row: row["end"]["monotonic_ns"])["end"]
    seconds = max(0, earliest["monotonic_ns"] - latest["monotonic_ns"]) / 1e9
    return {"seconds": seconds, "start_utc": latest["utc"] if seconds > 0 else None,
            "end_utc": earliest["utc"] if seconds > 0 else None, "observed": seconds > 0}


def run(args):
    prompts, fixtures = [], []
    for path in (args.fixture_a, args.fixture_b):
        prompt, evidence = load_fixture(path, args.context_length)
        prompts.append(prompt)
        fixtures.append(evidence)
    if prompts[0] == prompts[1]:
        raise ValueError("fixtures must provide distinct token histories for the two requests")
    prefix_length = args.warm_prefix_tokens
    if prefix_length and prompts[0][:prefix_length] != prompts[1][:prefix_length]:
        raise ValueError("fixtures do not share the requested warm prefix")
    run_id = uuid.uuid4().hex
    endpoint = args.base_url.rstrip("/") + "/v1/completions"
    report = {
        "schema_version": 1, "run_id": run_id, "started": stamp(),
        "scope": "HTTP acceptance only; server configuration is externally verified",
        "limitation": LIMITATION, "server_lane_residency_proven": False,
        "snapshot_restore_proven": False, "correlation_evidence_needed": CORRELATION,
        "configuration": {"context_length": args.context_length, "completion_tokens": args.completion_tokens,
                          "warm_prefix_tokens": prefix_length, "draft": args.draft,
                          "require_cache_reuse": args.require_cache_reuse or bool(prefix_length)},
        "fixtures": fixtures, "warmup": None, "requests": [], "errors": [],
    }
    if prefix_length:
        report["warmup"] = request_completion(
            endpoint, args.model, prompts[0][:prefix_length], 1, args.timeout, args.draft, run_id, "warmup")
        if not report["warmup"]["qualified"]:
            report["errors"].append("shared-prefix warmup failed; concurrent phase was not submitted")
    if not report["errors"]:
        barrier = threading.Barrier(2)
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(
                request_completion, endpoint, args.model, prompt, args.completion_tokens,
                args.timeout, args.draft, run_id, f"branch-{index}",
                args.require_cache_reuse or bool(prefix_length), barrier,
            ) for index, prompt in enumerate(prompts)]
            report["requests"] = [future.result() for future in futures]
        if any(not row["qualified"] for row in report["requests"]):
            report["errors"].append("one or more concurrent requests failed qualification")
    report["http_overlap"] = overlap(report["requests"]) if report["requests"] else {
        "seconds": 0, "start_utc": None, "end_utc": None, "observed": False,
    }
    if not report["http_overlap"]["observed"]:
        report["errors"].append("two HTTP request intervals did not overlap")
    report["finished"] = stamp()
    report["qualified"] = not report["errors"]
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fixture_a", type=Path)
    parser.add_argument("fixture_b", type=Path)
    parser.add_argument("--base-url", required=True, help="origin, without /v1/completions")
    parser.add_argument("--model", required=True)
    parser.add_argument("--context-length", required=True, type=int, help="exact prompt tokens per request")
    parser.add_argument("--completion-tokens", type=int, default=256, help="exact reply tokens; ignore_eos is set")
    parser.add_argument("--warm-prefix-tokens", type=int, default=0)
    parser.add_argument("--require-cache-reuse", action="store_true")
    parser.add_argument("--draft", action="store_true")
    parser.add_argument("--timeout", type=float, default=7200, help="HTTP socket timeout in seconds")
    parser.add_argument("--output", type=Path, help="also write the JSON receipt to this path")
    args = parser.parse_args(argv)
    if args.context_length <= 0 or args.completion_tokens <= 0:
        parser.error("context length and completion tokens must be positive")
    if not 0 <= args.warm_prefix_tokens < args.context_length:
        parser.error("warm prefix must be zero or shorter than the exact context length")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be positive and finite")
    url = urllib.parse.urlsplit(args.base_url)
    if url.scheme not in ("http", "https") or not url.netloc or url.query or url.fragment or url.path not in ("", "/"):
        parser.error("base URL must be an http(s) origin with no path, query or fragment")
    print(LIMITATION, file=sys.stderr)
    try:
        report = run(args)
        code = 0 if report["qualified"] else 1
    except (OSError, ValueError) as error:
        report = {"schema_version": 1, "qualified": False, "errors": [f"{type(error).__name__}: {error}"],
                  "limitation": LIMITATION, "correlation_evidence_needed": CORRELATION}
        code = 2
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    print(encoded, end="")
    if args.output:
        try:
            args.output.write_text(encoded, encoding="utf-8")
        except OSError as error:
            print(f"cannot save receipt: {error}", file=sys.stderr)
            return 2
    return code


if __name__ == "__main__":
    sys.exit(main())

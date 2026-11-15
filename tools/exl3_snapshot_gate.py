#!/usr/bin/env python3
"""Strict two-lane serving gate for BF16 paged prompt-cache restore."""

import argparse
import concurrent.futures
import json
import re
import threading
import time
import urllib.request
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("--tokens", type=int, default=4099)
    parser.add_argument("--decode", type=int, default=32)
    parser.add_argument("--drafts", action="store_true")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--record-reference", action="store_true", help="record a cold baseline; use a fresh server with prompt cache disabled")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.tokens < 32 or args.decode < 1:
        parser.error("require at least 32 prompt tokens and one decode token")
    if args.record_reference and not args.output:
        parser.error("--record-reference requires --output")
    if args.record_reference and args.reference:
        parser.error("--record-reference and --reference are mutually exclusive")
    shared = " cache" * max(args.tokens, 1)
    history = [{"role": "user", "content": shared}, {"role": "assistant", "content": "I have reviewed the shared document."}]
    prompts = [history + [{"role": "user", "content": f"Original ending for lane {lane}."}] for lane in range(2)]
    reference = json.loads(args.reference.read_text()) if args.reference else None
    report = {"prompt_tokens": args.tokens, "decode_tokens": args.decode, "drafts": args.drafts, "phases": {}}
    if reference:
        for field in ("prompt_tokens", "decode_tokens", "drafts"):
            if reference[field] != report[field]:
                raise AssertionError(f"reference configuration mismatch: {field}")

    def pair(request_prompts):
        barrier = threading.Barrier(2)

        def ask(ids):
            started = time.perf_counter()
            body = {"messages": ids, "max_tokens": args.decode, "temperature": 0, "ignore_eos": True, "draft": args.drafts}
            request = urllib.request.Request(args.url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
            barrier.wait(timeout=10)
            with urllib.request.urlopen(request, timeout=900) as reply:
                response = json.load(reply)
            usage = response["usage"]
            if usage["prompt_tokens"] < args.tokens or usage["completion_tokens"] != args.decode:
                raise AssertionError(f"inexact token counts: {usage}")
            runtime = response["tensorfold"]
            digest = runtime.get("token_sha256")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise AssertionError("missing full output token SHA256")
            return {
                "sha": digest,
                "cached_tokens": usage["prompt_tokens_details"]["cached_tokens"],
                "prompt_tokens": usage["prompt_tokens"],
                "elapsed_seconds": time.perf_counter() - started,
                "prefill_seconds": runtime.get("prefill_seconds"),
                "tokens_per_second": runtime.get("tokens_per_second"),
            }

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            return list(pool.map(ask, request_prompts))

    branches = [history + [{"role": "user", "content": f"Edited ending for branch {lane}."}] for lane in range(2)]
    for phase, ids in (("cold", prompts), ("warm", prompts), ("branch", branches), ("rewind", prompts)):
        results = pair(ids)
        for lane, result in enumerate(results):
            if args.record_reference and result["cached_tokens"] != 0:
                raise AssertionError(f"reference phase {phase} lane {lane} was not cold: {result}")
            if not args.record_reference and phase != "cold" and not 0 < result["cached_tokens"] < result["prompt_tokens"]:
                raise AssertionError(f"{phase} lane {lane} did not restore a cached prefix: {result}")
            if phase in ("warm", "rewind") and result["sha"] != report["phases"]["cold"][lane]["sha"]:
                raise AssertionError(f"{phase} lane {lane} differs from its cold continuation")
            if reference and result["sha"] != reference["phases"][phase][lane]["sha"]:
                raise AssertionError(f"{phase} lane {lane} differs from reference continuation")
        report["phases"][phase] = results
        print(json.dumps({"phase": phase, "lanes": results}), flush=True)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.record_reference:
        print("PASS: recorded no-cache cold continuations for cold, warm, branch and rewind inputs")
    else:
        print("PASS: exact prompt/reply lengths, two submitted lanes, warm/branch/rewind cache hits and cold-baseline parity")


if __name__ == "__main__":
    main()

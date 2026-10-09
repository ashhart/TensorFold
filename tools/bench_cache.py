"""Measure first, repeated and concurrent requests with observed cache evidence; never reset server state."""
import argparse
import hashlib
import json
import sys

import bench_concurrent as bench
from bench_evidence import token_equal


FIELDS = ("prompt_tokens", "cached_tokens", "tokens", "cache_state", "complete", "token_sha", "output_sha256",
          "ttft_s", "decode_tps", "error", "unmeasured")


def receipt(run, phase, reference=None):
    """Export counters and hashes only, with no origin, model alias, raw prompt, output or server error text."""
    out = {"phase": phase, **{k: run[k] for k in FIELDS if k in run}}
    if reference is not None:
        out["equal_first_tokens"] = token_equal(reference, run)
        out["equal_first_output"] = (None if not run.get("output_sha256") or not reference.get("output_sha256")
                                     else run["output_sha256"] == reference["output_sha256"])
    return out


def qualify(rows, expected=None, require_cold=False, require_reuse=False):
    """An assertion needs affirmative usage evidence; unknown cache state cannot satisfy it."""
    if not rows or any(r.get("error") or not r.get("complete") or not r.get("tokens") for r in rows):
        return False
    if any(r.get("equal_first_tokens") is False or r.get("equal_first_output") is False for r in rows):
        return False
    if expected is not None and any(r.get("prompt_tokens") != expected for r in rows):
        return False
    if require_cold and rows[0]["cache_state"] != "cold":
        return False
    if require_reuse and any(r["cache_state"] != "reused" for r in rows if r["phase"] == "repeated"):
        return False
    return True


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("base")
    p.add_argument("model")
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--streams", type=int, default=2)
    p.add_argument("--prompt-file", help="JSON array of raw token IDs; use the checkpoint's own tokenizer")
    p.add_argument("--expected-prompt-tokens", type=int)
    p.add_argument("--require-cold", action="store_true", help="require the first request to report zero cached tokens")
    p.add_argument("--require-reuse", action="store_true", help="require repeated requests to report positive cached tokens")
    p.add_argument("--output")
    args = p.parse_args(argv)
    if min(args.tokens, args.repeats, args.streams) < 1 or args.streams > 512:
        p.error("tokens/repeats/streams must be positive; streams must not exceed 512")
    if args.expected_prompt_tokens is not None and args.expected_prompt_tokens < 1:
        p.error("expected prompt tokens must be positive")
    prompt = "A public benchmark fixture: count the following words.\n" + "alpha beta gamma delta\n" * 512
    if args.prompt_file:
        try:
            with open(args.prompt_file, encoding="utf-8") as f:
                prompt = json.load(f)
            if not isinstance(prompt, list) or not prompt or any(type(n) is not int or not 0 <= n <= 0xffffffff for n in prompt):
                raise ValueError("invalid_token_array")
        except (OSError, ValueError):
            p.error("prompt file must contain a nonempty JSON array of unsigned 32-bit token IDs")
    item = {"name": "fixture", "kind": "completion", "prompt": prompt}
    first = bench.stream(args.base.rstrip("/"), args.model, item, args.tokens, 0, 1234)
    rows = [receipt(first, "first")]
    for _ in range(args.repeats):
        rows.append(receipt(bench.stream(args.base.rstrip("/"), args.model, item, args.tokens, 0, 1234), "repeated", first))
    runs = bench.together(args.base.rstrip("/"), args.model, [(item, 1234)] * args.streams, args.tokens, 0)
    rows.extend(receipt(r, "concurrent", first) for r in runs)
    report = {"schema_version": 1, "fixture_sha256": hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
              "requested_reply_tokens": args.tokens, "temperature": 0, "seed": 1234, "rows": rows,
              "qualified": qualify(rows, args.expected_prompt_tokens, args.require_cold, args.require_reuse)}
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(encoded)
    print(encoded, end="", flush=True)
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Compare greedy served tokens for exact prompt row counts around prompt16's dispatch."""
import argparse
import json
import sys
import urllib.request

from bench_auth import authorize, read_api_key
from bench_evidence import token_equal
import bench_openai


def tokenize(base, text, api_key):
    body = json.dumps({"prompt": text, "add_special_tokens": False}).encode()
    request = urllib.request.Request(base.rstrip("/") + "/tokenize", body,
                                     {"Content-Type": "application/json"})
    authorize(request, api_key)
    with urllib.request.urlopen(request, timeout=600) as response:
        ids = json.load(response)["tokens"]
    if not isinstance(ids, list) or any(type(token) is not int or token < 0 for token in ids):
        raise ValueError("tokenizer returned invalid IDs")
    return ids


def summary(result):
    return {key: result.get(key) for key in (
        "prompt_tokens", "cached_tokens", "tokens", "cache_state", "complete",
        "token_sha", "token_sha256", "output_sha256")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("baseline", "candidate"))
    parser.add_argument("base", help="OpenAI-compatible server URL")
    parser.add_argument("model")
    parser.add_argument("--source", default="README.md", help="public text tokenized for exact row counts")
    parser.add_argument("--lengths", default="17,32,64,128")
    parser.add_argument("--tokens", type=int, default=16, help="greedy completion tokens")
    parser.add_argument("--baseline", help="baseline JSON receipt required for candidate")
    parser.add_argument("--output", required=True)
    parser.add_argument("--api-key-env", metavar="NAME")
    args = parser.parse_args()
    if args.tokens < 1:
        parser.error("--tokens must be positive")
    try:
        lengths = [int(value) for value in args.lengths.split(",")]
    except ValueError:
        parser.error("--lengths must be comma-separated positive integers")
    if not lengths or any(length < 1 for length in lengths) or len(set(lengths)) != len(lengths):
        parser.error("--lengths must contain distinct positive integers")
    if args.role == "candidate" and not args.baseline:
        parser.error("candidate requires --baseline")
    api_key = read_api_key(parser, args.api_key_env)
    with open(args.source, encoding="utf-8", errors="replace") as source_file:
        text = source_file.read()

    prior = None
    if args.role == "candidate":
        with open(args.baseline, encoding="utf-8") as receipt_file:
            prior = json.load(receipt_file)
        if prior.get("model") != args.model or prior.get("lengths") != lengths or prior.get("reply_tokens") != args.tokens:
            parser.error("candidate settings must match the baseline receipt")

    ids = tokenize(args.base, text, api_key)
    if len(ids) < max(lengths):
        parser.error(f"source tokenized to {len(ids)} IDs; needs at least {max(lengths)}")
    if prior is not None and ids[:max(lengths)] != prior.get("prompt_ids"):
        parser.error("baseline and candidate tokenizers produced different prompt IDs")
    prompt_ids = prior["prompt_ids"] if prior is not None else ids[:max(lengths)]

    rows = []
    exact = True
    for length in lengths:
        item = {"name": f"prompt-{length}", "kind": "completion", "prompt": prompt_ids[:length]}
        result = bench_openai.stream(args.base.rstrip("/"), args.model, item, args.tokens, 0, 1234, api_key=api_key)
        row = {"rows": length, **summary(result)}
        if row["prompt_tokens"] != length or row["cache_state"] != "cold" or not row["complete"]:
            parser.error(f"prompt-{length} did not complete cold with exactly {length} prompt tokens")
        if prior is not None:
            baseline_row = next((entry for entry in prior["rows"] if entry["rows"] == length), None)
            equal = token_equal(baseline_row or {}, result)
            row["token_equal"] = equal
            exact = exact and equal is True
        rows.append(row)

    report = {"schema_version": 1, "role": args.role, "model": args.model,
              "lengths": lengths, "reply_tokens": args.tokens, "temperature": 0,
              "seed": 1234, "prompt_ids": prompt_ids, "rows": rows,
              "all_token_ids_equal": exact if prior is not None else None}
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(report, output_file, indent=2)
        output_file.write("\n")
    print(json.dumps({"role": args.role, "rows": rows,
                      "all_token_ids_equal": report["all_token_ids_equal"]}), flush=True)
    return 0 if prior is None or exact else 1


if __name__ == "__main__":
    sys.exit(main())

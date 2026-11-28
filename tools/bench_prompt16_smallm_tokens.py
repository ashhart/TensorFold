#!/usr/bin/env python3
"""Compare greedy served tokens with independent exact-length prompts.

Run with the server's prefix cache disabled (TF_KV_SHARE=0), then run baseline,
candidate, baseline in that order for an A/B/A measurement. Use the same
checkpoint, engine, GPU and prompt source for all three receipts; any cache hit
is rejected, and each measured prompt begins with a distinct token.
"""
import argparse
import hashlib
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
        "ttft_s", "prefill_tps", "decode_tps", "decode_s", "prompt_ttft_tps",
        "itl_p50_ms", "itl_p95_ms", "tpot_ms", "token_sha", "token_sha256",
        "output_sha256", "tool_sha256")}


def prompt_text(source, trial, length, nonce):
    return f"{nonce:04x}-{trial}-{length}: {source}"


def make_prompts(base, source, lengths, repeats, api_key, prior=None):
    existing = {}
    if prior is not None:
        existing = {(p["trial"], p["rows"]): p for p in prior.get("prompts", [])}
    prompts = []
    first_tokens = set()
    for trial in range(repeats):
        for length in lengths:
            prior_prompt = existing.get((trial, length))
            nonce_start = prior_prompt["nonce"] if prior_prompt else 0
            chosen = None
            for nonce in range(nonce_start, nonce_start + (1 if prior_prompt else 100)):
                ids = tokenize(base, prompt_text(source, trial, length, nonce), api_key)
                if len(ids) < length:
                    continue
                selected = ids[:length]
                if prior_prompt is not None:
                    if selected != prior_prompt.get("prompt_ids"):
                        raise ValueError(f"tokenizer mismatch at trial {trial}, prompt-{length}")
                    chosen = {"trial": trial, "rows": length, "nonce": nonce, "prompt_ids": selected}
                    break
                # Unique first tokens prevent prefix-cache hits across the prompt set.
                if selected[0] not in first_tokens:
                    first_tokens.add(selected[0])
                    chosen = {"trial": trial, "rows": length, "nonce": nonce, "prompt_ids": selected}
                    break
            if chosen is None:
                raise ValueError(f"could not create an independent exact-length prompt for trial {trial}, prompt-{length}")
            prompts.append(chosen)
    if prior is not None and len(prompts) != len(prior.get("prompts", [])):
        raise ValueError("baseline receipt prompt set does not match requested lengths and repeats")
    return prompts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("baseline", "candidate"))
    parser.add_argument("base", help="OpenAI-compatible server URL")
    parser.add_argument("model")
    parser.add_argument("--source", default="README.md", help="public text tokenized into independent exact-length prompts")
    parser.add_argument("--lengths", default="17,32,64,128")
    parser.add_argument("--repeats", type=int, default=3, help="independent requests per prompt length")
    parser.add_argument("--tokens", type=int, default=16, help="greedy completion tokens")
    parser.add_argument("--baseline", help="baseline JSON receipt required for candidate")
    parser.add_argument("--checkpoint", required=True, help="checkpoint name or revision")
    parser.add_argument("--engine", required=True, help="engine build/revision")
    parser.add_argument("--gpu", required=True, help="GPU model and physical SM count")
    parser.add_argument("--dispatch", required=True, help="active prompt dispatch, e.g. classic or smallm")
    parser.add_argument("--output", required=True)
    parser.add_argument("--api-key-env", metavar="NAME")
    args = parser.parse_args()
    if args.tokens < 1 or args.repeats < 1:
        parser.error("--tokens and --repeats must be positive")
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
        source = source_file.read()
    source_sha256 = hashlib.sha256(source.encode()).hexdigest()

    prior = None
    if args.role == "candidate":
        with open(args.baseline, encoding="utf-8") as receipt_file:
            prior = json.load(receipt_file)
        expected = {"model": args.model, "checkpoint": args.checkpoint, "engine": args.engine,
                    "gpu": args.gpu, "source_sha256": source_sha256,
                    "lengths": lengths, "repeats": args.repeats, "reply_tokens": args.tokens}
        if any(prior.get(key) != value for key, value in expected.items()):
            parser.error("candidate settings must match the baseline receipt's model, checkpoint, engine, GPU, source and lengths")
        if prior.get("dispatch") == args.dispatch:
            parser.error("candidate must use a different dispatch from the baseline")

    prompts = make_prompts(args.base.rstrip("/"), source, lengths, args.repeats, api_key, prior)
    prior_rows = {}
    if prior is not None:
        prior_rows = {(row["trial"], row["rows"]): row for row in prior.get("rows", [])}

    rows = []
    exact = True
    for prompt in prompts:
        key = (prompt["trial"], prompt["rows"])
        item = {"name": f"trial-{prompt['trial']}-prompt-{prompt['rows']}",
                "kind": "completion", "prompt": prompt["prompt_ids"]}
        result = bench_openai.stream(args.base.rstrip("/"), args.model, item, args.tokens, 0, 1234,
                                     api_key=api_key)
        row = {"trial": prompt["trial"], "rows": prompt["rows"], **summary(result)}
        if row["prompt_tokens"] != prompt["rows"] or row["cached_tokens"] != 0 or row["cache_state"] != "cold" or not row["complete"]:
            parser.error(f"trial {prompt['trial']} prompt-{prompt['rows']} was not cold with exactly the requested tokens")
        if prior is not None:
            equal = token_equal(prior_rows.get(key, {}), result)
            row["token_equal"] = equal
            exact = exact and equal is True
        rows.append(row)

    report = {"schema_version": 2, "role": args.role, "model": args.model,
              "checkpoint": args.checkpoint, "engine": args.engine, "gpu": args.gpu,
              "dispatch": args.dispatch, "source_sha256": source_sha256,
              "lengths": lengths, "repeats": args.repeats, "reply_tokens": args.tokens,
              "temperature": 0, "seed": 1234, "prompts": prompts, "rows": rows,
              "all_token_ids_equal": exact if prior is not None else None}
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(report, output_file, indent=2)
        output_file.write("\n")
    print(json.dumps({"role": args.role, "checkpoint": args.checkpoint,
                      "engine": args.engine, "gpu": args.gpu, "dispatch": args.dispatch,
                      "rows": rows, "all_token_ids_equal": report["all_token_ids_equal"]}), flush=True)
    return 0 if prior is None or exact else 1


if __name__ == "__main__":
    sys.exit(main())

"""Collect one native CUDA first-token panel from an already running server."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import signal
from typing import Any
import urllib.request

import bench_concurrent as bench


SCHEMA = "pr836_ttft_http_panel_v1"
LEVELS = (1, 4, 16)
OUTPUT_TOKENS = 256
LONG_OUTPUT = 32
SEED = 1234


def digest(path: Path) -> str:
    """Hash exact input bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, report: dict[str, Any]) -> None:
    """Persist complete and partial evidence atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def tokenize(base: str, body: dict[str, Any]) -> list[int]:
    """Read the server's actual rendered token IDs."""
    request = urllib.request.Request(
        base + "/tokenize", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        result = json.load(response)
    ids = result.get("tokens")
    if not isinstance(ids, list) or result.get("count") != len(ids):
        raise ValueError("invalid /tokenize response")
    return ids


def corpus(path: Path) -> list[dict[str, Any]]:
    """Require 42 unique, pinned prompt-ID arrays."""
    data = json.loads(path.read_text())
    if data.get("schema") != "pr836_ttft_literal_corpus_v1":
        raise ValueError("unknown corpus schema")
    cases = data.get("cases")
    if not isinstance(cases, list) or len(cases) != 42:
        raise ValueError("corpus must contain 42 cases")
    seen = set()
    for case in cases:
        key = (case["name"], case["streams"], case["stream_index"])
        if (case["case_id"] in seen or case["name"] not in ("code", "chat")
                or case["streams"] not in LEVELS
                or not 0 <= case["stream_index"] < case["streams"]
                or len(case["input_ids"]) != case["prompt_tokens"]):
            raise ValueError("corpus case identity or token array drift")
        seen.add(case["case_id"])
        if case["case_id"] != f"{case['name']}-n{case['streams']}-s{case['stream_index']:02d}":
            raise ValueError("corpus case name drift")
    if len({tuple(case["input_ids"]) for case in cases}) != len(cases):
        raise ValueError("fresh case token IDs are not distinct")
    return cases


def anchors(path: Path) -> list[dict[str, Any]]:
    """Require two tokenizer-pinned literal code/chat prompts."""
    data = json.loads(path.read_text())
    cases = data.get("anchors")
    if (not isinstance(cases, list) or len(cases) != 2
            or [case.get("case_id") for case in cases] != ["code-anchor", "chat-anchor"]
            or any(type(case.get("prompt_tokens")) is not int
                   or case["prompt_tokens"] <= 0
                   or len(case.get("input_ids", [])) != case["prompt_tokens"]
                   for case in cases)):
        raise ValueError("anchor prompt identity drift")
    return cases


def changes(path: Path) -> list[dict[str, Any]]:
    """Read pinned changed code and growing-chat smoke inputs."""
    data = json.loads(path.read_text())
    cases = data.get("changes")
    if (not isinstance(cases, list) or len(cases) != 2
            or [case.get("case_id") for case in cases]
            != ["code-changed", "chat-changed"]
            or any(type(case.get("prompt_tokens")) is not int
                   or case["prompt_tokens"] <= 0
                   or len(case.get("input_ids", [])) != case["prompt_tokens"]
                   for case in cases)):
        raise ValueError("changed prompt identity drift")
    return cases


def preflight(base: str, cases: list[dict[str, Any]]) -> dict[str, str]:
    """Fail before timing if either server renders any case differently."""
    observed = {}
    for case in cases:
        if case["kind"] == "chat":
            body = {"messages": case["messages"], "add_generation_prompt": True,
                    "chat_template_kwargs": {"enable_thinking": False}}
        else:
            body = {"prompt": case["prompt"], "add_special_tokens": True}
        ids = tokenize(base, body)
        if ids != case["input_ids"]:
            raise ValueError(f"{case['case_id']}: live token IDs differ")
        observed[case["case_id"]] = hashlib.sha256(
            ",".join(map(str, ids)).encode(),
        ).hexdigest()
    return observed


def validate(row: dict[str, Any], case: dict[str, Any], *, cache_rule: str,
             tokens: int = OUTPUT_TOKENS) -> None:
    """Require exact work completion and the measured cache condition."""
    if row.get("error") or row.get("complete") is not True:
        raise ValueError(f"{case['case_id']}: incomplete response")
    if row.get("tokens") != tokens or row.get("finish_reason") != "length":
        raise ValueError(f"{case['case_id']}: wrong output length or finish")
    if tokens == OUTPUT_TOKENS and row.get("unmeasured"):
        raise ValueError(f"{case['case_id']}: no decode interval")
    usage = row.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    ncache = details.get("cached_tokens")
    if (usage.get("prompt_tokens") != case["prompt_tokens"]
            or usage.get("completion_tokens") != tokens
            or row.get("cached_tokens") != ncache
            or type(ncache) is not int or ncache < 0
            or (cache_rule == "positive" and ncache <= 0)
            or (cache_rule == "full" and ncache != case["prompt_tokens"])
            or (cache_rule == "zero" and ncache != 0)):
        raise ValueError(f"{case['case_id']}: prompt or cache usage mismatch")
    body = row.get("request_body") or {}
    expected = {"model": body.get("model"), "max_tokens": tokens,
                "temperature": 0.0, "stream": True,
                "stream_options": {"include_usage": True},
                "ignore_eos": True, "seed": SEED}
    if case["kind"] == "chat":
        expected.update(messages=case["messages"],
                        chat_template_kwargs={"enable_thinking": False})
    else:
        expected["prompt"] = case["prompt"]
    if (body != expected or row.get("request_body_sha256") != hashlib.sha256(
            json.dumps(expected).encode()).hexdigest()):
        raise ValueError(f"{case['case_id']}: literal body mismatch")
    clocks = [row.get(key) for key in ("sent", "first", "last", "done")]
    if (any(type(value) not in (int, float) or not math.isfinite(value)
            for value in clocks) or not clocks[0] < clocks[1] <= clocks[2] <= clocks[3]):
        raise ValueError(f"{case['case_id']}: invalid SSE clock order")
    if (not math.isclose(row.get("ttft_s", -1), clocks[1] - clocks[0], abs_tol=1e-7)
            or not math.isclose(row.get("full_response_s", -1), clocks[3] - clocks[0], abs_tol=1e-7)):
        raise ValueError(f"{case['case_id']}: SSE duration mismatch")
    if not row.get("output_sha256") or not row.get("token_sha"):
        raise ValueError(f"{case['case_id']}: missing output evidence")


def long_cases(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read a tokenizer-pinned changing chat history without assuming reuse."""
    data = json.loads(path.read_text())
    if data.get("schema") != "pr836_chat_history_fixture_v1":
        raise ValueError("unknown changing-history fixture")
    initial = data["rendered"]["initial"]["generation"]
    followup = data["rendered"]["followup"]["generation"]
    common = data.get("common_full_prefix_tokens")
    if (type(common) is not int or common < 4096
            or common > min(len(initial), len(followup))
            or initial[:common] != followup[:common]):
        raise ValueError("changing-history fixture lacks a shared 4096-token prefix")
    return tuple({
        "case_id": f"long-{name}", "name": f"long_{name}", "kind": "chat",
        "messages": data["messages"][name],
        "prompt_tokens": len(data["rendered"][name]["generation"]),
        "input_ids": data["rendered"][name]["generation"],
    } for name in ("initial", "followup"))


def collect(args: argparse.Namespace) -> dict[str, Any]:
    """Run fresh and repeated batches while preserving every raw row."""
    cases = corpus(args.corpus)
    if args.repeat_cache != "zero":
        raise ValueError("PR836 Qwen3.5 CUDA has no prompt-state cache")
    original = anchors(args.corpus)
    changed = changes(args.corpus)
    report: dict[str, Any] = {
        "schema": SCHEMA, "complete": False, "label": args.label,
        "mode": "smoke" if args.smoke else "full",
        "repeat_cache_rule": args.repeat_cache,
        "base": args.base, "model": args.model,
        "corpus_sha256": digest(args.corpus),
        "long_manifest_sha256": digest(args.long_manifest),
        "source_sha256": {name: digest(Path(__file__).parent / name) for name in (
            "run_panel.py", "bench_concurrent.py", "bench_evidence.py")},
        "tokenize_preflight": {}, "anchors": {}, "changed": {},
        "cells": [], "long": {},
    }
    write(args.output, report)
    required = original + changed + ([] if args.smoke else cases)
    report["tokenize_preflight"] = preflight(args.base, required)
    write(args.output, report)
    for case in original:
        entry: dict[str, Any] = {}
        report["anchors"][case["name"]] = entry
        for phase in ("fresh", "repeat"):
            row = bench.stream(args.base, args.model, case, OUTPUT_TOKENS, 0.0, SEED)
            entry[phase] = row
            write(args.output, report)
            validate(row, case, cache_rule=(
                args.repeat_cache if phase == "repeat" else "zero"
            ))
        if (entry["fresh"]["output_sha256"], entry["fresh"]["token_sha"]) != (
                entry["repeat"]["output_sha256"], entry["repeat"]["token_sha"]):
            raise ValueError(f"{case['name']}: anchor output changed on repeat")
    for case in changed:
        row = bench.stream(args.base, args.model, case, OUTPUT_TOKENS, 0.0, SEED)
        report["changed"][case["case_id"]] = row
        write(args.output, report)
        validate(row, case, cache_rule="zero")
    if args.smoke:
        report["complete"] = True
        write(args.output, report)
        return report
    for name in ("code", "chat"):
        for streams in LEVELS:
            selected = [case for case in cases
                        if case["name"] == name and case["streams"] == streams]
            selected.sort(key=lambda case: case["stream_index"])
            if len(selected) != streams:
                raise ValueError("missing stream case")
            specs = [(case, SEED) for case in selected]
            cell: dict[str, Any] = {"name": name, "streams": streams,
                                    "case_ids": [case["case_id"] for case in selected]}
            report["cells"].append(cell)
            fresh = bench.together(args.base, args.model, specs, OUTPUT_TOKENS, 0.0)
            cell["fresh"] = fresh
            write(args.output, report)
            if len(fresh) != streams:
                raise ValueError(f"{name}/{streams}/fresh: incomplete burst")
            for row, case in zip(fresh, selected):
                validate(row, case, cache_rule="zero")
            # Use one literal key for an exact-repeat concurrency workload.
            anchor = selected[0]
            prime = bench.stream(args.base, args.model, anchor, OUTPUT_TOKENS, 0.0, SEED)
            cell["prime"] = prime
            write(args.output, report)
            validate(prime, anchor, cache_rule="zero")
            repeated = bench.together(
                args.base, args.model, [(anchor, SEED)] * streams,
                OUTPUT_TOKENS, 0.0,
            )
            cell["repeat"] = repeated
            write(args.output, report)
            if len(repeated) != streams:
                raise ValueError(f"{name}/{streams}/repeat: incomplete burst")
            for row in repeated:
                validate(row, anchor, cache_rule=args.repeat_cache)
                if (row["output_sha256"], row["token_sha"]) != (
                        fresh[0]["output_sha256"], fresh[0]["token_sha"]):
                    raise ValueError(f"{name}/{streams}: reply changed on repeat")
            if (prime["output_sha256"], prime["token_sha"]) != (
                    fresh[0]["output_sha256"], fresh[0]["token_sha"]):
                raise ValueError(f"{name}/{streams}: primer output changed")
    if not args.smoke:
        first, followup = long_cases(args.long_manifest)
        for name, case in (("initial", first), ("followup", followup)):
            body = {"messages": case["messages"], "add_generation_prompt": True,
                    "chat_template_kwargs": {"enable_thinking": False}}
            if tokenize(args.base, body) != case["input_ids"]:
                raise ValueError(f"long/{name}: live token IDs differ")
            row = bench.stream(args.base, args.model, case, LONG_OUTPUT, 0.0, SEED)
            report["long"][name] = row
            write(args.output, report)
            validate(row, case, cache_rule="zero", tokens=LONG_OUTPUT)
    report["complete"] = True
    write(args.output, report)
    return report


def main() -> None:
    """Collect the bounded server panel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--long-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=1800)
    parser.add_argument("--smoke", action="store_true", help="Run only the original code/chat fresh and repeat")
    parser.add_argument("--repeat-cache", choices=("zero",), default="zero",
                        help="PR836 CUDA exposes no prompt-state reuse")
    args = parser.parse_args()
    if args.max_seconds <= 0:
        parser.error("--max-seconds must be positive")
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("panel deadline")))
    signal.alarm(args.max_seconds)
    try:
        collect(args)
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()

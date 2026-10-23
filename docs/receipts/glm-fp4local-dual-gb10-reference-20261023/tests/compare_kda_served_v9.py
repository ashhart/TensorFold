#!/usr/bin/env python3
"""Compare retained served-decode JSONL receipts; never contact a device/API.

Usage:
  python tests/compare_kda_served_v9.py --baseline v5 \
    v5=results/decode-v7-baseline-observed.jsonl v9=results/decode-v9-on.jsonl
  python tests/compare_kda_served_v9.py --baseline v5 receipts.jsonl --output comparison.json
  python tests/compare_kda_served_v9.py --self-test

LABEL=PATH overrides labels in that file; a plain PATH uses each receipt's label.
Groups are (cell, exact request-settings SHA256), with prompt content excluded
from the settings only. Full request hashes still include the complete prompt.
Only decode tok/s is compared: cached-prefix differences prevent claims about
cold prefill or total request wall-time improvements.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics


PROMPT_KEYS = {"messages", "input", "prompt", "instructions"}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def distribution(values):
    values = [v for v in values if number(v)]
    if not values:
        return None
    mean = statistics.mean(values)
    return {"count": len(values), "mean": mean, "min": min(values), "max": max(values),
            "relative_spread_pct": 100 * (max(values) - min(values)) / mean if mean else None}


def unique(values):
    return [json.loads(v) for v in sorted({canonical(v) for v in values})]


def same_known(a, b):
    """Strict single-valued equality, not equality between two varying sets."""
    def known(value):
        if isinstance(value, dict):
            return all(known(v) for v in value.values())
        if isinstance(value, list):
            return all(known(v) for v in value)
        return value is not None
    return bool(a and b) and all(known(v) for v in a + b) and len(unique(a + b)) == 1


def request_thinking(request):
    kwargs = request.get("chat_template_kwargs", {})
    if isinstance(kwargs, dict) and "enable_thinking" in kwargs:
        return kwargs["enable_thinking"]
    thinking = request.get("thinking", {})
    if isinstance(thinking, dict) and thinking.get("type") in ("enabled", "disabled"):
        return thinking["type"] == "enabled"
    return request.get("enable_thinking")


def normalize(row, override=None, source=None, line=None):
    request = row.get("request")
    request = request if isinstance(request, dict) else {}
    result = row.get("result")
    result = result if isinstance(result, dict) else {}
    runtime = result.get("tensorfold", {})
    spec = result.get("speculative", {})
    usage = result.get("usage", {})
    runtime, spec, usage = [v if isinstance(v, dict) else {} for v in (runtime, spec, usage)]
    raw_errors = row.get("errors", [])
    errors = list(raw_errors) if isinstance(raw_errors, list) else [raw_errors]
    errors = [str(v) if isinstance(v, str) else canonical(v) for v in errors if v is not None]
    for owner in (row, result):
        if owner.get("error"):
            errors.append(canonical(owner["error"]))
    if not request:
        errors.append("missing request object")
    if not result:
        errors.append("missing result object")
    prompt_hash, token_hash = row.get("prompt_sha256"), runtime.get("token_sha256")
    for name, value in (("prompt_sha256", prompt_hash), ("token_sha256", token_hash)):
        if not valid_sha(value):
            errors.append(f"missing/invalid full {name}")
    counts = {k: spec.get(k) for k in ("rounds", "drafted", "accepted")}
    if not all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in counts.values()):
        errors.append("missing/invalid speculative counts")
    completion_details = usage.get("completion_tokens_details") or {}
    prompt_details = usage.get("prompt_tokens_details") or {}
    completion_details = completion_details if isinstance(completion_details, dict) else {}
    prompt_details = prompt_details if isinstance(prompt_details, dict) else {}
    tokens = {k: usage.get(k) for k in ("prompt_tokens", "completion_tokens")}
    tokens["reasoning_tokens"] = completion_details.get("reasoning_tokens")
    if not all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in tokens.values()):
        errors.append("missing/invalid token counts (including reasoning_tokens)")
    if row.get("expected_prompt_tokens") is not None and tokens["prompt_tokens"] != row["expected_prompt_tokens"]:
        errors.append("prompt count differs from fixture")
    thinking_request, thinking_runtime = request_thinking(request), runtime.get("enable_thinking")
    if thinking_request is not True or thinking_runtime is not True:
        errors.append("thinking=true not confirmed in both request and runtime")
    tps = runtime.get("tokens_per_second")
    if not number(tps) or tps <= 0:
        errors.append("missing/invalid positive decode tok/s")
        tps = None
    settings = {k: v for k, v in request.items() if k not in PROMPT_KEYS}
    return {"label": override or row.get("label") or "unlabeled", "cell": row.get("cell", "missing-cell"),
            "rep": row.get("rep"), "source": source, "line": line,
            "settings": settings, "settings_sha256": sha(settings),
            "request_sha256": sha(request) if request else None,
            "prompt_sha256": prompt_hash, "prompt_content_sha256": sha({k: request[k] for k in PROMPT_KEYS if k in request}),
            "output_token_sha256": token_hash, "speculative": counts, "tokens": tokens,
            "thinking_request": thinking_request, "thinking_runtime": thinking_runtime,
            "decode_tps": tps, "decode_seconds": runtime.get("seconds"),
            "cached_tokens": prompt_details.get("cached_tokens"),
            "errors": errors}


def compare(runs, baseline):
    grouped = defaultdict(lambda: defaultdict(list))
    for run in runs:
        grouped[run["cell"], run["settings_sha256"]][run["label"]].append(run)
    groups = []
    for (cell, settings_hash), labels in sorted(grouped.items()):
        reference = labels.get(baseline, [])
        items = []
        for label, records in sorted(labels.items(), key=lambda item: (item[0] != baseline, item[0])):
            metrics = distribution([r["decode_tps"] for r in records])
            baseline_metrics = distribution([r["decode_tps"] for r in reference])
            equality = {name: same_known([r[field] for r in reference], [r[field] for r in records])
                        for name, field in (("request", "request_sha256"), ("prompt", "prompt_sha256"),
                                            ("output_tokens", "output_token_sha256"),
                                            ("speculative_counts", "speculative"), ("token_counts", "tokens"))}
            equality["thinking_true"] = bool(reference) and all(
                r["thinking_request"] is True and r["thinking_runtime"] is True for r in reference + records)
            equality["no_errors"] = bool(reference) and all(not r["errors"] for r in reference + records)
            delta = (100 * (metrics["mean"] / baseline_metrics["mean"] - 1)
                     if metrics and baseline_metrics and baseline_metrics["mean"] else None)
            items.append({"label": label, "runs": records, "requests": len(records),
                          "decode_tps": metrics, "decode_tps_change_pct": delta,
                          "baseline_available": bool(reference), "equal_to_baseline": equality,
                          "decode_comparison_qualified": all(equality.values()),
                          "request_sha256": unique([r["request_sha256"] for r in records]),
                          "prompt_sha256": unique([r["prompt_sha256"] for r in records]),
                          "output_token_sha256": unique([r["output_token_sha256"] for r in records]),
                          "speculative_counts": unique([r["speculative"] for r in records]),
                          "token_counts": unique([r["tokens"] for r in records]),
                          "cached_tokens": unique([r["cached_tokens"] for r in records]),
                          "error_runs": sum(bool(r["errors"]) for r in records)})
        groups.append({"cell": cell, "settings_sha256": settings_hash,
                       "settings": next(iter(labels.values()))[0]["settings"], "settings_runs": items})
    return {"schema": "glm-kda-served-v9-comparison/1", "baseline": baseline,
            "requests": len(runs), "groups": groups,
            "errors": [{"source": r["source"], "line": r["line"], "label": r["label"],
                        "cell": r["cell"], "rep": r["rep"], "errors": r["errors"]}
                       for r in runs if r["errors"]],
            "all_comparisons_qualified": bool(groups) and all(
                item["decode_comparison_qualified"] for group in groups for item in group["settings_runs"]),
            "limitations": [
                "Request hashes are SHA256 of canonical parsed JSON (UTF-8, sorted keys, compact separators), not captured HTTP bytes.",
                "prompt_sha256 is the benchmark's full prompt-token-ID hash; prompt_content_sha256 separately hashes the recorded prompt fields.",
                "Output-token equality uses the runtime's full token_sha256; hashes are not recomputed without raw output token IDs.",
                "Equality requires one identical value across all baseline and candidate repeats, not merely matching sets.",
                "Decode means/ranges are descriptive; a few sequential repeats do not establish a confidence interval or eliminate run-order drift.",
                "Cached-prefix lengths may differ: cold prefill and total request wall-time gains are NOT compared or claimed.",
                "Fixed-token runs may stop in reasoning or mid-answer; token equality is not a broad model-quality evaluation.",
                "Reported decode tok/s is the API metric; for 256 completion tokens it may time 255 inter-token intervals.",
            ]}


def compact(result):
    print("cell | setting | label | n | mean tok/s [min,max] | vs baseline | req/prompt/out/spec/count/think/no-errors")
    for group in result["groups"]:
        for item in group["settings_runs"]:
            metric, delta = item["decode_tps"], item["decode_tps_change_pct"]
            speed = f"{metric['mean']:.3f} [{metric['min']:.3f},{metric['max']:.3f}]" if metric else "missing"
            change = f"{delta:+.3f}%" if delta is not None else "no baseline"
            equality = item["equal_to_baseline"]
            checks = "/".join("Y" if equality[k] else "N" for k in (
                "request", "prompt", "output_tokens", "speculative_counts", "token_counts", "thinking_true", "no_errors"))
            warning = "" if item["decode_comparison_qualified"] else " UNQUALIFIED"
            print(f"{group['cell']} | {group['settings_sha256'][:12]} | {item['label']} | {item['requests']} | "
                  f"{speed} | {change} | {checks}{warning}")
        print("  settings:", canonical(group["settings"]))
        for item in group["settings_runs"]:
            print(f"  {item['label']} request_sha256={canonical(item['request_sha256'])}")
            print(f"  {item['label']} prompt_sha256={canonical(item['prompt_sha256'])}")
            print(f"  {item['label']} output_token_sha256={canonical(item['output_token_sha256'])}")
            print(f"  {item['label']} tokens={canonical(item['token_counts'])} speculative={canonical(item['speculative_counts'])} "
                  f"cached_tokens={canonical(item['cached_tokens'])}")
    print("All comparisons qualified:", result["all_comparisons_qualified"], "; error runs:", len(result["errors"]))
    for error in result["errors"]:
        print("ERROR", canonical(error))
    print("Decode only: no cold-prefill or total-wall-time gain is claimed.")


def self_test():
    sample = {"label": "base", "cell": "prose-8192", "rep": 0,
              "request": {"model": "model", "messages": [{"role": "user", "content": "测试"}],
                          "temperature": 1.0, "top_p": 0.95, "seed": 119,
                          "chat_template_kwargs": {"enable_thinking": True}},
              "prompt_sha256": "1" * 64, "expected_prompt_tokens": 8192,
              "result": {"tensorfold": {"tokens_per_second": 40.0, "seconds": 6.375,
                                        "enable_thinking": True, "token_sha256": "2" * 64},
                         "speculative": {"rounds": 80, "drafted": 230, "accepted": 175},
                         "usage": {"prompt_tokens": 8192, "completion_tokens": 256,
                                   "completion_tokens_details": {"reasoning_tokens": 30},
                                   "prompt_tokens_details": {"cached_tokens": 0}}}, "errors": []}
    candidate = copy.deepcopy(sample)
    candidate["label"] = "new"
    candidate["result"]["tensorfold"]["tokens_per_second"] = 44.0
    candidate["result"]["usage"]["prompt_tokens_details"]["cached_tokens"] = 6144
    result = compare([normalize(sample), normalize(candidate)], "base")
    item = result["groups"][0]["settings_runs"][1]
    assert result["all_comparisons_qualified"] and abs(item["decode_tps_change_pct"] - 10) < 1e-9
    assert sha({"a": 1, "b": "测试"}) == sha({"b": "测试", "a": 1})
    changed = copy.deepcopy(candidate)
    changed["result"]["tensorfold"]["token_sha256"] = "3" * 64
    assert not compare([normalize(sample), normalize(changed)], "base")["all_comparisons_qualified"]
    changed = copy.deepcopy(candidate)
    changed["request"]["seed"] = 120
    result = compare([normalize(sample), normalize(changed)], "base")
    assert len(result["groups"]) == 2 and not result["all_comparisons_qualified"]
    changed = copy.deepcopy(candidate)
    changed["result"]["tensorfold"]["enable_thinking"] = False
    assert normalize(changed)["errors"]
    failed = {"label": "new", "cell": "prose-8192", "request": sample["request"], "error": "connection failed"}
    assert normalize(failed)["errors"]
    assert not same_known(["a", "b"], ["a", "b"])
    assert not same_known([None], [None])
    assert not same_known([{"rounds": None}], [{"rounds": None}])
    print("PASS: equality, exact hashes, settings isolation, +10% decode, cache exclusion, failures and thinking gates")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="*", help="JSONL paths, optionally LABEL=PATH")
    parser.add_argument("--baseline", help="Label used as baseline within the same cell/settings group")
    parser.add_argument("--output", type=Path, help="Optionally save the full comparison JSON")
    parser.add_argument("--self-test", action="store_true", help="Synthetic CPU-only tests; read no receipts")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.inputs or not args.baseline:
        parser.error("at least one input and --baseline are required")
    runs, sources, skipped = [], [], 0
    try:
        for spec in args.inputs:
            override, separator, filename = spec.partition("=")
            if not separator:
                filename, override = spec, None
            path = Path(filename)
            sources.append({"path": str(path.resolve()), "label_override": override,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            with path.open(encoding="utf-8") as handle:
                for line, text in enumerate(handle, 1):
                    if not text.strip():
                        continue
                    row = json.loads(text)
                    if not isinstance(row, dict) or "cell" not in row or not any(k in row for k in ("request", "result", "error")):
                        skipped += 1
                        continue
                    runs.append(normalize(row, override, str(path.resolve()), line))
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    if not runs:
        parser.error("no served request receipts found")
    if args.baseline not in {r["label"] for r in runs}:
        parser.error("baseline label does not occur in the receipts")
    result = compare(runs, args.baseline)
    result.update(sources=sources, skipped_metadata_rows=skipped)
    compact(result)
    if args.output:
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Summarize repeat stability and speculative rounds from retained API receipts."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def distribution(values):
    mean = statistics.mean(values)
    return {"mean": mean, "min": min(values), "max": max(values),
            "relative_spread_pct": 100 * (max(values) - min(values)) / mean if mean else None,
            "population_cv_pct": 100 * statistics.pstdev(values) / mean if mean else None}


def summarize(rows):
    groups = defaultdict(list)
    for r in rows:
        if "result" in r:
            groups[r["cell"]].append(r)
    cells = []
    contexts = defaultdict(list)
    for cell, records in groups.items():
        runs = []
        for r in records:
            response = r["result"]
            tf, spec, usage = (response[k] for k in ("tensorfold", "speculative", "usage"))
            run = {"rep": r["rep"], "decode_tps": tf["tokens_per_second"],
                   "decode_seconds": tf["seconds"], "rounds": spec["rounds"],
                   "api_decode_ms_per_round_estimate": 1000 * tf["seconds"] / spec["rounds"],
                   "drafted": spec["drafted"], "accepted": spec["accepted"],
                   "acceptance_rate": spec["accepted"] / spec["drafted"],
                   "tokens_per_round": spec["tokens_per_round"],
                   "drafts_per_round": spec["drafted"] / spec["rounds"],
                   "completion_tokens": usage["completion_tokens"],
                   "reasoning_tokens": usage.get("completion_tokens_details", {}).get("reasoning_tokens"),
                   "cached_tokens": usage.get("prompt_tokens_details", {}).get("cached_tokens"),
                   "prompt_tokens": usage["prompt_tokens"], "prefill_seconds": tf["prefill_seconds"],
                   "errors": r.get("errors", [])}
            runs.append(run)
            contexts[usage["prompt_tokens"]].append(run)
        item = {"cell": cell, "prompt_tokens": runs[0]["prompt_tokens"], "runs": runs,
                "decode_tps": distribution([r["decode_tps"] for r in runs]),
                "round_ms_estimate": distribution([r["api_decode_ms_per_round_estimate"] for r in runs]),
                "acceptance_rate": distribution([r["acceptance_rate"] for r in runs]),
                "tokens_per_round": distribution([r["tokens_per_round"] for r in runs]),
                "same_request": len({json.dumps(r["request"], sort_keys=True) for r in records}) == 1,
                "same_output_tokens": len({r["result"]["tensorfold"]["token_sha256"] for r in records}) == 1,
                "same_speculative_counts": len({tuple(r["result"]["speculative"][k]
                                                       for k in ("rounds", "drafted", "accepted"))
                                                 for r in records}) == 1}
        cells.append(item)
    cells.sort(key=lambda r: (r["prompt_tokens"], r["cell"]))
    rollup = []
    for ctx, runs in sorted(contexts.items()):
        rounds = sum(r["rounds"] for r in runs)
        drafted = sum(r["drafted"] for r in runs)
        seconds = sum(r["decode_seconds"] for r in runs)
        rollup.append({"prompt_tokens": ctx, "requests": len(runs),
                       "decode_tps_equal_request_mean": statistics.mean(r["decode_tps"] for r in runs),
                       "rounds_total": rounds, "drafted_total": drafted,
                       "accepted_total": sum(r["accepted"] for r in runs),
                       "acceptance_rate_weighted": sum(r["accepted"] for r in runs) / drafted,
                       "tokens_per_round_weighted": sum(r["completion_tokens"] for r in runs) / rounds,
                       "api_decode_ms_per_round_weighted_estimate": 1000 * seconds / rounds,
                       "cached_tokens_observed": sorted({r["cached_tokens"] for r in runs})})
    contrasts = []
    lookup = {r["cell"]: r for r in cells}
    for task in ("prose", "code"):
        for before, after in ((8192, 32768), (32768, 131072), (8192, 131072)):
            a, b = lookup.get(f"{task}-{before}"), lookup.get(f"{task}-{after}")
            if a is None or b is None:
                continue
            contrasts.append({"task": task, "before": before, "after": after,
                              "decode_tps_change_pct": 100 * (b["decode_tps"]["mean"] / a["decode_tps"]["mean"] - 1),
                              "round_ms_estimate_change_pct": 100 * (b["round_ms_estimate"]["mean"] /
                                                                      a["round_ms_estimate"]["mean"] - 1),
                              "tokens_per_round_change_pct": 100 * (b["tokens_per_round"]["mean"] /
                                                                     a["tokens_per_round"]["mean"] - 1),
                              "acceptance_change_percentage_points": 100 * (b["acceptance_rate"]["mean"] -
                                                                            a["acceptance_rate"]["mean"])})
    return {"schema": "glm-decode-v7-baseline/1", "requests": sum(map(len, groups.values())),
            "cells": cells, "context_rollup": rollup, "context_contrasts": contrasts,
            "all_repeat_token_sequences_equal": all(c["same_output_tokens"] for c in cells),
            "all_repeat_speculative_counts_equal": all(c["same_speculative_counts"] for c in cells),
            "max_repeat_decode_relative_spread_pct": max(c["decode_tps"]["relative_spread_pct"] for c in cells),
            "limitations": [
                "Only two repetitions per cell; spread/CV is descriptive, not a confidence interval.",
                "API decode seconds / speculative rounds is an amortized estimate, not instrumented complete-round wall time.",
                "API tokens_per_second currently uses 255 timed intervals for 256 completion tokens; round counts cover the request.",
                "Cached prefix lengths differ; prefill seconds and request wall times are not cold-prefill comparisons.",
                "Different context lengths produce different continuation tokens and reasoning content, even with the same task and seed.",
                "The 128K code run's 256 completion tokens are all reasoning; do not call it a completed HTML-generation benchmark.",
                "Acceptance and tokens/round help explain throughput, but these API receipts cannot identify kernel bottlenecks.",
            ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.input.open(encoding="utf-8") as handle:
        result = summarize([json.loads(line) for line in handle if line.strip()])
    result["source"] = str(args.input)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print("cell | mean tok/s | repeat spread | est. ms/round | accept | tokens/round")
    for c in result["cells"]:
        print(f"{c['cell']:>13} | {c['decode_tps']['mean']:.3f} | {c['decode_tps']['relative_spread_pct']:.3f}%"
              f" | {c['round_ms_estimate']['mean']:.3f} | {100*c['acceptance_rate']['mean']:.2f}%"
              f" | {c['tokens_per_round']['mean']:.4f}")
    print("Repeated tokens equal:", result["all_repeat_token_sequences_equal"],
          "; speculative counts equal:", result["all_repeat_speculative_counts_equal"])


if __name__ == "__main__":
    main()

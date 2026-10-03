"""Paired full-decode confirmation of Qwen 27B DeltaNet scalar launch geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

from bench_qwen_deltanet import scalar_plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--output", default="deltanet-decode-confirmation.json")
    args = p.parse_args()
    if min(args.tokens, args.reps) < 1:
        p.error("tokens and reps must be positive")
    import mlx.core as mx
    from tensorfold.families.qwen3_5 import load
    from tensorfold.engine.family_common import cache_contents
    from tensorfold.kernels.qwen.dense.v1 import simd_qmm as q

    original = q._launch
    original_tuning = q._QWEN27_GDN_TUNE
    q._QWEN27_GDN_TUNE = False
    family, tokenizer = load(Path(args.model), lane_kernels="off", drafter="")
    fixtures = [
        tokenizer.encode("Explain how a GPU works in plain English."),
        tokenizer.encode("Write a short Python function that computes the Fibonacci sequence and explain it."),
    ]
    geometries = {(16480, 5120): (4, 1, 32), (5120, 6144): (4, 1, 16)}
    target_shapes = set(geometries)

    def candidate(kind, rows, n, dims, group=q.GROUP, most=q.MMA_SGS):
        if kind == "scalar" and rows == 1 and group == 64 and (n, dims) in target_shapes:
            return scalar_plan(n, dims, group, *geometries[n, dims])
        return original(kind, rows, n, dims, group, most)

    def run(prompt, count):
        cache = family.make_cache()
        h = family.prefill(prompt, cache)
        pending = int(family.sample(family.head(h[:, -1:]), None, [len(prompt)])[0])
        mx.eval([a for item in cache for a in cache_contents(item)])
        mx.synchronize()
        tokens = []
        start = time.perf_counter()
        for _ in range(count):
            h = family.hidden([pending], cache)
            logits = family.head(h)
            pending = int(family.sample(logits, None, [family._position(cache)])[0])
            mx.eval([a for item in cache for a in cache_contents(item)])
            mx.synchronize()
            tokens.append(pending)
        seconds = time.perf_counter() - start
        return {"token_ids": tokens, "seconds": seconds, "ms_per_token": seconds * 1000 / count}, logits, cache

    results = []
    try:
        for prompt in fixtures:
            for launch in (original, candidate):
                q._launch = launch
                q._plans.clear()
                run(prompt, 8)
            runs = {"baseline": [], "candidate": []}
            for rep in range(args.reps):
                reference = None
                order = [
                    ("baseline", original),
                    ("candidate", candidate),
                    ("candidate", candidate),
                    ("baseline", original),
                ]
                if rep % 2:
                    order = [
                        ("candidate", candidate),
                        ("baseline", original),
                        ("baseline", original),
                        ("candidate", candidate),
                    ]
                for label, launch in order:
                    q._launch = launch
                    q._plans.clear()
                    row, logits, cache = run(prompt, args.tokens)
                    runs[label].append(row)
                    if reference is None:
                        reference = row, logits, cache
                    else:
                        expected, base_logits, base_cache = reference
                        if expected["token_ids"] != row["token_ids"]:
                            raise RuntimeError("candidate changed tokens")
                        if not bool(mx.all(base_logits.view(mx.uint16) == logits.view(mx.uint16)).item()):
                            raise RuntimeError("candidate changed final logits")
                        for base, changed in zip(base_cache, cache):
                            for a, b in zip(cache_contents(base), cache_contents(changed)):
                                words = mx.uint32 if a.dtype == mx.float32 else mx.uint16
                                if not bool(mx.all(a.view(words) == b.view(words)).item()):
                                    raise RuntimeError("candidate changed cache state")
                print(
                    f"Prompt {len(prompt)} tokens, pair {rep + 1}: "
                    f"{statistics.mean(r['ms_per_token'] for r in runs['baseline'][-2:]):.3f} -> "
                    f"{statistics.mean(r['ms_per_token'] for r in runs['candidate'][-2:]):.3f} ms/token; bits match",
                    flush=True,
                )
            base_ms = statistics.median(r["ms_per_token"] for r in runs["baseline"])
            cand_ms = statistics.median(r["ms_per_token"] for r in runs["candidate"])
            results.append(
                {
                    "prompt_ids": prompt,
                    "runs": runs,
                    "baseline_median_ms": base_ms,
                    "candidate_median_ms": cand_ms,
                    "speedup": base_ms / cand_ms,
                    "tokens_logits_cache_equal": True,
                }
            )
    finally:
        q._launch = original
        q._QWEN27_GDN_TUNE = original_tuning
        q._plans.clear()
    Path(args.output).write_text(
        json.dumps(
            {
                "device": mx.device_info(),
                "model": args.model,
                "tokens": args.tokens,
                "reps": args.reps,
                "method": "ABBA/BAAB width-one greedy decode; prefill excluded; original geometry forced for baseline",
                "candidate": {"geometries": {str(shape): config for shape, config in geometries.items()}},
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )
    for row in results:
        print(f"{row['baseline_median_ms']:.3f} -> {row['candidate_median_ms']:.3f} ms/token ({row['speedup']:.3f}x)")


if __name__ == "__main__":
    main()

"""Qwen 27B one-token DeltaNet projections, recurrence and residual/RMSNorm.

Cycles eight real layers for projections and all FP32 layer states for recurrence.
Launch geometry candidates retain all FMA and reduction order. Time serialized
batches, not per-stage synchronization; compare ABBA sweeps and exact outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time


def scalar_plan(n, k, group, sgs, nr, xb):
    from tensorfold.kernels.qwen.dense.v1 import simd_qmm as q

    s = q.splits(n, k)
    if xb % s or xb * 76 * 4 > 20480:
        raise ValueError("staging must fit within 20 KiB and preserve chunk boundaries")
    per = sgs * (32 // s) * nr
    consts = (("K", k), ("N", n), ("S", s), ("SGS", sgs), ("NR", nr), ("XB", xb), ("GS", group), ("RS", 1))
    return consts, ((n + per - 1) // per * sgs * 32, 1, 1), (sgs * 32, 1, 1), [(1, n)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--reps", type=int, default=9)
    p.add_argument("--calls", type=int, default=48)
    p.add_argument("--output", default="qwen-deltanet-experiment.json")
    args = p.parse_args()
    if min(args.reps, args.calls) < 1:
        p.error("reps and calls must be positive")
    import mlx.core as mx
    from tensorfold.kernels.qwen.dense.v1 import simd_qmm as q, row_glue as rg

    q._QWEN27_GDN_TUNE = False
    q._plans.clear()
    root = Path(args.model).expanduser().resolve()
    raw_config = json.loads((root / "config.json").read_text())
    cfg = raw_config["text_config"]
    quant = raw_config.get("quantization") or raw_config.get("quantization_config") or {}
    if cfg.get("hidden_size") != 5120 or (quant.get("bits"), quant.get("group_size")) != (4, 64):
        p.error("this benchmark requires the dense Qwen 27B affine 4-bit/group-64 checkpoint")
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    layers = [i for i, kind in enumerate(cfg["layer_types"]) if kind == "linear_attention"][:8]
    suffixes = ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"]
    names = [
        f"language_model.model.layers.{i}.linear_attn.{name}.{part}"
        for i in layers
        for name in suffixes
        for part in ("weight", "scales", "biases")
    ]
    loaded_weights = {}
    for shard in sorted({index[n] for n in names}):
        got = mx.load(str(root / shard))
        loaded_weights.update({n: got[n] for n in names if index[n] == shard})
        del got
    projections = {"gdn_in": [], "gdn_out": []}
    for i in layers:
        prefix = f"language_model.model.layers.{i}.linear_attn."
        projections["gdn_in"].append(
            tuple(
                mx.concatenate([loaded_weights[prefix + name + "." + part] for name in suffixes[:-1]], axis=0)
                for part in ("weight", "scales", "biases")
            )
        )
        projections["gdn_out"].append(
            tuple(loaded_weights[prefix + "out_proj." + part] for part in ("weight", "scales", "biases"))
        )
    mx.eval([a for packs in projections.values() for pack in packs for a in pack])
    for label, packs in projections.items():
        expected = (16480, 5120) if label == "gdn_in" else (5120, 6144)
        for w, sc, bi in packs:
            n, k = expected
            if (
                w.shape != (n, k // 8)
                or sc.shape != (n, k // 64)
                or bi.shape != sc.shape
                or w.dtype != mx.uint32
                or sc.dtype != mx.bfloat16
                or bi.dtype != mx.bfloat16
            ):
                p.error("DeltaNet tensors must use the expected 4-bit/group-64 BF16 affine layout")
    del loaded_weights
    mx.clear_cache()
    mx.random.seed(1234)
    all_results = []

    def sweep(label, fns, bytes_per_call, refs):
        samples = {name: [] for name in fns}
        rejected = {}
        for name, fn in list(fns.items()):
            try:
                for i in range(len(refs)):
                    got = fn(i, None)
                    mx.eval(got)
                    for a, b in zip(got if isinstance(got, tuple) else (got,), refs[i]):
                        dtype = mx.uint32 if a.dtype == mx.float32 else mx.uint16
                        if not bool(mx.all(a.view(dtype) == b.view(dtype)).item()):
                            raise ValueError("output bits differ")
                mx.eval(fn(0, refs[0][0]))  # warm dependent variant
            except ValueError as exc:
                rejected[name] = str(exc)
                del fns[name]
        for rep in range(args.reps):
            order = list(fns) if rep % 2 == 0 else list(reversed(fns))
            for name in order:
                dep = None
                start = time.perf_counter()
                for call in range(args.calls):
                    got = fns[name](call % len(refs), dep)
                    dep = got[0] if isinstance(got, tuple) else got
                mx.eval(got)
                ms = (time.perf_counter() - start) * 1000 / args.calls
                samples[name].append(ms)
        baseline = statistics.median(samples["baseline"])
        cells = []
        for name, values in samples.items():
            if name in rejected:
                cells.append({"variant": name, "rejected": rejected[name]})
                continue
            ms = statistics.median(values)
            cell = {
                "variant": name,
                "median_ms": ms,
                "all_ms": values,
                "speedup": baseline / ms,
                "bit_equal": True,
                "effective_GB_s": bytes_per_call / ms / 1e6 if bytes_per_call else None,
            }
            cells.append(cell)
            print(f"{label:12s} {name:24s} {ms:.5f} ms ({baseline / ms:.3f}x)", flush=True)
        result = {"unit": label, "bytes_per_call": bytes_per_call, "cells": cells}
        all_results.append(result)
        return result

    for label, packs in projections.items():
        n, k = int(packs[0][0].shape[0]), int(packs[0][0].shape[1]) * 8
        xs = [(mx.random.normal((1, k)) * 0.5).astype(mx.bfloat16) for _ in packs]
        mx.eval(xs)

        def base(i, dep):
            return q.qmm(xs[i], *packs[i], kind="scalar", dep=dep)

        refs = [(base(i, None),) for i in range(len(packs))]
        mx.eval(refs)
        fns = {"baseline": base}
        for sgs, nr, xb in (
            (2, 2, 16),
            (2, 2, 64),
            (4, 2, 32),
            (4, 2, 64),
            (4, 1, 16),
            (4, 1, 32),
            (4, 1, 64),
            (8, 1, 16),
            (8, 1, 32),
            (8, 1, 64),
        ):
            plan = scalar_plan(n, k, 64, sgs, nr, xb)

            def fn(i, dep, plan=plan):
                return q._go(
                    "scalar",
                    plan,
                    dep is not None,
                    q._DEFAULT,
                    [xs[i], *packs[i], q._one] + ([dep] if dep is not None else []),
                )

            fns[f"sgs{sgs}_nr{nr}_xb{xb}"] = fn
        result = sweep(label, fns, sum(a.nbytes for a in packs[0]), refs)
        result.update({"n": n, "k": k, "cycled_weight_bytes": sum(a.nbytes for pack in packs for a in pack)})

    # Standalone recurrence, including output state completion in every timed call.
    nk, nv, dk, dv = (
        cfg[k]
        for k in ("linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim")
    )
    qs = mx.random.normal((1, 1, nk, dk)).astype(mx.bfloat16) * 0.05
    ks = mx.random.normal((1, 1, nk, dk)).astype(mx.bfloat16) * 0.05
    vs = mx.random.normal((1, 1, nv, dv)).astype(mx.bfloat16)
    gs = mx.full((1, 1, nv), 0.95, dtype=mx.float32)
    beta = mx.full((1, 1, nv), 0.5, dtype=mx.bfloat16)
    states = [mx.random.normal((1, nv, dv, dk)) * 0.01 for _ in range(cfg["layer_types"].count("linear_attention"))]
    mx.eval(qs, ks, vs, gs, beta, states)
    refs = [rg.gated_delta(qs, ks, vs, gs, beta, st, [-1], chain=True) for st in states]
    mx.eval(refs)
    src = rg._SPECS["chain"][0]
    # Extra DEP is an MLX graph dependency. Every call still reads a different state.
    dep_kernel = mx.fast.metal_kernel(
        name="qwen27_recurrence_dep_" + hashlib.sha256(src.encode()).hexdigest()[:12],
        input_names=["q", "k", "v", "g", "beta", "state_in", "DEP"],
        output_names=["y", "state_out"],
        source=src,
    )
    fns = {}
    for name, dv_rows in [("baseline", 4), ("dv_rows1", 1), ("dv_rows2", 2), ("dv_rows8", 8), ("dv_rows16", 16)]:

        def recur(i, dep, dv_rows=dv_rows):
            return tuple(
                dep_kernel(
                    inputs=[qs, ks, vs, gs, beta, states[i], qs if dep is None else dep],
                    template=[("InT", qs.dtype), ("Dk", dk), ("Dv", dv), ("Hk", nk), ("Hv", nv), ("W", 1)],
                    grid=(32, dv, nv),
                    threadgroup=(32, dv_rows, 1),
                    output_shapes=[(1, 1, nv, dv), tuple(states[i].shape)],
                    output_dtypes=[qs.dtype, mx.float32],
                )
            )

        fns[name] = recur

    def recurrence_sweep():
        samples = {name: [] for name in fns}
        cells = []
        for name, fn in fns.items():
            for i in range(len(states)):
                got = fn(i, None)
                mx.eval(got)
                if not all(
                    bool(
                        mx.all(
                            a.view(mx.uint32 if a.dtype == mx.float32 else mx.uint16)
                            == b.view(mx.uint32 if b.dtype == mx.float32 else mx.uint16)
                        ).item()
                    )
                    for a, b in zip(got, refs[i])
                ):
                    raise RuntimeError("recurrence changed bits: " + name)
            mx.eval(fn(0, refs[0][1]))
        for rep in range(args.reps):
            for name in list(fns) if rep % 2 == 0 else list(reversed(fns)):
                dep = None
                outputs = []
                start = time.perf_counter()
                for call in range(args.calls):
                    out = fns[name](call % len(states), dep)
                    dep = out[1]
                    outputs.append(out)
                mx.eval(outputs)
                samples[name].append((time.perf_counter() - start) * 1000 / args.calls)
        baseline = statistics.median(samples["baseline"])
        for name, values in samples.items():
            ms = statistics.median(values)
            cells.append(
                {"variant": name, "median_ms": ms, "all_ms": values, "speedup": baseline / ms, "bit_equal": True}
            )
            print(f"recurrence {name:24s} {ms:.5f} ms ({baseline / ms:.3f}x)", flush=True)
        all_results.append(
            {
                "unit": "recurrence",
                "state_bytes": states[0].nbytes,
                "read_write_bytes": 2 * states[0].nbytes,
                "cycled_state_bytes": sum(st.nbytes for st in states),
                "cells": cells,
            }
        )

    recurrence_sweep()

    # Complete both outputs of the existing fused residual and RMSNorm kernel.
    h = mx.random.normal((1, 1, 5120)).astype(mx.bfloat16)
    r = mx.random.normal((1, 1, 5120)).astype(mx.bfloat16)
    wt = mx.random.uniform(0.8, 1.2, (5120,)).astype(mx.bfloat16)
    mx.eval(h, r, wt)

    def norm(i, dep):
        return rg.add_norm(h if dep is None else dep, r, wt, 1e-6)

    norm_ref = norm(0, None)
    mx.eval(norm_ref)
    # No alternative arithmetic accepted here; benchmark the existing fused kernel.
    values = []
    for rep in range(args.reps):
        dep = None
        outputs = []
        start = time.perf_counter()
        for call in range(args.calls):
            ho, xo = norm(0, dep)
            dep = ho
            outputs.append(xo)
        mx.eval(outputs, dep)
        values.append((time.perf_counter() - start) * 1000 / args.calls)
    all_results.append(
        {
            "unit": "residual_rmsnorm",
            "median_ms": statistics.median(values),
            "all_ms": values,
            "note": "Existing add and RMSNorm already share one kernel; both outputs completed.",
        }
    )
    report = {
        "model": str(root),
        "device": mx.device_info(),
        "layers_cycled": layers,
        "reps": args.reps,
        "calls": args.calls,
        "method": "alternating forward/reverse candidate sweeps; serialized lazy batches, exact output checks",
        "results": all_results,
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print("Saved " + args.output, flush=True)


if __name__ == "__main__":
    main()

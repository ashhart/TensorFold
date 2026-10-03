"""Where a verify round's time goes on the GGUF path: GPU time by kernel family, and wall time against GPU busy time.

    python tools/gguf_profile.py MODEL_DIR [--widths 1 16]
"""

from __future__ import annotations

import argparse
import re
import time
from collections import defaultdict

import torch

FAMILIES = [("gguf linear", r"Quant|KQuant|GEMV|Q8_0|SmallBatch|Kernel_2Rows|wave64|Wave64"),
            ("attention", r"attn|attention|flash|_tile|_attend"),
            ("gdn / deltanet", r"gdn|delta|conv|recur|chunk"),
            ("norm / swiglu / rope glue", r"rmsnorm|swiglu|rope|_embed|norm|gate"),
            ("copies / casts / index", r"copy|Copy|cast|index|gather|cat|elementwise|fill|reduce")]


def family(name: str) -> str:
    for label, pattern in FAMILIES:
        if re.search(pattern, name, re.I):
            return label
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--widths", type=int, nargs="+", default=[1, 16])
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()
    from tensorfold.families.qwen3_5.cuda.decode import prefill
    from tensorfold.families.qwen3_5.cuda.forward import tree_forward
    from tensorfold.families.qwen3_5.cuda.weights import load

    w = load(args.model, tiled=True)
    prompt = list(range(1000, 1512))
    st, pending = prefill(w, prompt, None)[:2]
    for width in args.widths:
        tokens = torch.full((width,), int(pending), dtype=torch.int32, device="cuda")
        parents = [-1] + list(range(width - 1))
        for _ in range(3):
            tree_forward(w, tokens, parents, st)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.reps):
            tree_forward(w, tokens, parents, st)
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) / args.reps * 1e3
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(args.reps):
                tree_forward(w, tokens, parents, st)
            torch.cuda.synchronize()
        by = defaultdict(float)
        count = defaultdict(int)
        top = defaultdict(float)
        for e in prof.key_averages():
            t = getattr(e, "device_time_total", None) or getattr(e, "cuda_time_total", 0)
            if t <= 0:
                continue
            by[family(e.key)] += t / args.reps / 1e3
            count[family(e.key)] += e.count // args.reps
            top[e.key] += t / args.reps / 1e3
        gpu = sum(by.values())
        print(f"\nwidth {width}: wall {wall:.2f} ms/round, GPU busy {gpu:.2f} ms ({gpu / wall:.0%}), "
              f"{width / wall * 1e3:.1f} rows/s")
        for label, ms in sorted(by.items(), key=lambda kv: -kv[1]):
            print(f"  {label:28} {ms:7.2f} ms  {ms / gpu:5.1%}  {count[label]:5d} launches")
        print("  top kernels:")
        for name, ms in sorted(top.items(), key=lambda kv: -kv[1])[:8]:
            print(f"    {ms:6.2f} ms  {name[:100]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

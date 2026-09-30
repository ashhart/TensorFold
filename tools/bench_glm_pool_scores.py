"""Measure GLM pool-score row reuse on CUDA without loading model weights.

PYTHONPATH=src python tools/bench_glm_pool_scores.py
Uses GLM's real per-rank index shapes, but generated inputs; not an end-to-end
model benchmark. Each candidate must match RB=1 bitwise before timing.
"""
from __future__ import annotations

import importlib.util
import json
import os

import torch
import triton

from tensorfold.families.glm5_next.cuda import sparse


def main():
    torch.manual_seed(127)
    reference = sparse
    if os.environ.get("POOL_REFERENCE"):
        spec = importlib.util.spec_from_file_location("pool_reference", os.environ["POOL_REFERENCE"])
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
    for pos in (2048, 16384, 205000):
        for rows in (1, 3, 6, 8, 16):
            heads, width = 32, 128
            pools = sparse.pool_bucket(pos, rows, 65536)
            qi = torch.randn((rows, heads * width), device="cuda", dtype=torch.bfloat16)
            weights = torch.randn((rows, heads), device="cuda", dtype=torch.bfloat16)
            keys = torch.randn((65538, width), device="cuda", dtype=torch.bfloat16)
            position = torch.tensor([pos], device="cuda", dtype=torch.int32)
            ref = torch.empty((rows, pools), device="cuda", dtype=torch.float32)
            out = torch.empty_like(ref)

            def score(rb, dest, module=sparse, rows=rows, pools=pools, qi=qi, weights=weights,
                      keys=keys, position=position, width=width, heads=heads):
                module._scores[(triton.cdiv(rows, rb), triton.cdiv(pools, 64))](
                    qi, weights, weights.stride(0), keys, dest, position, rows, pools,
                    width ** -0.5, 1.0 / 5.656854249492381, H=heads, HP=heads,
                    D=width, BP=64, RB=rb, num_warps=4)

            score(1, ref, reference)
            baseline = triton.testing.do_bench(lambda ref=ref: score(1, ref, reference), warmup=100, rep=200)
            for rb in (1, 2, 4, 8):
                score(rb, out)
                assert torch.equal(ref, out), (pos, rows, rb)
                ms = triton.testing.do_bench(lambda rb=rb, out=out: score(rb, out), warmup=100, rep=200)
                print(json.dumps({"position": pos, "rows": rows, "pools": pools, "rb": rb,
                                  "ms": ms, "speedup": baseline / ms, "bits_equal": True}), flush=True)


if __name__ == "__main__":
    main()

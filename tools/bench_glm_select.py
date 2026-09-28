"""Time GLM-5.3-Flash's DSA token selection for one prefill chunk deep in a long context: the pool scores kernel
at several row blocks (checking each gives RB=1's bits) and the top-512 ranking.

    python tools/bench_glm_select.py --pos 126976 --rows 2048
"""

from __future__ import annotations

import argparse

import torch
import triton

from tensorfold.families.glm5_next.cuda import sparse


def timed(fn, reps):
    fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / reps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", type=int, default=126976)
    ap.add_argument("--rows", type=int, default=2048)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    dev = "cuda"
    R, H, D = a.rows, 32, 128
    npool_cap = (a.pos + R) // 4 + 8
    torch.manual_seed(0)
    qi = torch.randn((R, H * D), device=dev).to(torch.bfloat16)
    wts = torch.randn((R, H), device=dev).to(torch.bfloat16)
    pk = torch.randn((npool_cap, D), device=dev).to(torch.bfloat16)
    pos_dev = torch.tensor([a.pos], dtype=torch.int32, device=dev)
    np_b = sparse.pool_bucket(a.pos, R, npool_cap)
    print(f"rows {R} at position {a.pos}: {(a.pos + R) // 4} visible pools, bucket {np_b}")
    ref = None
    for rb in (1, 2, 4, 8, 16):
        scores = torch.empty((R, np_b), dtype=torch.float32, device=dev)
        fn = lambda: sparse._scores[(triton.cdiv(R, rb), triton.cdiv(np_b, 64))](
            qi, wts, wts.stride(0), pk, scores, pos_dev, R, np_b, D ** -0.5, 1.0 / 5.656854249492381,
            H=H, HP=32, D=D, BP=64, RB=rb, num_warps=4)
        ms = timed(fn, a.reps)
        if ref is None:
            ref = scores.clone()
        same = torch.equal(scores, ref)
        print(f"scores RB {rb:2d}: {ms:7.2f} ms  {'same bits' if same else 'BITS DIFFER'}")
    ms = timed(lambda: sparse._top_pools(ref, sparse.TOPK_POOLS), a.reps)
    print(f"top 512 pools, torch: {ms:7.2f} ms")
    ms = timed(lambda: sparse.top_pools(ref, sparse.TOPK_POOLS), a.reps)
    same = torch.equal(sparse.top_pools(ref, sparse.TOPK_POOLS), sparse._top_pools(ref, sparse.TOPK_POOLS))
    print(f"top 512 pools, radix select: {ms:7.2f} ms  {'same pools' if same else 'POOLS DIFFER'}")
    ms = timed(lambda: sparse.select_tokens(qi, wts, pk, a.pos, R, npool_cap, pos_dev), a.reps)
    print(f"select_tokens (all): {ms:7.2f} ms")


if __name__ == "__main__":
    main()

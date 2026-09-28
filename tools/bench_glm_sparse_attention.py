"""Time GLM-5.3-Flash's sparse latent attention for one prefill chunk deep in a long context (each row attends
to 2,051 selected tokens), at several tile settings, checking each gives the default's bits.

    python tools/bench_glm_sparse_attention.py --rows 2048 --pos 126976
"""

from __future__ import annotations

import argparse

import torch
import triton

from tensorfold.families.glm5_next.cuda import latent, sparse


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
    ap.add_argument("--rows", type=int, default=2048)
    ap.add_argument("--pos", type=int, default=126976)
    ap.add_argument("--heads", type=int, default=32, help="this rank's heads")
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    dev = "cuda"
    R, H, LW = a.rows, a.heads, latent.L
    torch.manual_seed(0)
    cache = torch.randn((a.pos + R, LW), device=dev).to(torch.bfloat16)
    qa = (torch.randn((R, H, LW), device=dev) * 0.05).to(torch.bfloat16)
    # each row's selection: 512 random pools of those it can see (ascending) plus its incomplete pool
    q = a.pos + torch.arange(R, device=dev)
    npool = (q + 1) // 4
    scores = torch.rand((R, int(npool.max())), device=dev)
    scores = torch.where(torch.arange(scores.shape[1], device=dev)[None, :] < npool[:, None], scores, float("-inf"))
    pools = sparse.top_pools(scores, sparse.TOPK_POOLS)
    W = sparse.TOPK_POOLS * 4 + 3
    tokens = torch.empty((R, W), dtype=torch.int32, device=dev)
    tokens[:, :2048] = (pools[:, :, None] * 4 + torch.arange(4, device=dev)).reshape(R, -1)
    tail = npool[:, None] * 4 + torch.arange(3, device=dev)
    ok = tail <= q[:, None]
    tokens[:, 2048:] = torch.where(ok, tail, -1)
    counts = (2048 + ok.sum(1)).to(torch.int32)
    scale = 0.05
    nch = triton.cdiv(W, latent.CHUNK)
    n = nch * R * H
    po = torch.empty((n * LW,), dtype=torch.float32, device=dev)
    pm = torch.empty((n,), dtype=torch.float32, device=dev)
    pl = torch.empty((n,), dtype=torch.float32, device=dev)
    out = torch.zeros((R, H, LW), dtype=torch.bfloat16, device=dev)
    flops = 4 * R * H * LW * int(counts.float().mean())
    ref = None
    for hb, kt, warps, stages in ((16, 32, 8, 1), (32, 32, 8, 1), (16, 64, 8, 1), (32, 16, 8, 1), (16, 32, 4, 1),
                                  (16, 32, 8, 2), (32, 32, 8, 2)):
        def run():
            latent._sparse_chunks[(R, triton.cdiv(H, hb), nch)](qa, cache, tokens, counts, po, pm, pl, R, W=W, H=H,
                                                                LW=LW, CH=latent.CHUNK, SCALE=scale, HBT=hb, KTT=kt,
                                                                num_warps=warps, num_stages=stages)
            latent._merge[(R, H)](po, pm, pl, out, counts, R, H=H, LW=LW, NCH=nch, SPARSE=True, num_warps=4)
        try:
            ms = timed(run, a.reps)
        except Exception as e:  # noqa: BLE001
            print(f"HB {hb} KT {kt} warps {warps} stages {stages}: {type(e).__name__}")
            continue
        if ref is None:
            ref = out.clone()
        same = torch.equal(out, ref)
        print(f"HB {hb:2d} KT {kt:2d} warps {warps} stages {stages}: {ms:7.2f} ms  {flops / ms / 1e9:5.1f} TFLOP/s  "
              f"{'same bits' if same else 'BITS DIFFER'}")


if __name__ == "__main__":
    main()

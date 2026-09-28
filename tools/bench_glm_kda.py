"""Time one GLM-5.3-Flash KDA layer (this rank's 32 heads) at a prefill chunk's shape: the input projection (4-bit
matmul) against the chain (conv, delta rule, gated norm), random weights and inputs.

    python tools/bench_glm_kda.py --rows 2048
"""

from __future__ import annotations

import argparse

import torch

from tensorfold.families.glm5_next.cuda import kda, qmm


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


def inputs(R: int, H: int, dev: str = "cuda", seed: int = 0):
    """Random chain inputs: projection rows [q | k | v | fa | ga | b], gates, conv and delta-rule states."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    C = 3 * H * kda.DK
    b_off = C + 256
    width = -(-(b_off + H) // 64) * 64
    p = (torch.randn((R, width), generator=g) * 0.5).to(torch.bfloat16).to(dev)
    a = torch.randn((R, H * kda.DK), generator=g).to(torch.bfloat16).to(dev)
    gate = torch.randn((R, H * kda.DV), generator=g).to(torch.bfloat16).to(dev)
    conv_state = (torch.randn((3, C), generator=g) * 0.5).to(torch.bfloat16).to(dev)
    conv_w = (torch.randn((C, 4), generator=g) * 0.5).to(torch.bfloat16).to(dev)
    state = (torch.randn((H, kda.DV, kda.DK), generator=g) * 0.1).to(dev)
    a_log = (torch.rand(H, generator=g) * 2 - 1).to(dev)
    dt_bias = (torch.randn(H * kda.DK, generator=g) * 0.1).to(dev)
    norm_w = (torch.rand(kda.DV, generator=g) + 0.5).to(torch.bfloat16).to(dev)
    return p, b_off, a, gate, conv_state, conv_w, state, a_log, dt_bias, norm_w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=2048)
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    R, H = a.rows, a.heads
    p, b_off, ga, gate, cs, cw, state, a_log, dt_bias, norm_w = inputs(R, H)
    scratch = kda.KDAScratch(R, H, "cuda")
    out_state = torch.empty_like(state)
    for wide in (False, True):
        ms = timed(lambda: kda.chain(p, b_off, ga, gate, cs, cw, state, a_log, dt_bias, norm_w, 1e-5, -5.0, R,
                                     scratch, out_state, wide=wide), a.reps)
        print(f"chain ({'three kernels' if wide else 'fused'}), {R} rows x {H} heads: {ms:7.2f} ms  "
              f"({ms * 1e3 / R:.2f} us a row)")
    x = torch.randn((R, 4096), device="cuda").to(torch.bfloat16)
    w = qmm.quantize4(torch.randn((p.shape[1], 4096), device="cuda").to(torch.bfloat16) * 0.02)
    xs = qmm.group_sums(x)
    ms = timed(lambda: qmm.matmul(x, w, xs, out=p), a.reps)
    print(f"projection 4096 -> {p.shape[1]}: {ms:7.2f} ms  ({2 * R * 4096 * p.shape[1] / ms / 1e9:.1f} TFLOP/s)")


if __name__ == "__main__":
    main()


def profile_kernels(rows: int = 2048, heads: int = 32) -> None:
    """Each kernel's share of a long window's chain (torch profiler)."""
    from torch.profiler import ProfilerActivity, profile

    p, b_off, ga, gate, cs, cw, state, a_log, dt_bias, norm_w = inputs(rows, heads)
    scratch = kda.KDAScratch(rows, heads, "cuda")
    out_state = torch.empty_like(state)
    run = lambda: kda.chain(p, b_off, ga, gate, cs, cw, state, a_log, dt_bias, norm_w, 1e-5, -5.0, rows, scratch,
                            out_state, wide=True)
    run()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            run()
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=6))

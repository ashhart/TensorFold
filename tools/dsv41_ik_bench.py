"""Microbenchmark of the fp4 decode / verify indexer (kernels.index_scores over MXFP4 keys): the per-row kernel
(_index_scores, TF_DSV41_SHARED_IK=0) against the shared-tile kernel (_index_scores_tile), for rows of several
streams at full visibility on a synthetic pool (random nibbles, ue8m0 scales near 2^0). One GPU, no model:

    python tools/dsv41_ik_bench.py [--keys 16384,65536,153600,262144,614400] [--iters 50]

Each case: R rows as S streams x R/S rows (a stream's rows at consecutive positions, as a verify round), each stream
with n_keys keys at its own base, every key visible to every row. Prints a markdown table (us per call, old / new)
and checks torch.equal between the two on each case.
"""

from __future__ import annotations

import argparse

import torch

from tensorfold.families.deepseek_v41.cuda import kernels as K

CASES = [(16, 16), (16, 4), (16, 1), (4, 1), (1, 1), (8, 8), (32, 8)]   # (rows, streams)


def pool(streams: int, n: int, seed: int) -> K.Fp4Rows:
    g = torch.Generator(device="cuda").manual_seed(seed)
    keys = K.Fp4Rows(streams * n, 128, group=32, scale="ue8m0", device="cuda")
    keys.q.copy_(torch.randint(0, 256, keys.q.shape, generator=g, device="cuda", dtype=torch.uint8))
    keys.s.copy_(torch.randint(124, 130, keys.s.shape, generator=g, device="cuda", dtype=torch.uint8))
    return keys


def timed(fn, iters: int) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return 1e3 * a.elapsed_time(b) / iters


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", default="16384,65536,153600,262144,614400")
    ap.add_argument("--cases", default=",".join(f"{r}x{s}" for r, s in CASES), help="ROWSxSTREAMS,...")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--ratio", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=3)
    a = ap.parse_args()
    cases = [tuple(int(v) for v in c.split("x")) for c in a.cases.split(",")]
    print(f"programs target {K.SHARED_IK_PROGRAMS}, ratio {a.ratio}")
    print("| keys a stream | rows x streams | per-row us | shared-tile us | new / old | equal |")
    print("|---:|---:|---:|---:|---:|:---:|")
    for n in [int(v) for v in a.keys.split(",")]:
        for R, S in cases:
            keys = pool(S, n, n + S)
            g = torch.Generator(device="cuda").manual_seed(R * 7 + S)
            iq = (torch.randn((R, 64, 128), generator=g, device="cuda") * 0.3).to(torch.bfloat16)
            wts = torch.randn((R, 64), generator=g, device="cuda")
            per = R // S
            stream = torch.arange(R, device="cuda") // per
            kbase = stream * n
            # full visibility: (pos + 1) // ratio == n for the stream's last row, the earlier rows one position less
            # each (a verify window, still every key visible up to its own bound)
            off = per - 1 - (torch.arange(R, device="cuda") % per)
            pos = (n * a.ratio - 1 - off).long()
            old = lambda: K.index_scores(iq, wts, keys, pos, a.ratio, kbase=kbase, n_keys=n, shared=False)
            new = lambda: K.index_scores(iq, wts, keys, pos, a.ratio, kbase=kbase, n_keys=n, shared=True)
            eq = torch.equal(old(), new())
            # interleaved, best of rounds (the GPU clock drifts over a run)
            to = tn = float("inf")
            for _ in range(a.rounds):
                tn = min(tn, timed(new, a.iters))
                to = min(to, timed(old, a.iters))
            print(f"| {n} | {R} x {S} | {to:.0f} | {tn:.0f} | {tn / to:.2f} | {'yes' if eq else 'NO'} |", flush=True)
            del keys
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

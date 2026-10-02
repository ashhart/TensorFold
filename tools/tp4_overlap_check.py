#!/usr/bin/env python3
"""Can an NCCL all-reduce on a side stream overlap compute on the main stream on GB10? Times a bf16 matmul loop
alone, a prompt-sized all-reduce (24 MB) alone, and both together, for a few NCCL settings passed in the env.

Run on every rank (rank 0 last): python tools/tp4_overlap_check.py --rank R --master ADDR
"""
import argparse
import os
import time

import torch

from tensorfold.cuda.comm import NCCL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--master", required=True)
    a = ap.parse_args()
    torch.cuda.set_device(0)
    nccl = NCCL(a.rank, 4, a.master, 29571)
    side = torch.cuda.Stream()
    A = torch.randn(4096, 6144, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(6144, 6144, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(2048, 6144, device="cuda", dtype=torch.bfloat16)
    y = torch.empty_like(x)
    n_mm, n_ar = 20, 10

    def compute():
        for _ in range(n_mm):
            A @ B

    def comm():
        with torch.cuda.stream(side):
            for _ in range(n_ar):
                nccl.all_reduce(x, y)

    def t(fn):
        nccl.barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1e3

    for _ in range(2):
        t(compute), t(comm)
    tc, tr = t(compute), t(comm)
    tb = t(lambda: (comm(), compute()))
    if a.rank == 0:
        env = {k: v for k, v in os.environ.items() if k.startswith("NCCL_") and k not in
               ("NCCL_SOCKET_IFNAME", "NCCL_IB_HCA", "NCCL_IB_GID_INDEX", "NCCL_DEBUG")}
        print(f"[overlap] {env}: compute {tc:.1f} ms, all-reduce {tr:.1f} ms, both {tb:.1f} ms "
              f"(serial would be {tc + tr:.1f}; overlap hides {tc + tr - tb:.1f} ms)", flush=True)


if __name__ == "__main__":
    main()

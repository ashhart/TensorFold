"""All-gather bandwidth between the two nodes at prompt-chunk sizes: dsv41_nccl_bench.py --rank R --master IP."""

import argparse
import time

import torch

from tensorfold.cuda.comm import NCCL

ap = argparse.ArgumentParser()
ap.add_argument("--rank", type=int, required=True)
ap.add_argument("--master", required=True)
ap.add_argument("--port", type=int, default=29533)
a = ap.parse_args()
torch.cuda.set_device(0)
nccl = NCCL(a.rank, 2, a.master, a.port)
nccl.barrier()
for rows in (128, 512, 1024, 2048):
    n = rows * 5120
    send = torch.randn(n, device="cuda").to(torch.bfloat16)
    recv = torch.empty(2 * n, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        nccl.all_gather(send, recv)
    nccl.barrier()
    t0 = time.perf_counter()
    for _ in range(20):
        nccl.all_gather(send, recv)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / 20 * 1e3
    if a.rank == 0:
        print(f"{rows} rows bf16 ({n * 2 / 1e6:.1f} MB a rank): {ms:.2f} ms, {n * 2 / ms / 1e6:.1f} GB/s", flush=True)

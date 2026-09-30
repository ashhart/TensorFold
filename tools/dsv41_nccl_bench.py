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
for rows, dt in ((1, torch.float32), (2, torch.float32), (4, torch.float32), (8, torch.float32), (16, torch.float32), (128, torch.bfloat16),
                 (512, torch.bfloat16), (1024, torch.bfloat16), (2048, torch.bfloat16)):
    n = rows * 5120
    send = torch.randn(n, device="cuda").to(dt)
    recv = torch.empty(2 * n, device="cuda", dtype=dt)
    for _ in range(3):
        nccl.all_gather(send, recv)
    nccl.barrier()
    t0 = time.perf_counter()
    for _ in range(200):
        nccl.all_gather(send, recv)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / 200 * 1e3
    if a.rank == 0:
        b = n * send.element_size()
        print(f"{rows} rows {str(dt)[6:]} ({b / 1e6:.2f} MB a rank): {ms * 1e3:.0f} us, {b / ms / 1e6:.1f} GB/s", flush=True)

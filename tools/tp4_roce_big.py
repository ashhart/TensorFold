#!/usr/bin/env python3
"""Prompt-sized reductions on the four-Spark fabric: NCCL ring all-reduce vs the b12x RoCE one-shot all-reduce,
bf16 [rows, 6144] for rows 512..4096, and whether the RoCE sums match on every rank."""
import argparse
import os
import time

import torch

from tensorfold.cuda.comm import NCCL


def timed(fn, iters=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--master", required=True)
    a = ap.parse_args()
    torch.cuda.set_device(0)
    nccl = NCCL(a.rank, 4, a.master, 29563)
    import torch.distributed as dist
    from datetime import timedelta
    os.environ.setdefault("GLOO_SOCKET_IFNAME", os.environ.get("NCCL_SOCKET_IFNAME", ""))
    dist.init_process_group("gloo", init_method=f"tcp://{a.master}:29575", rank=a.rank, world_size=4,
                            timeout=timedelta(seconds=120))
    from b12x.comm.roce import AllReduce
    rt = AllReduce(exchange_group=dist.group.WORLD, device=torch.device("cuda", 0), max_size=64 << 20,
                   max_gather_bytes=1 << 20, hca_names=[os.environ["NCCL_IB_HCA"]],
                   gid_index=int(os.environ["NCCL_IB_GID_INDEX"]))
    rt.prepare((torch.bfloat16,))
    lines = []
    for rows in (512, 1024, 2048, 4096):
        x = torch.randn(rows, 6144, device="cuda", dtype=torch.bfloat16)
        y = torch.empty_like(x)
        t_n = timed(lambda: nccl.all_reduce(x, y))
        z = torch.empty_like(x)
        t_r = timed(lambda: rt.all_reduce(x, out=z))
        mb = rows * 6144 * 2 / 2**20
        lines.append(f"rows {rows:5d} ({mb:5.1f} MB bf16): NCCL ring {t_n:7.2f} ms   RoCE one-shot {t_r:7.2f} ms")
    got = torch.empty((4 * z.numel(),), dtype=torch.bfloat16, device="cuda")
    nccl.all_gather(z.reshape(-1), got)
    same = all(torch.equal(got.view(4, -1)[r], got.view(4, -1)[0]) for r in range(4))
    nccl.barrier()
    if a.rank == 0:
        for ln in lines:
            print("[roce-big] " + ln, flush=True)
        print(f"[roce-big] RoCE sums identical on all ranks: {same}", flush=True)


if __name__ == "__main__":
    main()

"""Two-rank check and timing of the RoCE all-gather against NCCL: rdma_bench.py --rank R --master IP."""

import argparse
import time

import torch

from tensorfold.cuda.comm import NCCL
from tensorfold.cuda.rdma import RdmaComm

ap = argparse.ArgumentParser()
ap.add_argument("--rank", type=int, required=True)
ap.add_argument("--master", required=True)
ap.add_argument("--port", type=int, default=29541)
a = ap.parse_args()
torch.cuda.set_device(0)
nccl = NCCL(a.rank, 2, a.master, a.port)
nccl.barrier()
rc = RdmaComm(nccl, a.rank)
rc.settle()
g = torch.Generator(device="cuda").manual_seed(a.rank + 1)
cases = [(rows * 5120, torch.float32) for rows in (1, 2, 3, 4, 8, 16, 32)] + [(3168 * 4, torch.uint8), (8192, torch.bfloat16)]
for n, dt in cases:
    send = (torch.randn(n, generator=g, device="cuda") * 3).to(dt) if dt != torch.uint8 else \
        torch.randint(0, 255, (n,), generator=g, device="cuda", dtype=torch.uint8)
    ref = torch.empty(2 * n, dtype=dt, device="cuda")
    got = torch.empty_like(ref)
    nccl.all_gather(send, ref)
    rc.all_gather(send, got)
    eager_ok = torch.equal(ref, got)
    # a graph of 20 gathers, replayed 3 times (the device epoch carries the sequence across replays)
    outs = [torch.empty_like(ref) for _ in range(20)]
    gr = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.cuda.graph(gr):
        for o in outs:
            rc.all_gather(send, o)
    torch.cuda.current_stream().wait_stream(s)
    for _ in range(3):
        gr.replay()
    torch.cuda.synchronize()
    rc.check()
    graph_ok = all(torch.equal(ref, o) for o in outs)
    times = {}
    for name, fn in (("nccl", lambda s=send, r=ref: nccl.all_gather(s, r)), ("rdma", lambda s=send, r=got: rc.all_gather(s, r))):
        for _ in range(20):
            fn()
        nccl.barrier()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(300):
            fn()
        torch.cuda.synchronize()
        times[name] = (time.perf_counter() - t) / 300 * 1e6
    nccl.barrier()
    t = time.perf_counter()
    for _ in range(20):
        gr.replay()
    torch.cuda.synchronize()
    graph_us = (time.perf_counter() - t) / 400 * 1e6
    rc.check()
    if a.rank == 0:
        print(f"{n * send.element_size():8d} B {str(dt)[6:]:9s}: equal eager {eager_ok} graph {graph_ok}; "
              f"nccl {times['nccl']:6.1f} us, rdma {times['rdma']:6.1f} us, rdma in a graph {graph_us:6.1f} us", flush=True)
nccl.barrier()
rc.rdma.close()

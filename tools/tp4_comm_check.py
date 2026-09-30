#!/usr/bin/env python3
"""Four-machine check of TensorFold's NCCL collectives (full GLM-5.3, milestone M0): latency of the fp32 all-gather at
decode sizes and prompt-chunk sizes, the all-to-all, and the exact reduce-scatter's bits against the all-gather sum.

Run on every rank (rank 0 last):  python tools/tp4_comm_check.py --rank R --master ADDR [--world 4]
"""
from __future__ import annotations

import argparse
import time

import torch

from tensorfold.cuda.comm import NCCL

HIDDEN = 6144


def timed(fn, iters: int) -> float:
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world", type=int, default=4)
    ap.add_argument("--master", required=True)
    ap.add_argument("--port", type=int, default=29561)
    a = ap.parse_args()
    torch.cuda.set_device(0)
    comm = NCCL(a.rank, a.world, a.master, a.port)
    w = a.world
    g = torch.Generator(device="cpu").manual_seed(100 + a.rank)

    # 1. bits: exact reduce-scatter == all-gather + rank-order sum, at a prompt-chunk size
    rows = 2048
    part = torch.randn(rows, HIDDEN, generator=g).mul_(10.0 ** (a.rank - 1)).cuda()
    recv = torch.empty((w, rows, HIDDEN), dtype=torch.float32, device="cuda")
    comm.all_gather(part, recv)
    ref = recv[0].clone()
    for r in range(1, w):
        ref += recv[r]
    q = HIDDEN // w
    send = part.view(rows, w, q).permute(1, 0, 2).contiguous().view(w, rows * q)
    got = torch.empty_like(send)
    comm.all_to_all(send, got)
    mine = got[0].clone()
    for r in range(1, w):
        mine += got[r]
    gathered = torch.empty((w, rows * q), dtype=torch.float32, device="cuda")
    comm.all_gather(mine, gathered)
    rs = gathered.view(w, rows, q).permute(1, 0, 2).reshape(rows, HIDDEN)
    same = bool(torch.equal(rs, ref))

    # 2. latency
    lines = []
    for rows in (1, 3, 8, 16):                          # decode windows: fp32 [rows, 6144] partials
        x = torch.randn(rows, HIDDEN, device="cuda")
        out = torch.empty((w, rows, HIDDEN), device="cuda")
        us = timed(lambda: comm.all_gather(x, out), 300)
        lines.append(f"all_gather fp32 rows {rows:5d} ({rows * HIDDEN * 4 / 1024:7.0f} KB): {us:8.1f} us")
    for rows in (512, 2048):                            # prompt chunks: all-gather vs exact reduce-scatter
        x = torch.randn(rows, HIDDEN, device="cuda")
        out = torch.empty((w, rows, HIDDEN), device="cuda")
        us_ag = timed(lambda: comm.all_gather(x, out), 20)
        s = x.view(rows, w, q).permute(1, 0, 2).contiguous().view(w, rows * q)
        r2 = torch.empty_like(s)
        m = torch.empty((rows * q,), device="cuda")
        gg = torch.empty((w, rows * q), device="cuda")

        def rs_step():
            comm.all_to_all(s, r2)
            m.copy_(r2[0])
            comm.all_gather(m, gg)
        us_rs = timed(rs_step, 20)
        lines.append(f"rows {rows:5d} ({rows * HIDDEN * 4 / 2**20:5.1f} MB): all_gather {us_ag / 1e3:7.2f} ms  "
                     f"exact reduce-scatter {us_rs / 1e3:7.2f} ms")
    comm.barrier()
    if a.rank == 0:
        print(f"[tp4-comm] world {w}: exact reduce-scatter bits == all-gather rank-order sum: {same}")
        for ln in lines:
            print("[tp4-comm] " + ln)
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())

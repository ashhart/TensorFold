# SPDX-License-Identifier: MIT

# Copyright (c) 2026 Jay Leaton. The DeepSeek-V4.1-Flash family of TensorFold (Apache-2.0): see THIRD_PARTY_NOTICES.md.
"""CPU check of the INDEX_BOUND top-k twins against the served kernels' models on random index-score rows."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def plan(nk: int, cl: int | None = None) -> tuple[int, int]:
    """block.zig's topkPlan (attn_cuda.plan): cluster CTAs, odd entries a thread."""

    cl = cl or max(1, min(8, -(-nk // 4096)))
    ept = max(1, -(-nk // (cl * 512)))
    return cl, ept + 1 - ept % 2


def rows(np, rng, n: int, nk_pos: int, ratio: int, count: int):
    """(scores [count, nk] fp32, positions [count]) as `_scores` writes them for a bucket ending at nk_pos."""

    nk = max((nk_pos + 1) // ratio, 1)
    q = np.concatenate([[0, 1, 300, 1023, 1500, nk_pos], rng.integers(0, nk_pos + 1, count - 6)])[:count]
    s = np.full((count, nk), -np.inf, np.float32)
    for r in range(count):
        nv = (int(q[r]) + 1) // ratio
        kind = r % 5
        if kind == 0:
            v = np.floor(rng.random(nv) * 8) / 8
            v[rng.random(nv) < 0.05] = 0.0
            v[rng.random(nv) < 0.05] = -0.0
        elif kind == 1:
            v = rng.random(nv) * 4 - 1
        elif kind == 2:
            v = rng.random(nv) * 4 - 1
            z = rng.random(nv)
            v[z < 0.02] = -np.inf
            v[(z >= 0.02) & (z < 0.03)] = np.inf
            v[(z >= 0.03) & (z < 0.04)] = np.nan
        elif kind == 3:
            v = np.full(nv, 0.5)
        else:
            v = np.full(nv, -np.inf)                             # a visible part all -inf: ties with the padding
            v[rng.random(nv) < 0.01] = 1.0
        s[r, :nv] = v.astype(np.float32)
    return s, q.astype(np.int64)


def cuda_model(np, emu, s, q, ratio: int, k: int, mode: int) -> int:
    """attn_cuda_emu.topk_row on every row: full images vs topk_b_kernel's (absent past B). Returns mismatches."""

    bad = 0
    nk = -(-s.shape[1] // 8) if mode == 2 else s.shape[1]
    cl, ept = plan(nk)
    for r in range(s.shape[0]):
        nvis = (int(q[r]) + 1) // ratio
        u, p = emu.entries(mode, s[r], None, nvis, 8)
        bnd = min(nk, max(nvis, k)) if mode == 0 else min(nk, max(0, -(-nvis // 8)))
        ub = u.copy()
        ub[bnd:] = 0
        want = emu.topk_row(u, p, k, nvis, cl, ept)
        got = emu.topk_row(ub, p, k, nvis, cl, ept)
        if want != got:
            bad += 1
            print(f"  cuda mode {mode} row {r} (q {q[r]}, nvis {nvis}, B {bnd}): {len(got[0])} / {got[1]} vs {len(want[0])} / {want[1]}")
    return bad


def triton_run(torch, dt, db, s, q, ratio: int, k: int, mode: int, rows_mode: bool) -> int:
    nk = -(-s.shape[1] // 8) if mode == 2 else s.shape[1]
    R = s.shape[0]
    st = torch.from_numpy(s)
    pos = torch.from_numpy(q) if rows_mode else torch.tensor([int(q[0])], dtype=torch.int32)
    outs = []
    for fn in (dt._dtopk, db._dtopk_b):
        out = torch.full((R, k), -7, dtype=torch.int32)
        cnt = torch.full((R,), -7, dtype=torch.int32)
        fn[(R,)](st, st.stride(0), st, 0, pos, nk, out, out.stride(0), cnt if mode == 0 else out, RATIO=ratio, K=k,
                 MODE=mode, ROWS=rows_mode, T=4096, BS=8, RB=8, NPASS=8, SORT=False, COUNT=mode == 0,
                 KP=1 << (k - 1).bit_length(), num_warps=8)
        outs.append((out, cnt))
    (o0, c0), (o1, c1) = outs
    bad = int(not torch.equal(o0, o1)) + int(not torch.equal(c0, c1))
    if bad:
        rr = [r for r in range(R) if not torch.equal(o0[r], o1[r]) or c0[r] != c1[r]]
        print(f"  triton mode {mode} rows {rows_mode}: rows differ {rr[:8]}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="the served TensorFold src (csa2's package root)")
    ap.add_argument("--rows", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    os.environ["TRITON_INTERPRET"] = "1"
    sys.path[:0] = [a.src, str(HERE)]
    import numpy as np
    import torch
    from tensorfold.families.deepseek_v41.cuda.csa2 import attn_cuda_emu as emu
    from tensorfold.families.deepseek_v41.cuda.csa2 import dtopk as dt
    from dsv41_zig_triton import dtopk_b as db

    rng = np.random.default_rng(a.seed)
    bad = 0
    # the CUDA path: ratio 2 select at 40K (NK 20K), ratio 1 select + blocks at 30K (NK 30K, 3,750 blocks), K 512 / 2048
    for name, end, ratio, k, mode in (("cuda select r2", 40_000, 2, 512, 0), ("cuda select r1", 30_000, 1, 512, 0),
                                      ("cuda blocks r1", 30_000, 1, 2048, 2), ("cuda blocks r1 short", 9_000, 1, 2048, 2)):
        s, q = rows(np, rng, a.rows, end, ratio, a.rows)
        b = cuda_model(np, emu, s, q, ratio, k, mode)
        print(f"{name}: {'ok' if b == 0 else f'FAIL ({b} rows)'} ({a.rows} rows)")
        bad += b
    # the Triton path (interpreted): select and blocks, row mode and one slot (one slot: consecutive positions)
    for name, end, ratio, k, mode in (("_dtopk select r1", 20_000, 1, 512, 0), ("_dtopk select r2", 30_000, 2, 512, 0),
                                      ("_dtopk blocks r1", 70_000, 1, 2048, 2)):
        for rows_mode in (True, False):
            n = 12
            s, q = rows(np, rng, n, end, ratio, n)
            if not rows_mode:
                q = np.arange(end - 400, end - 400 + n, dtype=np.int64)
                s, _ = rows(np, rng, n, end, ratio, n)
                for r in range(n):                               # rows at q: -inf past their own nvis
                    s[r, (int(q[r]) + 1) // ratio:] = -np.inf
            b = triton_run(torch, dt, db, s, q, ratio, k, mode, rows_mode)
            print(f"{name} {'rows' if rows_mode else 'one slot'}: {'ok' if b == 0 else 'FAIL'} ({n} rows)")
            bad += b
    print("PASS" if bad == 0 else f"FAIL ({bad})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jay Leaton. The DeepSeek-V4.1-Flash family of TensorFold (Apache-2.0): see THIRD_PARTY_NOTICES.md.
# Modified for TensorFold dsv41-cuda: from expert_loads.py, without the PDL launch and probe knobs; TF_EXPERT_LOADS.
"""The routed experts' load path (``x3ld.cu``): upstream's grouped EXL3 expert kernel with 16-byte, several-deep
weight loads, the same Z (each output element's warp K range, mma chain and warp sum order are upstream's; only when
the bytes arrive changes).

``TF_EXPERT_LOADS``          routed layers' two grouped launches (gate/up, down) run ``ld_kernel`` where it fits
                             (default 1; 0: upstream's kernel).
``TF_EXPERT_LOADS_CFG``      "nt,pd" for both or "nt,pd/nt,pd" (gate/up, down), default "4,2" (DeepSeek-V4.1 on
                             GB10: 16 clients 122.8 -> 128.2 tok/s, serial +2.7%); nt column tiles a program, pd k
                             steps in flight a warp. Every setting places work only.
"""

from __future__ import annotations

import os
from functools import lru_cache

CFGS = ((8, 1), (8, 2), (4, 2))
WARPS = 4
RANGES = ((8, 8), (2, 10))              # the half-bit width ranges the extension instantiates


def _cfg(text: str) -> tuple[int, int]:
    nt, pd = (int(p) for p in text.split(","))
    if (nt, pd) not in CFGS:
        raise ValueError(f"TF_EXPERT_LOADS_CFG: {text!r} is not one of {CFGS}")
    return nt, pd


def _parse() -> dict:
    raw = os.environ.get("TF_EXPERT_LOADS_CFG", "").strip()
    halves = raw.split("/") if raw else ["4,2"]
    gu = _cfg(halves[0])
    return {"on": os.environ.get("TF_EXPERT_LOADS", "1") != "0", "gu": gu,
            "dn": _cfg(halves[1]) if len(halves) > 1 else gu}


CFG = _parse()


def fits(K: int, N: int, sk: int, warps: int, cfg: tuple[int, int]) -> bool:
    nt, pd = cfg
    if warps != WARPS or K % (16 * sk * warps) or N % (16 * nt):
        return False
    per_warp = K // (16 * sk * warps)
    return per_warp >= pd and per_warp % pd == 0


def k2_range(lo: int, hi: int) -> tuple[int, int] | None:
    return next(((a, b) for a, b in RANGES if a <= lo and hi <= b), None)


@lru_cache(maxsize=1)
def _ext():
    from pathlib import Path

    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    return load(name="tf_x3ld_v1", sources=[str(here / "x3ld.cpp"), str(here / "x3ld.cu")],
                extra_include_paths=[str(here)], extra_cuda_cflags=["-O3", "-lineinfo"], verbose=False)


def grouped(x0, x1, tp0, tp1, k2_0, k2_1, ids, count, members, z, mats: int, K: int, N: int, P: int, sk: int,
            slots: int, cb: int, warps: int, lo: int, hi: int, cfg: tuple[int, int] | None = None) -> bool:
    """Upstream's ``ext.grouped`` through ``ld_kernel`` (the same Z); False when not taken (off, or a shape, width
    or codebook it does not cover): the caller runs upstream's kernel."""

    if cfg is None:
        if not CFG["on"]:
            return False
        cfg = CFG["dn" if mats == 1 else "gu"]
    rng = k2_range(lo, hi)
    if rng is None or cb != 2 or not fits(K, N, sk, warps, cfg):
        return False
    _ext().grouped(x0, x1, tp0, tp1, k2_0, k2_1, ids, count, members, z, mats, K, N, P, sk, slots, cb, cfg[0],
                   cfg[1], 0, rng[0], rng[1], False)
    return True

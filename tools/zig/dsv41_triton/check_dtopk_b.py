# SPDX-License-Identifier: MIT

# Copyright (c) 2026 Jay Leaton. The DeepSeek-V4.1-Flash family of TensorFold (Apache-2.0): see THIRD_PARTY_NOTICES.md.
"""Offline check that `_dtopk_b` compiles to `_dtopk`'s layouts, tile-loop bodies, radix pass and reduction counts."""

from __future__ import annotations

import argparse
import collections
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

LOC = re.compile(r"\s*loc\([^()]*(?:\([^()]*\)[^()]*)*\)")
SSA = re.compile(r"%[A-Za-z0-9_#]+")
BLOCKED = re.compile(r"#blocked\d* = (#ttg\.blocked<[^>]*>)")
OPS = re.compile(r"(?:= |^\s*)\"?((?:tt|ttg|arith|scf|math)\.[a-z_.]+)")


def region(lines: list[str], head: int) -> list[str]:
    """The body of the `{` region opened on line `head` (to its matching close), without the head line."""

    depth, out = 0, []
    for i in range(head, len(lines)):
        depth += lines[i].count("{") - lines[i].count("}")
        if i > head:
            out.append(lines[i])
        if depth == 0:
            return out
    raise ValueError("unbalanced region")


TILE_FOR = re.compile(r"(scf\.for %t0 = %[\w#]+ to )(%[\w#]+)")


def norm(body: list[str]) -> list[str]:
    """Locations dropped, a nested tile loop's upper bound named BOUND, SSA names numbered by first use."""

    names: dict[str, str] = {}
    body = [TILE_FOR.sub(r"\1%BOUND", ln) for ln in body]

    def rename(m: re.Match) -> str:
        return names.setdefault(m.group(0), f"%v{len(names)}")

    return [SSA.sub(rename, LOC.sub("", ln)).rstrip() for ln in body if ln.strip()]


def loops(ttgir: str, var: str) -> list[list[str]]:
    lines = ttgir.splitlines()
    return [region(lines, i) for i, ln in enumerate(lines) if re.search(rf"scf\.for {re.escape(var)} = ", ln)]


def opcount(lines: list[str]) -> collections.Counter:
    return collections.Counter(m.group(1) for ln in lines for m in [OPS.search(ln)] if m)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="the served TensorFold src (csa2/dtopk.py's package root)")
    a = ap.parse_args()
    sys.path[:0] = [a.src, str(HERE)]
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from tensorfold.families.deepseek_v41.cuda.csa2 import dtopk as dt
    from dsv41_zig_triton import dtopk_b as db

    tgt = GPUTarget("cuda", 121, 32)

    def comp(fn, sig, consts):
        sig = dict(sig, **{k: "constexpr" for k in consts})
        k = triton.compile(ASTSource(fn=fn, signature=sig, constexprs=consts), target=tgt, options=dict(num_warps=8))
        return k.asm["ttgir"]

    sig = {"S": "*fp32", "s_stride": "i32", "P": "*fp32", "p_stride": "i32", "POS": "*i32", "NK": "i32",
           "OUT": "*i32", "o_stride": "i32", "CNT": "*i32"}
    base = dict(T=4096, BS=8, RB=8, NPASS=8, SORT=False)
    cases = []
    for rows in (False, True):
        s = dict(sig, POS="*i64") if rows else sig
        tag = "rows" if rows else "one slot"
        for ratio in (1, 2):
            cases.append((f"select r{ratio} {tag}", s, dict(base, RATIO=ratio, K=512, MODE=0, ROWS=rows, COUNT=True, KP=512)))
        cases.append((f"blocks r1 {tag}", s, dict(base, RATIO=1, K=2048, MODE=2, ROWS=rows, COUNT=False, KP=2048)))
    bad = 0
    for name, s, consts in cases:
        rt = comp(dt._dtopk, s, consts)
        bt = comp(db._dtopk_b, s, consts)
        rl, bl = loops(rt, "%t0"), loops(bt, "%t0")
        rp, bp = loops(rt, "%t"), loops(bt, "%t")
        checks = {
            "layouts": (set(BLOCKED.findall(bt)), set(BLOCKED.findall(rt))),
            "tile loops": (len(bl), len(rl)),
        }
        for i, (x, y) in enumerate(zip(bl, rl)):
            checks[f"tile loop {i} body"] = (norm(x), norm(y))
        for op in ("tt.reduce", "tt.scan", "tt.histogram", "tt.sort"):
            checks[f"{op} count"] = (bt.count(op + '"') + bt.count(op + " "), rt.count(op + '"') + rt.count(op + " "))
        if consts["MODE"] == 2:
            extra = opcount(bp[0]) - opcount(rp[0])
            missing = opcount(rp[0]) - opcount(bp[0])
            print(f"{name} radix pass: {sum(extra.values())} ops added (the skipped blocks' count) {dict(extra)}")
            checks["radix pass keeps every op"] = (dict(missing), {})
        else:
            checks["radix pass"] = (norm(bp[0]) if bp else None, norm(rp[0]) if rp else None)
        for what, (got, want) in checks.items():
            ok = got == want
            bad += not ok
            if ok:
                print(f"{name} {what}: ok")
            elif isinstance(got, list) and isinstance(want, list):
                d = next((i for i, (x, y) in enumerate(zip(got, want)) if x != y), min(len(got), len(want)))
                print(f"{name} {what}: FAIL at line {d}: {got[d] if d < len(got) else None!r} != {want[d] if d < len(want) else None!r}")
            else:
                print(f"{name} {what}: FAIL {got} != {want}")
    print("PASS" if bad == 0 else f"FAIL ({bad})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

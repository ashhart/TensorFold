# SPDX-License-Identifier: MIT

# Copyright (c) 2026 Jay Leaton. The DeepSeek-V4.1-Flash family of TensorFold (Apache-2.0): see THIRD_PARTY_NOTICES.md.
"""Offline check that `_scores_b` keeps `_scores`' MMA layouts, dot count, float ops and head-sum shuffle offsets."""

from __future__ import annotations

import argparse
import collections
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

FLOAT_OPS = re.compile(r"^\s*(mma\.sync\S*|shfl\.sync\.bfly\.b32|fma\.rn\.f32|add\.f32|add\.rn\.f32|mul\.f32|mul\.rn\.f32|max\.f32|max\.NaN\.f32)",
                       re.M)
SHFL = re.compile(r"shfl\.sync\.bfly\.b32\s+[^,]+,\s*[^,]+,\s*(\d+)")
MMA = re.compile(r"#mma\d* = (#ttg\.nvidia_mma<[^>]*>)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="the served TensorFold src (csa2/index.py's package root)")
    a = ap.parse_args()
    sys.path[:0] = [a.src, str(HERE)]
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from tensorfold.families.deepseek_v41.cuda.csa2 import index as ix
    from dsv41_zig_triton import scores_b as sb

    tgt = GPUTarget("cuda", 121, 32)
    base = dict(H=32, D=128, BP=64, WS=1.0 / math.sqrt(32), SCALE=1.0 / math.sqrt(128), KFP8=True)

    def comp(fn, sig, consts):
        sig = dict(sig, **{k: "constexpr" for k in consts})
        k = triton.compile(ASTSource(fn=fn, signature=sig, constexprs=consts), target=tgt,
                           options=dict(num_warps=8, num_stages=3, enable_fp_fusion=True))
        return k.asm["ttgir"], k.asm["ptx"]

    sig = {"QI": "*bf16", "W": "*bf16", "w_stride": "i32", "IK": "*u8", "OUT": "*fp32", "POS": "*i32", "KEYS": "*fp32",
           "k_stride": "i32", "NK": "i32", "o_stride": "i32", "PT": "*i32", "PTS": "i32"}
    cases = []
    for ratio, psh in ((2, 7), (4, 8)):
        cases.append((f"dense r{ratio}", sig, dict(base, RATIO=ratio, PSH=psh, GATHER=False, SL=None, ROWS=False, CBS=0)))
        cases.append((f"rows r{ratio}", dict(sig, SL="*i32"), dict(base, RATIO=ratio, PSH=psh, GATHER=False, ROWS=True, CBS=0)))
    cases.append(("gather cbs8", dict(sig, KEYS="*i32"), dict(base, RATIO=1, PSH=8, GATHER=True, SL=None, ROWS=False, CBS=8)))
    bad = 0
    for name, s, consts in cases:
        rt, rp = comp(ix._scores, s, consts)
        bt, bp = comp(sb._scores_b, s, consts)
        checks = {
            "mma layout": (set(MMA.findall(bt)), set(MMA.findall(rt))),
            "dots": (bt.count("tt.dot "), rt.count("tt.dot ")),
            "float ops": (collections.Counter(FLOAT_OPS.findall(bp)), collections.Counter(FLOAT_OPS.findall(rp))),
            "shuffle offsets": (collections.Counter(SHFL.findall(bp)), collections.Counter(SHFL.findall(rp))),
        }
        for what, (got, want) in checks.items():
            ok = got == want
            bad += not ok
            print(f"{name} {what}: {'ok' if ok else f'FAIL {got} != {want}'}")
    print("PASS" if bad == 0 else f"FAIL ({bad})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

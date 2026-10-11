# SPDX-License-Identifier: MIT

# Copyright (c) 2026 Jay Leaton. The DeepSeek-V4.1-Flash family of TensorFold (Apache-2.0): see THIRD_PARTY_NOTICES.md.
"""Offline check that the STREAM_RB twins keep `_stream`'s MMA layouts, score float ops and shuffle offsets."""

from __future__ import annotations

import argparse
import collections
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

FLOAT_OPS = re.compile(r"^\s*(mma\.sync\S*|shfl\.sync\.bfly\.b32|fma\.rn\.f32|add\.f32|add\.rn\.f32|max\.f32|max\.NaN\.f32)",
                       re.M)
SHFL = re.compile(r"shfl\.sync\.bfly\.b32\s+[^,]+,\s*[^,]+,\s*(\d+)")
MMA = re.compile(r"#mma\d* = (#ttg\.nvidia_mma<[^>]*>)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="the served TensorFold src (csa2/stream_topk.py's package root)")
    a = ap.parse_args()
    sys.path[:0] = [a.src, str(HERE)]
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from tensorfold.families.deepseek_v41.cuda.csa2 import stream_topk as st
    from dsv41_zig_triton import stream_rb as rb

    tgt = GPUTarget("cuda", 121, 32)
    common = dict(RATIO=2, H=32, D=128, BP=64, WS=1.0 / math.sqrt(32), SCALE=1.0 / math.sqrt(128), SPLIT=16384,
                  K=512, CAP=1024, PSH=7, KFP8=True)

    def comp(fn, sig, consts):
        sig = dict(sig, **{k: "constexpr" for k in consts})
        k = triton.compile(ASTSource(fn=fn, signature=sig, constexprs=consts), target=tgt,
                           options=dict(num_warps=8, num_stages=3, enable_fp_fusion=True))
        return k.asm["ttgir"], k.asm["ptx"]

    ref = comp(st._stream, {"QI": "*bf16", "W": "*fp32", "w_stride": "i32", "IK": "*u8", "POS": "*i32", "KEYS": "*i64",
                            "k_stride": "i32", "NK": "i32", "BUF": "*i64", "nsplit": "i32", "PT": "*i32"},
               dict(common, MODE=0, BS=8))
    ref_mma = set(MMA.findall(ref[0]))
    ref_ops = collections.Counter(FLOAT_OPS.findall(ref[1]))
    ref_shfl = collections.Counter(SHFL.findall(ref[1]))
    sig = {"QI": "*bf16", "W": "*fp32", "w_stride": "i32", "IK": "*u8", "POS": "*i32", "R": "i32", "BUF": "*i64",
           "nsplit": "i32", "PT": "*i32"}
    bad = 0
    for n, fn in ((1, rb._stream_pf), (2, rb._stream_rb2), (4, rb._stream_rb4)):
        ttgir, ptx = comp(fn, sig, dict(common))
        checks = {
            "mma layout": (set(MMA.findall(ttgir)), ref_mma),
            "dots": (ttgir.count("tt.dot "), n * ref[0].count("tt.dot ")),
            "float ops": (collections.Counter(FLOAT_OPS.findall(ptx)), collections.Counter({k: n * v for k, v in ref_ops.items()})),
            "shuffle offsets": (collections.Counter(SHFL.findall(ptx)), collections.Counter({k: n * v for k, v in ref_shfl.items()})),
        }
        for what, (got, want) in checks.items():
            ok = got == want
            bad += not ok
            print(f"{'pf' if n == 1 else f'rb{n}'} {what}: {'ok' if ok else f'FAIL {got} != {want}'}")
    print("PASS" if bad == 0 else f"FAIL ({bad})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

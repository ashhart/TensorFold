# SPDX-License-Identifier: Apache-2.0
"""Wall-time speed through the Level Zero host (tf-xpu-exl3-test with reps): launch gaps included.
Cases: one layer's routed experts (top-8 of 256, one token), the same 8 experts as plain groups, the fused MLP
(top-8 of 256 and the shared expert: gate, up, SiLU, down) and lm_head. Each call reads the same experts, so some
of a layer's 14 MB can stay in L2 between calls (about 6% on the fused MLP, by exl3_moe_check.py with fresh picks).
Inputs are real checkpoint weights (constant fills read faster than DRAM on Xe2: memory compression).
Usage: exl3_zig_bench.py <EXL3 checkpoint> <tf-xpu-exl3-test> <work dir> [reps]   (one GPU job at a time)"""
import os, struct, subprocess, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import exl3_ref as ref, exl3_check as ck
from exl3_zig_check import MAGIC
import exl3_moe_check as mc


def bench(prog, work, mode, mats, rows, x, reps, topk=0, per=0, ids=None):
    k, n = mats[0][0].shape[0] * 16, mats[0][0].shape[1] * 16
    parts = [struct.pack("<10I", MAGIC, mode, len(mats), k, n, mats[0][0].shape[2] // 8, rows, topk, per, 1)]
    for tr, suh, svh, _ in mats:
        parts += [np.ascontiguousarray(a).tobytes() for a in (tr, suh, svh)]
    if ids is not None:
        parts.append(np.ascontiguousarray(ids, np.uint32).tobytes())
    parts.append(np.ascontiguousarray(x).tobytes())
    case = os.path.join(work, "case.bin")
    open(case, "wb").write(b"".join(parts))
    r = subprocess.run([prog, case, os.path.join(work, "y.bin"), str(reps)], check=True, capture_output=True, text=True)
    return r.stderr.strip().splitlines()[-1]


def main():
    sage, prog, work = sys.argv[1:4]
    reps = int(sys.argv[4]) if len(sys.argv) > 4 else 200
    os.makedirs(work, exist_ok=True)
    st = ref.SafeTensors(sage)
    rng = np.random.default_rng(4)
    L = "model.language_model.layers.0.mlp.experts"
    ids = rng.permutation(256)[:8].astype(np.uint32).reshape(1, 8)
    for proj, per in (("gate_proj", 0), ("down_proj", 1)):
        experts = [st.group(f"{L}.{i}.{proj}") for i in range(256)]
        k = experts[0][0].shape[0] * 16
        x = ck.to_bf16(rng.standard_normal((8 if per else 1, k)).astype(np.float32))
        print(f"routed {proj} top-8 of 256, 1 token:  {bench(prog, work, 1, experts, 1, x, reps, topk=8, per=per, ids=ids)}", flush=True)
        picked = [experts[int(i)] for i in ids[0]]
        xg = ck.to_bf16(rng.standard_normal((8, 1, k)).astype(np.float32))
        print(f"groups {proj} same 8 experts, 1 row: {bench(prog, work, 0, picked, 1, xg, reps)}", flush=True)
    grp = lambda p: tuple(st.group(f"{p}.{j}_proj") for j in ("gate", "up", "down"))
    table = [grp(f"{L}.{i}") for i in range(256)] + [grp(L.replace("experts", "shared_expert"))]
    parts = [struct.pack("<10I", MAGIC, 2, len(table), 2048, 512, 8, 1, 8, 1, 1)]
    for ex in table:
        for m in ex:
            parts += [np.ascontiguousarray(a).tobytes() for a in m[:3]]
    parts.append(np.ascontiguousarray(ids, np.uint32).tobytes())
    parts.append(np.ascontiguousarray(ck.to_bf16(rng.standard_normal((1, 2048)).astype(np.float32))).tobytes())
    case = os.path.join(work, "case.bin")
    open(case, "wb").write(b"".join(parts))
    r = subprocess.run([prog, case, os.path.join(work, "y.bin"), str(reps)], check=True, capture_output=True, text=True)
    print(f"fused MLP top-8 of 256 + shared, 1 token: {r.stderr.strip().splitlines()[-1]}", flush=True)
    head = st.group("lm_head")
    for rows in (1, 4):
        x = ck.to_bf16(rng.standard_normal((rows, 2048)).astype(np.float32))
        print(f"lm_head rows={rows}:                       {bench(prog, work, 0, [head], rows, x, reps)}", flush=True)


if __name__ == "__main__":
    main()

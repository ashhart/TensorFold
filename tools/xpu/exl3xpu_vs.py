# SPDX-License-Identifier: Apache-2.0
"""Head-to-head: exl3xpu decode path (via exl3xpu_bench) vs exl3.cl on the same Sage groups, same B70.
Usage: exl3xpu_vs.py <repo> <sage dir> <exl3xpu_bench binary> <work dir>   (one GPU job at a time)"""
import os, subprocess, sys
import numpy as np
sys.path.insert(0, os.path.join(sys.argv[1], "tools/xpu"))
import exl3_ref as ref, exl3_check as ck

repo, sage, binp, work = sys.argv[1:5]
st = ref.SafeTensors(sage)
e = ck.Exl3(repo)
rng = np.random.default_rng(5)
cases = [("experts gate_proj 8x40", [f"model.language_model.layers.{l}.mlp.experts.{x}.gate_proj" for l in range(40) for x in range(8)]),
         ("experts down_proj 8x40", [f"model.language_model.layers.{l}.mlp.experts.{x}.down_proj" for l in range(40) for x in range(8)]),
         ("lm_head", ["lm_head"])]
for label, names in cases:
    groups = [st.group(n) for n in names]
    kt, nt, w = groups[0][0].shape
    K, N, bits = kt * 16, nt * 16, w // 16
    tb = sum(g[0].nbytes for g in groups)
    os.makedirs(work, exist_ok=True)
    np.concatenate([g[0].ravel() for g in groups]).tofile(f"{work}/tr.bin")
    np.concatenate([g[1] for g in groups]).tofile(f"{work}/suh.bin")
    np.concatenate([g[2] for g in groups]).tofile(f"{work}/svh.bin")
    for M in (1, 8):
        x = ck.to_bf16(rng.standard_normal((M, K)).astype(np.float32))
        x.tofile(f"{work}/x.bin")
        r = subprocess.run([binp, work, str(K), str(N), str(bits), str(len(groups)), str(M), "20"], capture_output=True, text=True,
                           env=dict(os.environ, LD_LIBRARY_PATH=os.path.join(os.path.dirname(binp), "sycl/lib")))
        tline = [l for l in r.stdout.splitlines() if l.startswith("exl3xpu")]
        if not tline:
            print(label, M, "exl3xpu failed:", r.stdout[-400:], r.stderr[-800:]); continue
        t_x = float(tline[0].split(":")[-1].split()[0]) * 1e-6
        out = np.fromfile(f"{work}/out.bin", np.float32).reshape(len(groups), M, N)
        # accuracy on group 0 (lm_head: first 16 384 columns) against the fp64 reference
        g0 = groups[0]; ncol = min(N, 16384)
        Wr = ref.dequantize(2 * bits, 2, np.ascontiguousarray(g0[0][:, : ncol // 16]), g0[1], g0[2][:ncol])
        yr = ck.from_bf16(x).astype(np.float64) @ Wr
        err_x = np.abs(out[0][:, :ncol] - yr).max() / np.abs(yr).max()
        ys, t = ck.project(e, groups, [x] * len(groups), M, reps=20)
        err_o = np.abs(ys[0][:, :ncol] - yr).max() / np.abs(yr).max()
        t_o = t.sum()
        print(f"{label:24s} M={M}: exl3xpu {t_x*1e6:8.1f} us {tb/t_x/1e9:6.1f} GB/s err {err_x:.1e} | "
              f"exl3.cl {t_o*1e6:8.1f} us {tb/t_o/1e9:6.1f} GB/s err {err_o:.1e} | ours/theirs speed {t_x/t_o:.2f}x  [{tline[0] and r.stdout.splitlines()[0]}]")

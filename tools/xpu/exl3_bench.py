# SPDX-License-Identifier: Apache-2.0
"""Gate 4: EXL3 kernel speed on the B70 against the measured copy bandwidth.

Usage: exl3_bench.py <repo> <sage EXL3 dir>   (one GPU job at a time)
Effective GB/s = trellis bytes / kernel time (OpenCL profiling events, median of reps).
"""
import os, sys, time
import numpy as np
import pyopencl as cl

sys.path.insert(0, os.path.dirname(__file__))
import exl3_ref as ref, exl3_check as ck


def copy_bw(e, nbytes=1 << 30, reps=10):
    a = cl.Buffer(e.ctx, cl.mem_flags.READ_WRITE, nbytes); b = cl.Buffer(e.ctx, cl.mem_flags.READ_WRITE, nbytes)
    # random bytes: Xe2 compresses constant data, which inflates a copy of a filled buffer
    cl.enqueue_copy(e.q, a, np.random.default_rng(0).integers(0, 2**32, nbytes // 4, dtype=np.uint32)); e.q.finish()
    t = []
    for _ in range(reps):
        ev = cl.enqueue_copy(e.q, b, a); ev.wait()
        t.append((ev.profile.end - ev.profile.start) * 1e-9)
    copy = 2 * nbytes / np.median(t) / 1e9  # read + write
    src = """__kernel void rd(__global const uint4 *a, __global uint *o, ulong n) {
        uint4 s = 0; for (ulong i = get_global_id(0); i < n; i += get_global_size(0)) s ^= a[i];
        if ((s.x ^ s.y ^ s.z ^ s.w) == 0x12345678u) o[0] = 1; }"""
    p = cl.Program(e.ctx, src).build()
    o = cl.Buffer(e.ctx, cl.mem_flags.READ_WRITE, 4)
    t = []
    for _ in range(reps):
        ev = p.rd(e.q, (256 * 1024,), (256,), a, o, np.uint64(nbytes // 16)); ev.wait()
        t.append((ev.profile.end - ev.profile.start) * 1e-9)
    return copy, nbytes / np.median(t) / 1e9


def main():
    repo, sage = sys.argv[1], sys.argv[2]
    e = ck.Exl3(repo)
    st = ref.SafeTensors(sage)
    copy, read = copy_bw(e)
    print(f"{e.q.device.name} driver {e.q.device.driver_version}")
    print(f"copy bandwidth (enqueue_copy 1 GiB, read+write): {copy:.0f} GB/s; read-only kernel: {read:.0f} GB/s")
    rng = np.random.default_rng(3)
    cases = []
    for proj in ("gate_proj", "down_proj"):
        groups = [st.group(f"model.language_model.layers.{l}.mlp.experts.{x}.{proj}") for l in range(40) for x in range(8)]
        cases.append((f"experts {proj} 8x40", groups))
    cases.append(("lm_head", [st.group("lm_head")]))
    rows_out = []
    for label, groups in cases:
        tb = sum(g[0].nbytes for g in groups)
        for rows in (1, 8, 16):
            xs = [ck.to_bf16(rng.standard_normal((rows, g[0].shape[0] * 16)).astype(np.float32)) for g in groups]
            _, t = ck.project(e, groups, xs, rows, reps=20)
            mv, total = t[1], t.sum()
            line = (f"{label:24s} rows={rows:2d} trellis {tb/1e6:7.1f} MB  rotate_in {t[0]*1e6:7.1f} us  matvec {mv*1e6:8.1f} us"
                    f"  rotate_out {t[2]*1e6:7.1f} us  matvec {tb/mv/1e9:6.1f} GB/s ({100*tb/mv/1e9/copy:4.1f}% of copy)"
                    f"  all three {tb/total/1e9:6.1f} GB/s")
            print(line); rows_out.append((label, rows, tb, mv, total))
    return copy, read, rows_out


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0
"""Gate 3, the route check and the fused MLP check through the Level Zero host (zig/src/xpu/exl3.zig, program tf-xpu-exl3-test).
Each case runs in the zig program; its outputs must equal the PyOpenCL host's bit for bit (same kernels, same
picks) and meet the numpy reference within 1e-3.
Usage: exl3_zig_check.py <repo> <sage EXL3 dir> <tf-xpu-exl3-test> <work dir>   (one GPU job at a time)"""
import os, struct, subprocess, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import exl3_ref as ref, exl3_check as ck, exl3_moe_check as mc

MAGIC = 0x334C5845


def run(prog, work, mode, mats, rows, x, topk=0, per=0, ids=None, out_f32=1):
    k, n = mats[0][0].shape[0] * 16, mats[0][0].shape[1] * 16
    halves = mats[0][0].shape[2] // 8
    parts = [struct.pack("<10I", MAGIC, mode, len(mats), k, n, halves, rows, topk, per, out_f32)]
    for tr, suh, svh, mul1 in mats:
        ck.refuse(halves, mul1)
        parts += [np.ascontiguousarray(a).tobytes() for a in (tr, suh, svh)]
    if ids is not None:
        parts.append(np.ascontiguousarray(ids, np.uint32).tobytes())
    parts.append(np.ascontiguousarray(x).tobytes())
    case, out = os.path.join(work, "case.bin"), os.path.join(work, "y.bin")
    open(case, "wb").write(b"".join(parts))
    subprocess.run([prog, case, out], check=True, stdout=subprocess.DEVNULL)
    y = np.fromfile(out, np.float32 if out_f32 else np.uint16)
    return y if out_f32 else ck.from_bf16(y)


def run_moe(prog, work, experts, shared, x, ids, topk, formats=1):
    """Mode 2: experts and shared are (gate, up, down) groups; y [slots, hidden]. formats: output in the low byte
    (0 bf16, 1 fp32, 2 fp16), input in the next (0 bf16 x, 1 fp16 x)."""
    tr = experts[0][0][0]
    k, n, halves = tr.shape[0] * 16, tr.shape[1] * 16, tr.shape[2] // 8
    table = experts + [shared]
    parts = [struct.pack("<10I", MAGIC, 2, len(table), k, n, halves, x.shape[0], topk, 1, formats)]
    for ex in table:
        for tr_, suh, svh, mul1 in ex:
            ck.refuse(halves, mul1)
            parts += [np.ascontiguousarray(a).tobytes() for a in (tr_, suh, svh)]
    parts.append(np.ascontiguousarray(ids, np.uint32).tobytes())
    parts.append(np.ascontiguousarray(x).tobytes())
    case, out = os.path.join(work, "case.bin"), os.path.join(work, "y.bin")
    open(case, "wb").write(b"".join(parts))
    subprocess.run([prog, case, out], check=True, stdout=subprocess.DEVNULL)
    y = np.fromfile(out, (np.uint16, np.float32, np.float16)[formats & 0xff])
    return (ck.from_bf16(y) if formats & 0xff == 0 else y.astype(np.float32)).reshape(-1, k)


def main():
    repo, sage, prog, work = sys.argv[1:5]
    os.makedirs(work, exist_ok=True)
    e = ck.Exl3(repo)
    st = ref.SafeTensors(sage)
    rng = np.random.default_rng(9)
    worst = 0.0
    for label, name in ck.SAGE.items():
        tr, suh, svh, mul1 = st.group(name)
        if label == "lm_head":
            tr = np.ascontiguousarray(tr[:, :1024]); svh = svh[: 1024 * 16]
        kt, nt, w = tr.shape
        W = ref.dequantize(w // 8, 2, tr, suh, svh)
        for rows in (1, 3, 5, 16):
            x = ck.to_bf16(rng.standard_normal((rows, kt * 16)).astype(np.float32))
            y = run(prog, work, 0, [(tr, suh, svh, mul1)], rows, x).reshape(rows, nt * 16)
            (yp,), _ = ck.project(e, [(tr, suh, svh, mul1)], [x], rows)
            same = (y.view(np.uint32) == yp.view(np.uint32)).all()
            yr = ck.from_bf16(x).astype(np.float64) @ W
            err = np.abs(y - yr).max() / np.abs(yr).max()
            worst = max(worst, err)
            print(f"zig {label} rows={rows:2d}: max|y-ref|/max|ref| = {err:.2e}, bit-exact vs PyOpenCL host: {same}")
            assert same and err <= 1e-3
    # eight groups in one launch, bf16 output
    groups = [st.group(f"model.language_model.layers.0.mlp.experts.{i}.gate_proj") for i in range(8)]
    rows = 2
    x = ck.to_bf16(rng.standard_normal((8, rows, 2048)).astype(np.float32))
    y = run(prog, work, 0, groups, rows, x, out_f32=0).reshape(8, rows, 512)
    bfw = 0.0
    for i, g in enumerate(groups):
        yr = ck.from_bf16(x[i]).astype(np.float64) @ ref.dequantize(g[0].shape[2] // 8, 2, *g[:3])
        bfw = max(bfw, np.abs(y[i] - yr).max() / np.abs(yr).max())
    print(f"zig 8 groups rows=2 bf16 out: worst max|y-ref|/max|ref| = {bfw:.2e} (bf16 rounding, limit 1e-2)")
    assert bfw <= 1e-2
    # routed experts: ids on the device, slot table built by exl3_route
    E, TOPK = 16, 8
    for proj, per in (("gate_proj", 0), ("down_proj", 1)):
        experts = [st.group(f"model.language_model.layers.0.mlp.experts.{i}.{proj}") for i in range(E)]
        k, n = experts[0][0].shape[0] * 16, experts[0][0].shape[1] * 16
        cache = {}
        for tokens in (1, 4):
            ids = np.stack([rng.permutation(E)[:TOPK] for _ in range(tokens)]).astype(np.uint32)
            x = ck.to_bf16(rng.standard_normal((tokens * TOPK if per else tokens, k)).astype(np.float32))
            y = run(prog, work, 1, experts, tokens, x, topk=TOPK, per=per, ids=ids).reshape(tokens, TOPK, n)
            err = 0.0
            for t in range(tokens):
                for j in range(TOPK):
                    i = int(ids[t, j])
                    if i not in cache:
                        cache[i] = ref.dequantize(experts[i][0].shape[2] // 8, 2, *experts[i][:3])
                    yr = ck.from_bf16(x[t * TOPK + j] if per else x[t]).astype(np.float64) @ cache[i]
                    err = max(err, np.abs(y[t, j] - yr).max() / np.abs(yr).max())
            worst = max(worst, err)
            print(f"zig route {proj} tokens={tokens} slots={tokens * TOPK}: worst max|y-ref|/max|ref| = {err:.2e}")
            assert err <= 1e-3
    # fused MLP: 8 picks of 24 experts and the shared expert, against the PyOpenCL host and the float64 reference
    L = mc.LAYER.format(0)
    grp = lambda p: tuple(st.group(f"{p}.{j}_proj") for j in ("gate", "up", "down"))
    sub = [grp(f"{L}.experts.{i}") for i in range(24)]
    shared = grp(f"{L}.shared_expert")
    moe = mc.Moe(e, 4, 2, 2)  # exl3_variants.moe: p, pd, ds
    dq = {}
    for tokens, f16 in ((1, False), (4, False), (4, True)):
        xr = rng.standard_normal((tokens, 2048)).astype(np.float32)
        ids = np.stack([rng.permutation(24)[:TOPK] for _ in range(tokens)]).astype(np.uint32)
        if f16:  # the Sage loop's activations: fp16 x in, fp16 y out
            x = xr.astype(np.float16)
            y = run_moe(prog, work, sub, shared, x, ids, TOPK, formats=2 | 1 << 8)
            same, xf = True, x.astype(np.float64)
        else:
            x = ck.to_bf16(xr)
            y = run_moe(prog, work, sub, shared, x, ids, TOPK)
            moe.setup(sub, shared, x, [ids], TOPK)
            moe.launch()
            same = (y.view(np.uint32) == moe.y_out().view(np.uint32)).all()
            xf = ck.from_bf16(x).astype(np.float64)
        err = 0.0
        for s_ in range(y.shape[0]):
            i = int(ids.flat[s_]) if s_ < tokens * TOPK else 24
            t = s_ // TOPK if s_ < tokens * TOPK else s_ - tokens * TOPK
            if i not in dq:
                dq[i] = [mc.dq(m) for m in (sub[i] if i < 24 else shared)]
            g, u = xf[t] @ dq[i][0], xf[t] @ dq[i][1]
            yr = (g / (1 + np.exp(-g)) * u) @ dq[i][2]
            err = max(err, np.abs(y[s_] - yr).max() / np.abs(yr).max())
        if f16:
            print(f"zig fused MLP fp16 in and out tokens={tokens}: worst max|y-ref|/max|ref| = {err:.2e} (fp16 rounding, limit 1e-3)")
        else:
            worst = max(worst, err)
            print(f"zig fused MLP tokens={tokens} slots={y.shape[0]}: worst max|y-ref|/max|ref| = {err:.2e}, bit-exact vs PyOpenCL host: {same}")
        assert same and err <= 1e-3
    print(f"zig host: all cases pass, worst fp32 {worst:.2e}")


if __name__ == "__main__":
    main()

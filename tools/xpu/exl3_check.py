# SPDX-License-Identifier: Apache-2.0
"""Gates 1 to 3 for zig/kernels/xpu/exl3.cl, through pyopencl (run-time compile, no ocloc).

Usage: exl3_check.py <repo> <sage EXL3 dir>   (one GPU job at a time)
"""
import os, sys
import numpy as np
import warnings
import pyopencl as cl
warnings.filterwarnings("ignore", category=cl.RepeatedKernelRetrieval) if hasattr(cl, "RepeatedKernelRetrieval") else None

sys.path.insert(0, os.path.dirname(__file__))
import exl3_ref as ref

DESC = np.dtype([("trellis", "<u8"), ("suh", "<u8"), ("svh", "<u8"), ("x", "<u8"), ("rot", "<u8"), ("sums", "<u8"), ("out", "<u8"),
                 ("k", "<u4"), ("n", "<u4"), ("halves", "<u4"), ("rows", "<u4"),
                 ("x_stride", "<u4"), ("out_stride", "<u4"), ("out_type", "<u4"), ("x_f16", "<u4")])
assert DESC.itemsize == 88


class Exl3:
    def __init__(self, repo):
        self.ctx = cl.create_some_context(interactive=False)
        self.q = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
        src = open(os.path.join(repo, "zig/kernels/xpu/exl3.cl")).read()
        self.src, self.progs, self.kern = src, {}, {}
        self.p = cl.Program(self.ctx, src).build("-cl-std=CL3.0")

    def ns(self, bits, rows, strips=1 << 30):
        """Column strips a sub-group (measured on the B70 by exl3_ab.py, 2026-10-24). 8 spills."""
        fixed = getattr(self, "ns_fixed", None) or int(os.environ.get("EXL3_NS", "0"))
        if fixed:
            return fixed
        if strips < 128:  # few strips a group (expert gate/up, N 512): keep the sub-group count up
            ns = 2 if rows <= 4 else 1
        else:
            ns = 4 if rows <= 4 else (2 if rows <= 8 else 1)
        while strips % ns:
            ns //= 2
        return ns

    def shape(self, bits, rows, strips, ks_):
        """(NS, KS) for a launch: strips are the groups' N / 16, ks_ their K / 16. The per-group rule above, then
        fewer strips a sub-group and K splits until the launch has SG_TARGET sub-groups (one layer's experts
        is far below it; measured by exl3_ab.py with AB_LAYERS=1, 2026-10-25)."""
        target = int(os.environ.get("EXL3_SG", "4096"))
        ns = self.ns(bits, rows, min(strips))
        fixed_ns = getattr(self, "ns_fixed", None) or int(os.environ.get("EXL3_NS", "0"))
        if not fixed_ns:
            while ns > 1 and sum(s // ns for s in strips) < target:
                ns //= 2
        while any(s % ns for s in strips):
            ns //= 2
        ks = getattr(self, "ks_fixed", None) or int(os.environ.get("EXL3_KS", "0"))
        if not ks:
            ks, sg = 1, sum(s // ns for s in strips)
            while sg * ks < target and all(k % (2 * ks) == 0 and k // (2 * ks) >= 8 for k in ks_):
                ks *= 2
        return ns, ks

    def hs(self, rows, ns):
        """Split-half matvec (HS) at 2 to 4 rows: +3 to +14% there, a loss at 1, 8 and 16 rows (exl3_ab.py, 2026-10-24)."""
        env = getattr(self, "hs_fixed", None)
        env = os.environ.get("EXL3_HS") if env is None else env
        if env is not None:
            return int(env) and ns >= 2
        return 2 <= rows <= 4 and ns >= 2

    def matvec(self, bits, rows, ns=1):
        hs = int(self.hs(rows, ns))
        key = (bits, rows, ns, hs)
        if key not in self.kern:
            p = cl.Program(self.ctx, self.src).build(f"-cl-std=CL3.0 -DBITS={bits} -DROWS={rows} -DNS={ns} -DHS={hs} " + getattr(self, "extra_opts", os.environ.get("EXL3_OPTS", "")))
            self.progs[key] = p
            self.kern[key] = cl.Kernel(p, "exl3_matvec")
        return self.kern[key]

    def gemm(self, bits):
        """exl3_gemm (prompt rows on XMX) from the fused MLP image of these bits (exl3_variants.moe)."""
        key = ("gemm", bits)
        if key not in self.kern:
            p = cl.Program(self.ctx, self.src).build(f"-cl-std=CL3.0 -DBITS={bits} -DMOE_P=4 -DMOE_PD=2 -DMOE_DS=2 -DMOE_KMAX=4096 " + os.environ.get("EXL3_OPTS", ""))
            self.progs[key] = p
            self.kern[key] = cl.Kernel(p, "exl3_gemm")
        return self.kern[key]

    def buf(self, a):
        return cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(a))

    def decode_tiles(self, halves, trellis):
        kt, nt, w = trellis.shape
        tb = self.buf(trellis)
        st = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, kt * nt * 256 * 4)
        wq = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, kt * nt * 256 * 2)
        self.p.exl3_decode_tiles(self.q, (nt * 16, kt), (16, 1), tb, st, wq, np.uint32(halves))
        s = np.empty(kt * nt * 256, np.uint32); w_ = np.empty((kt * 16, nt * 16), np.float16)
        cl.enqueue_copy(self.q, s, st); cl.enqueue_copy(self.q, w_, wq)
        return s.reshape(kt * nt, 256), w_


class Arena:
    """One device buffer; groups are byte offsets into it (the Level Zero host stores pointer differences)."""

    def __init__(self, ctx):
        self.parts, self.size, self.ctx = [], 0, ctx

    def add(self, a=None, nbytes=None):
        off = (self.size + 255) // 256 * 256
        n = a.nbytes if a is not None else nbytes
        self.parts.append((off, a))
        self.size = off + n
        return off

    def build(self, q):
        self.buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, max(self.size, 4))
        for off, a in self.parts:
            if a is not None:
                cl.enqueue_copy(q, self.buf, np.ascontiguousarray(a), dst_offset=off)
        return self.buf

    def read(self, q, off, shape, dt):
        out = np.empty(shape, dt)
        cl.enqueue_copy(q, out, self.buf, src_offset=off)
        return out


def refuse(halves, mul1):
    if not mul1:
        raise ValueError("exl3: only the mul1 codebook is supported")
    if halves % 2 or not 4 <= halves <= 16:
        raise ValueError(f"exl3: {halves / 2} bits not supported (2 to 8 whole bits)")


KS_MAX = 16  # sums scratch holds this many K splits


def project(e, groups, xs, rows, out_f32=True, reps=0):
    """groups: list of (trellis, suh, svh, mul1). xs: bf16 inputs [rows, K] per group. Returns fp32 outputs."""
    ar = Arena(e.ctx)
    d = np.zeros(len(groups), DESC)
    for g, ((tr, suh, svh, mul1), x) in enumerate(zip(groups, xs)):
        kt, nt, w = tr.shape
        halves = w // 8
        refuse(halves, mul1)
        k, n = kt * 16, nt * 16
        d[g] = (ar.add(tr), ar.add(suh), ar.add(svh), ar.add(x), ar.add(nbytes=rows * k * 4), ar.add(nbytes=rows * n * 4 * KS_MAX),
                ar.add(nbytes=rows * n * 4), k, n, halves, rows, k, n, int(out_f32), 0)
    base = ar.build(e.q)
    db = e.buf(d)
    kmax, nmax = int(d["k"].max()), int(d["n"].max())
    bits = {int(h) // 2 for h in d["halves"]}
    assert len(bits) == 1, "one width a launch"
    bits = bits.pop()
    if rows > 16:  # prompt rows: rotate_in, exl3_gemm (XMX), rotate_out with one split
        gm = e.gemm(bits)
        gp = e.progs[("gemm", bits)]  # the prompt kernels live in the fused MLP image
        gx = nmax // 16 // GEMM_S * 128
        def launch():
            return [gp.exl3_rotate_in_h(e.q, (kmax // 128 * 16, rows, len(groups)), (16, 1, 1), base, db, np.uint32(rows)),
                    gm(e.q, (gx, (rows + 8 * GEMM_M - 1) // (8 * GEMM_M), len(groups)), (128, 1, 1), base, db, np.uint32(rows)),
                    gp.exl3_rotate_out_rows(e.q, (nmax // 128 * 16, rows, len(groups)), (16, 1, 1), base, db, np.uint32(rows))]
        return finish(e, ar, d, groups, rows, launch, reps)
    rb = next(r for r in (1, 2, 4, 8, 16) if r >= rows)
    ns, ks = e.shape(bits, rb, [int(n) // 16 for n in d["n"]], [int(k) // 16 for k in d["k"]])
    assert ks <= KS_MAX
    mv = e.matvec(bits, rb, ns)
    fuse = fused(rb, nmax, [int(k) // 16 for k in d["k"]], ks)
    WG = 128
    gx = (nmax // 16 // ns * 16 + WG - 1) // WG * WG
    def launch():
        ev = [] if fuse else [e.p.exl3_rotate_in(e.q, (kmax // 128 * 16, rows, len(groups)), (16, 1, 1), base, db)]
        ev += [mv(e.q, (gx, len(groups), ks), (WG, 1, 1), base, db, np.uint32(ks), np.uint32(fuse)),
               e.p.exl3_rotate_out(e.q, (nmax // 128 * 16, rows * 8, len(groups)), (16, 8, 1), base, db, np.uint32(ks))]
        return ev
    return finish(e, ar, d, groups, rows, launch, reps)


GEMM_M = int(os.environ.get("GEMM_M", "16"))    # exl3.cl GEMM_M and GEMM_S (pass the same in EXL3_OPTS)
GEMM_S = int(os.environ.get("GEMM_S", "4"))


def finish(e, ar, d, groups, rows, launch, reps):
    ev = launch()
    e.q.finish()
    outs = [ar.read(e.q, int(d[g]["out"]), (rows, int(d[g]["n"])), np.float32) for g in range(len(groups))]
    times = None
    if reps:
        t = []
        for _ in range(reps):
            ev = launch(); e.q.finish()
            t.append([(v.profile.end - v.profile.start) * 1e-9 for v in ev])
        times = np.median(np.array(t), axis=0)
    return outs, times


def fused(rows, n_max, k_tiles, ks):
    """exl3.zig fused: the matvec rotates x itself at 1 to 4 rows when every K split is whole 128 blocks
    below 65536 columns (EXL3_FUSE=0 keeps exl3_rotate_in, to compare)."""
    return os.environ.get("EXL3_FUSE", "1") != "0" and rows <= 4 and n_max < 65536 and all(t // ks % 8 == 0 for t in k_tiles)


def to_bf16(a):
    u = np.ascontiguousarray(a, np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def from_bf16(u):
    return (u.astype(np.uint32) << 16).view(np.float32)


def gate1(e, repo):
    recs, _, _ = ref.check_oracle(os.path.join(repo, "zig/tests/fixtures/exl3_oracle.bin"))
    ok = 0
    for halves, cb, tr, want in recs:
        s, _ = e.decode_tiles(halves, tr.reshape(1, -1, tr.shape[1]))
        assert (s == want).all(), f"oracle record halves={halves} cb={cb}"
        ok += len(tr)
    widths = sorted({r[0] for r in recs})
    print(f"gate1 oracle: {len(recs)} records, {ok} tiles, halves {widths}: GPU states equal the oracle")
    rng = np.random.default_rng(1)
    for bits in range(2, 9):
        h = 2 * bits
        tr = rng.integers(-32768, 32768, (4, 8, 8 * h), dtype=np.int64).astype(np.int16)
        s, wq = e.decode_tiles(h, tr)
        assert (s == ref.tile_states(h, tr.reshape(32, -1))).all(), bits
        assert (wq.view(np.uint16) == ref.unpack(h, 2, tr).view(np.uint16)).all(), bits
    out = cl.Buffer(e.ctx, cl.mem_flags.WRITE_ONLY, 65536 * 4)
    e.p.exl3_mul1_check(e.q, (65536,), None, out)
    got = np.empty(65536, np.float32); cl.enqueue_copy(e.q, got, out)
    assert (got == ref.codebook_values(2, np.arange(65536, dtype=np.uint32)).astype(np.float32)).all()
    print("gate1 widths: 2..8 bits, 32 random tiles each: states and placed mul1 W_q bit-exact; mul1 table 65536/65536 bit-exact")


SAGE = {
    "expert gate_proj": "model.language_model.layers.0.mlp.experts.0.gate_proj",
    "expert down_proj": "model.language_model.layers.0.mlp.experts.0.down_proj",
    "lm_head": "lm_head",
}


def gate2(e, st):
    for label, name in SAGE.items():
        tr, suh, svh, mul1 = st.group(name)
        h = tr.shape[2] // 8
        refuse(h, mul1)
        kt, nt, _ = tr.shape
        if label == "lm_head":  # host reference on a 1/16 column slice; the GPU decodes all of it below in gate 3
            tr = np.ascontiguousarray(tr[:, : nt // 16])
        _, wq = e.decode_tiles(h, tr)
        want = ref.unpack(h, 2, tr)
        assert (wq.view(np.uint16) == want.view(np.uint16)).all(), label
        print(f"gate2 {label}: K={kt*16} N={nt*16} bits={h/2:g} decoded tiles {tr.shape[0]*tr.shape[1]}: W_q bit-exact")


def gate3(e, st):
    rng = np.random.default_rng(2)
    worst = 0.0
    for label, name in SAGE.items():
        tr, suh, svh, mul1 = st.group(name)
        kt, nt, w = tr.shape
        if label == "lm_head":
            tr = np.ascontiguousarray(tr[:, :1024]); svh = svh[: 1024 * 16]
        W = ref.dequantize(w // 8, 2, tr, suh, svh)
        for rows in (1, 3, 16, 64, 333):
            x = to_bf16(rng.standard_normal((rows, kt * 16)).astype(np.float32))
            (y,), _ = project(e, [(tr, suh, svh, mul1)], [x], rows)
            yr = from_bf16(x).astype(np.float64) @ W
            err = np.abs(y - yr).max() / np.abs(yr).max()
            worst = max(worst, err)
            print(f"gate3 {label} rows={rows}: max|y-ref|/max|ref| = {err:.2e}, rms rel = {np.sqrt(((y-yr)**2).mean()/(yr**2).mean()):.2e}")
    for bits in range(2, 9):  # every specialized width, random trellis, two groups in one launch
        groups = []
        for kt, nt in ((16, 24), (8, 40)):
            tr = rng.integers(-32768, 32768, (kt, nt, 16 * bits)).astype(np.int16)
            suh = rng.uniform(0.5, 1.5, kt * 16).astype(np.float16); svh = rng.uniform(0.5, 1.5, nt * 16).astype(np.float16)
            groups.append((tr, suh, svh, True))
        for rows in (1, 5, 40):
            xs = [to_bf16(rng.standard_normal((rows, g[0].shape[0] * 16)).astype(np.float32)) for g in groups]
            ys, _ = project(e, groups, xs, rows)
            for g, x, y in zip(groups, xs, ys):
                yr = from_bf16(x).astype(np.float64) @ ref.dequantize(2 * bits, 2, *g[:3])
                worst = max(worst, np.abs(y - yr).max() / np.abs(yr).max())
    print(f"gate3 synthetic: bits 2..8, two groups a launch, rows 1, 5 and 40: worst so far {worst:.2e}")
    assert worst <= 1e-3, worst
    print(f"gate3: worst {worst:.2e} <= 1e-3")


if __name__ == "__main__":
    repo, sage = sys.argv[1], sys.argv[2]
    e = Exl3(repo)
    print(e.q.device.name, e.q.device.driver_version)
    gate1(e, repo)
    st = ref.SafeTensors(sage)
    gate2(e, st)
    gate3(e, st)

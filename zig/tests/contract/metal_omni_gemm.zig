//! LFM3-Omni's Metal products: the few-row om_rows_R, the simdgroup om_gemm_*, and the tensor-unit om_nax_*.
const std = @import("std");
const mtl = @import("metal");
const k = @import("metal_omni_common.zig");

const c = k.c;
const R = k.R;
const Probe = k.Probe;

/// Dot rows for gate 4: integers to 255 times to 31 (partials under 2^23), the second half mirrored with a nudge.
fn stressPair(rng: std.Random, x: []u16, w: []u16) void {
    const h = x.len / 2;
    const top: u32 = @intCast(@min(31, (1 << 23) / (@max(h, 1) * 255)));
    for (0..h) |i| {
        const v: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 255));
        const u: f32 = @floatFromInt(1 + rng.uintLessThan(u32, top));
        x[i] = c.toBf16(v);
        x[h + i] = c.toBf16(v + @as(f32, if (i < 8) 1 else 0));
        w[i] = c.toBf16(u);
        w[h + i] = c.toBf16(-u);
    }
    if (x.len % 2 == 1) {
        x[x.len - 1] = 0;
        w[x.len - 1] = 0;
    }
}

fn plantRows(x: []u16, rows: usize, d: usize) void {
    const at = [_][2]usize{ .{ 0, d / 3 }, .{ 1, d / 2 }, .{ 2, d - 1 }, .{ 3, 0 }, .{ 3, d - 1 } };
    const put = [_]u16{ k.nan16, k.inf16, k.ninf16, k.inf16, k.ninf16 };
    for (at, put) |p, v| if (p[0] < rows) {
        x[p[0] * d + p[1]] = v;
    };
}

// --- om_rows_R: variant R | epilogue << 8 (0 bf16 y, 1 fp32 yf, 2 SwiGLU pairs) | batched << 12 (12 products)

const RowsArgs = extern struct { K: i32, N: i32, T: i32, ldx: i32, ldy: i32, has_y: i32, has_yf: i32, epi: i32, sw: i64, sx: i64, sy: i64 };

fn rowsR(p: *const Probe) usize {
    return p.variant & 0xff;
}
fn epi(p: *const Probe) u32 {
    return (p.variant >> 8) & 0xf;
}
fn batch(p: *const Probe) usize {
    return if (p.variant & 0x1000 != 0) 12 else 1;
}
fn yCols(p: *const Probe) usize {
    return if (epi(p) == 2) p.shape.n / 2 else p.shape.n;
}

/// Shape rows T, n the weights' rows, k their width; slots w, x, y.
fn rowsPrep(p: *Probe, rng: std.Random) !bool {
    const s = p.shape;
    if (s.rows == 0 or s.k == 0 or s.k % 8 != 0 or s.n == 0 or (epi(p) == 2 and s.n % 4 != 0)) return false;
    const nb = batch(p);
    const w = try p.alloc(u16, 0, nb * s.n * s.k);
    const x = try p.alloc(u16, 1, nb * s.rows * s.k);
    if (epi(p) == 1) _ = try p.alloc(f32, 2, nb * s.rows * s.n) else _ = try p.alloc(u16, 2, nb * s.rows * yCols(p));
    if (p.fill == .stress) {
        for (0..nb * s.n) |n| for (0..nb * s.rows) |t| if (t / s.rows == n / s.n) {
            var prng = std.Random.DefaultPrng.init(n *% 31 +% t);
            stressPair(prng.random(), x[t * s.k ..][0..s.k], w[n * s.k ..][0..s.k]);
        };
    } else {
        k.randomBf16(rng, w, -1, 1);
        k.randomBf16(rng, x, -1, 1);
        if (p.fill == .nonfinite) plantRows(x, s.rows, s.k);
    }
    try p.in(0);
    try p.in(1);
    try p.out(2, false);
    return true;
}

fn rowsLaunch(p: *Probe, e: mtl.ComputeEncoder) void {
    const s = p.shape;
    const yf = epi(p) == 1;
    const ldy = if (yf) s.n else yCols(p);
    // more rows than the pipeline takes: a launch per R rows, as successive calls
    var r0: usize = 0;
    while (r0 < s.rows) : (r0 += rowsR(p)) {
        const x = Probe.at(p.refs[1], r0 * s.k * 2);
        const y = Probe.at(p.refs[2], r0 * ldy * @as(usize, if (yf) 4 else 2));
        e.setPipeline(p.pipe);
        e.setBuffer(p.refs[0].buf, p.refs[0].off, 0);
        e.setBuffer(x.buf, x.off, 1);
        e.setBuffer(if (yf) p.dummy.buf else y.buf, if (yf) p.dummy.off else y.off, 2);
        e.setBuffer(if (yf) y.buf else p.dummy.buf, if (yf) y.off else p.dummy.off, 3);
        const a = RowsArgs{ .K = k.i32_(s.k), .N = k.i32_(s.n), .T = k.i32_(@min(rowsR(p), s.rows - r0)), .ldx = k.i32_(s.k), .ldy = k.i32_(yCols(p)), .has_y = @intFromBool(!yf), .has_yf = @intFromBool(yf), .epi = if (epi(p) == 2) 2 else 0, .sw = @intCast(s.n * s.k), .sx = @intCast(s.rows * s.k), .sy = @intCast(s.rows * (if (yf) s.n else yCols(p))) };
        e.setBytes(std.mem.asBytes(&a), 4);
        Probe.go(e, .{ k.blocks(s.n, 16), 1, batch(p) }, 128);
    }
}

fn rowsCount(p: *Probe) usize {
    return batch(p) * p.shape.rows * yCols(p);
}

fn dot(mode: k.Mode, p: *Probe, x: []const u16, w: []const u16, t: []R, depth: usize) R {
    for (t, x, w) |*q, a, b| q.* = R.bf16(a).mulExact(R.bf16(b));
    return k.sum(mode, p.scratch, t, depth);
}

fn rowsEval(p: *Probe, mode: k.Mode, out: []R) !void {
    const s = p.shape;
    const t = try p.a.alloc(R, s.k);
    defer p.a.free(t);
    const x = p.get(u16, 1);
    const w = p.get(u16, 0);
    const depth = s.k / 32 + 6;
    for (0..batch(p)) |z| for (0..s.rows) |r| {
        const xr = x[(z * s.rows + r) * s.k ..][0..s.k];
        const at = (z * s.rows + r) * yCols(p);
        if (epi(p) == 2) {
            for (0..s.n / 2) |j| {
                const a = dot(mode, p, xr, w[(z * s.n + 2 * j) * s.k ..][0..s.k], t, depth).bf();
                const g = dot(mode, p, xr, w[(z * s.n + 2 * j + 1) * s.k ..][0..s.k], t, depth).bf();
                out[at + j] = k.siluR(a).bf().mul(g).bf();
            }
        } else for (0..s.n) |n| {
            const v = dot(mode, p, xr, w[(z * s.n + n) * s.k ..][0..s.k], t, depth);
            out[at + n] = if (epi(p) == 1) v else v.bf();
        }
    };
}

const rows_def: k.Def = .{ .prep = rowsPrep, .launch = rowsLaunch, .count = rowsCount, .eval = rowsEval };

fn rowsDepth(s: c.Shape) usize {
    return s.k / 32 + 6;
}

// --- om_gemm_* and om_nax_*: C [m, n] = A B; variant: ta | tb << 1 | in fp32 << 2 | out fp32 << 3 | tensor units << 4

const GemmArgs = extern struct { M: i32, N: i32, K: i32, lda: i32, ldb: i32, ldc: i32, ta: i32, tb: i32, sa: i64, sb: i64, sc: i64 };
const NaxArgs = extern struct { M: i32, N: i32, K: i32, lda: i32, ldb: i32, ldd: i32, kblocks: i32, pad: i32 = 0, sa: i64, sb: i64, sd: i64 };

fn ta(p: *const Probe) bool {
    return p.variant & 1 != 0;
}
fn tb(p: *const Probe) bool {
    return p.variant & 2 != 0;
}
fn f32in(p: *const Probe) bool {
    return p.variant & 4 != 0;
}
fn f32out(p: *const Probe) bool {
    return p.variant & 8 != 0;
}
fn nax(p: *const Probe) bool {
    return p.variant & 16 != 0;
}

/// A [m, k] (stored [k, m] when ta) at (i, j), B [k, n] (stored [n, k] when tb) at (i, j): their places.
fn aAt(p: *const Probe, i: usize, j: usize) usize {
    return if (ta(p)) j * p.shape.rows + i else i * p.shape.k + j;
}
fn bAt(p: *const Probe, i: usize, j: usize) usize {
    return if (tb(p)) j * p.shape.k + i else i * p.shape.n + j;
}

/// Shape rows m, n n, k the depth; slots a, b, c (bf16 or fp32 each as the variant says).
fn gemmPrep(p: *Probe, rng: std.Random) !bool {
    const s = p.shape;
    if (s.rows == 0 or s.n == 0 or s.k == 0) return false;
    const ab = [2]usize{ s.rows * s.k, s.k * s.n };
    if (f32in(p)) {
        const a = try p.alloc(f32, 0, ab[0]);
        const b = try p.alloc(f32, 1, ab[1]);
        for (a) |*v| v.* = rng.float(f32) * 2 - 1;
        for (b) |*v| v.* = rng.float(f32) * 2 - 1;
        if (p.fill == .stress) {
            const xs = try p.a.alloc(u16, s.k);
            defer p.a.free(xs);
            const ws = try p.a.alloc(u16, s.k);
            defer p.a.free(ws);
            for (0..s.rows) |i| for (0..s.n) |j| if (i == j % s.rows) {
                stressPair(rng, xs, ws);
                for (0..s.k) |q| {
                    a[aAt(p, i, q)] = c.fromBf16(xs[q]);
                    b[bAt(p, q, j)] = c.fromBf16(ws[q]);
                }
            };
        }
        if (p.fill == .nonfinite) for ([_]usize{ 0, 1, 2 }, [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }) |i, v| if (i < s.rows) {
            a[aAt(p, i, s.k / 2)] = v;
        };
    } else {
        const a = try p.alloc(u16, 0, ab[0]);
        const b = try p.alloc(u16, 1, ab[1]);
        k.randomBf16(rng, a, -1, 1);
        k.randomBf16(rng, b, -1, 1);
        if (p.fill == .stress) {
            const xs = try p.a.alloc(u16, s.k);
            defer p.a.free(xs);
            const ws = try p.a.alloc(u16, s.k);
            defer p.a.free(ws);
            for (0..s.rows) |i| for (0..s.n) |j| if (i == j % s.rows) {
                stressPair(rng, xs, ws);
                for (0..s.k) |q| {
                    a[aAt(p, i, q)] = xs[q];
                    b[bAt(p, q, j)] = ws[q];
                }
            };
        }
        if (p.fill == .nonfinite) for ([_]usize{ 0, 1, 2 }, [_]u16{ k.nan16, k.inf16, k.ninf16 }) |i, v| if (i < s.rows) {
            a[aAt(p, i, s.k / 2)] = v;
        };
    }
    if (f32out(p)) _ = try p.alloc(f32, 2, s.rows * s.n) else _ = try p.alloc(u16, 2, s.rows * s.n);
    try p.in(0);
    try p.in(1);
    try p.out(2, false);
    return true;
}

fn gemmLaunch(p: *Probe, e: mtl.ComputeEncoder) void {
    const s = p.shape;
    const lda = if (ta(p)) s.rows else s.k;
    const ldb = if (tb(p)) s.k else s.n;
    if (nax(p)) {
        p.bind(e, &.{ 0, 1, 2 }, NaxArgs{ .M = k.i32_(s.rows), .N = k.i32_(s.n), .K = k.i32_(s.k), .lda = k.i32_(lda), .ldb = k.i32_(ldb), .ldd = k.i32_(s.n), .kblocks = k.i32_(s.k / 256), .sa = 0, .sb = 0, .sd = 0 });
        return Probe.go(e, .{ k.blocks(s.n, 128), k.blocks(s.rows, 64), 1 }, 256);
    }
    p.bind(e, &.{ 0, 1, 2 }, GemmArgs{ .M = k.i32_(s.rows), .N = k.i32_(s.n), .K = k.i32_(s.k), .lda = k.i32_(lda), .ldb = k.i32_(ldb), .ldc = k.i32_(s.n), .ta = @intFromBool(ta(p)), .tb = @intFromBool(tb(p)), .sa = 0, .sb = 0, .sc = 0 });
    Probe.go(e, .{ k.blocks(s.n, 64), k.blocks(s.rows, 64), 1 }, 128);
}

fn gemmCount(p: *Probe) usize {
    return p.shape.rows * p.shape.n;
}

fn gemmEval(p: *Probe, mode: k.Mode, out: []R) !void {
    const s = p.shape;
    const t = try p.a.alloc(R, s.k);
    defer p.a.free(t);
    for (0..s.rows) |i| for (0..s.n) |j| {
        for (t, 0..) |*q, x| {
            const a = if (f32in(p)) R.of(p.get(f32, 0)[aAt(p, i, x)]) else R.bf16(p.get(u16, 0)[aAt(p, i, x)]);
            const b = if (f32in(p)) R.of(p.get(f32, 1)[bAt(p, x, j)]) else R.bf16(p.get(u16, 1)[bAt(p, x, j)]);
            q.* = a.mulExact(b);
        }
        const v = k.sum(mode, p.scratch, t, s.k + 40);
        out[i * s.n + j] = if (f32out(p)) v else v.bf();
    };
}

const gemm_def: k.Def = .{ .prep = gemmPrep, .launch = gemmLaunch, .count = gemmCount, .eval = gemmEval };

fn gemmDepth(s: c.Shape) usize {
    return s.k + 40;
}

fn rowsSpec(name: []const u8, out: c.Elem, served: []const c.Shape) c.Spec {
    return .{ .name = name, .accum = .f32, .depth = rowsDepth, .ops = 8, .out = out, .tiles = .{ .{}, .{ .size = 16 }, .{ .size = 256, .step = 8, .min = 8 } }, .served = served };
}

fn gemmSpec(name: []const u8, out: c.Elem, served: []const c.Shape, tk: c.Tile) c.Spec {
    return .{ .name = name, .accum = .f32, .depth = gemmDepth, .out = out, .tiles = .{ .{ .size = 64 }, .{ .size = 64 }, tk }, .served = served };
}

const loop = "for (int c = lane; c < K / 8; c += 32) {";
const drop = "for (int c = lane; c < K / 8 - 1; c += 32) {";
const kloop = "for (int k0 = 0; k0 < a.K; k0 += BK) {";
const ktrim = "for (int k0 = 0; k0 + BK <= a.K; k0 += BK) {";
const nrows = "const int rows = min(32, a.M - row), cols = min(32, a.N - col);";
const nrows_ctl = "const int rows = min(31, a.M - row), cols = min(32, a.N - col);";
const k16: c.Tile = .{ .size = 16 };
const k256: c.Tile = .{ .size = 256, .step = 8, .min = 8 };

pub const entries = [_]k.Entry{
    .{ .def = &rows_def, .spec = rowsSpec("metal lfm3 omni om_rows_1", .bf16, &.{ .{ .rows = 1, .n = 2048, .k = 2048 }, .{ .rows = 1, .n = 3072, .k = 2048 } }), .file = .rows, .pipe = "om_rows_1", .variant = 1, .control = .{ loop, drop }, .what = "drops each row's last 8 columns" },
    .{ .def = &rows_def, .spec = rowsSpec("metal lfm3 omni om_rows_4", .bf16, &.{.{ .rows = 4, .n = 1536, .k = 512 }}), .file = .rows, .pipe = "om_rows_4", .variant = 4, .control = .{ loop, drop }, .what = "drops each row's last 8 columns" },
    .{ .def = &rows_def, .spec = rowsSpec("metal lfm3 omni om_rows_8", .bf16, &.{.{ .rows = 6, .n = 1536, .k = 512 }}), .file = .rows, .pipe = "om_rows_8", .variant = 8, .control = .{ loop, drop }, .what = "drops each row's last 8 columns" },
    .{ .def = &rows_def, .spec = rowsSpec("metal lfm3 omni om_rows_12", .bf16, &.{.{ .rows = 12, .n = 1024, .k = 1024 }}), .file = .rows, .pipe = "om_rows_12", .variant = 12, .control = .{ loop, drop }, .what = "drops each row's last 8 columns" },
    .{ .def = &rows_def, .spec = rowsSpec("metal lfm3 omni om_rows_16", .bf16, &.{.{ .rows = 16, .n = 2048, .k = 2048 }}), .file = .rows, .pipe = "om_rows_16", .variant = 16, .control = .{ loop, drop }, .what = "drops each row's last 8 columns" },
    .{ .def = &rows_def, .spec = rowsSpec("metal lfm3 omni om_rows_1 fp32 out", .f32, &.{.{ .rows = 1, .n = 12288, .k = 2048 }}), .file = .rows, .pipe = "om_rows_1", .variant = 1 | 1 << 8, .control = .{ "if (a.has_yf) YF[(long)r * a.N + n0 + j] = v[j];", "if (a.has_yf) YF[(long)r * a.N + n0 + j] = rb(v[j]);" }, .what = "adds bf16 to the fp32 output" },
    .{ .def = &rows_def, .spec = rowsSpec("metal lfm3 omni om_rows_12 batched (depth logits)", .bf16, &.{.{ .rows = 1, .n = 2017, .k = 1024 }}), .file = .rows, .pipe = "om_rows_12", .variant = 12 | 0x1000, .control = .{ "X += (long)tg.z * a.sx;", "X += (long)(tg.z / 2) * a.sx;" }, .what = "reads every other batch member's rows" },
    .{ .def = &rows_def, .spec = rowsSpec("metal lfm3 omni om_rows_1 SwiGLU pairs", .bf16, &.{.{ .rows = 1, .n = 15360, .k = 2048 }}), .file = .rows, .pipe = "om_rows_1", .variant = 1 | 2 << 8, .control = .{ "const float x = rb(v[2 * p]), c = rb(v[2 * p + 1]);", "const float x = rb(v[2 * p]), c = (v[2 * p + 1]);" }, .what = "drops bf16(c)" },
    .{ .def = &rows_def, .spec = rowsSpec("metal lfm3 omni om_rows_12 SwiGLU pairs", .bf16, &.{.{ .rows = 12, .n = 5632, .k = 1024 }}), .file = .rows, .pipe = "om_rows_12", .variant = 12 | 2 << 8, .control = .{ "Y[(long)r * a.ldy + n0 / 2 + p] = b(rb(x / (1.0f + precise::exp(-x))) * c);", "Y[(long)r * a.ldy + n0 / 2 + p] = b((x / (1.0f + precise::exp(-x))) * c);" }, .what = "drops bf16(silu(a))" },
    .{ .def = &gemm_def, .spec = gemmSpec("metal lfm3 omni om_gemm_bb a b^T", .bf16, &.{ .{ .rows = 40, .n = 768, .k = 768 }, .{ .rows = 24, .n = 1792, .k = 768 } }, k16), .file = .gemm, .pipe = "om_gemm_bb", .variant = 2, .control = .{ kloop, ktrim }, .what = "drops the last partial K tile" },
    .{ .def = &gemm_def, .spec = gemmSpec("metal lfm3 omni om_gemm_bf a b", .f32, &.{.{ .rows = 256, .n = 400, .k = 256 }}, k16), .file = .gemm, .pipe = "om_gemm_bf", .variant = 8, .control = .{ kloop, ktrim }, .what = "drops the last partial K tile" },
    .{ .def = &gemm_def, .spec = gemmSpec("metal lfm3 omni om_gemm_bf a^T b^T", .f32, &.{.{ .rows = 24, .n = 768, .k = 4096 }}, k16), .file = .gemm, .pipe = "om_gemm_bf", .variant = 1 | 2 | 8, .control = .{ "v = ld_in<TI>(A, a.ta ? (long)gk * a.lda + gm : (long)gm * a.lda + gk);", "v = ld_in<TI>(A, a.ta ? (long)gk * a.lda + (gm ^ 1) : (long)gm * a.lda + gk);" }, .what = "reads A's neighbouring column when transposed" },
    .{ .def = &gemm_def, .spec = gemmSpec("metal lfm3 omni om_gemm_ff a b (log-mel)", .f32, &.{ .{ .rows = 300, .n = 402, .k = 400 }, .{ .rows = 300, .n = 128, .k = 201 } }, k16), .file = .gemm, .pipe = "om_gemm_ff", .variant = 4 | 8, .control = .{ "As[r * BK + k] = v;", "As[r * BK + k] = rb(v);" }, .what = "adds bf16 to the fp32 operand" },
    .{ .def = &gemm_def, .spec = gemmSpec("metal lfm3 omni om_nax_bb_nt", .bf16, &.{ .{ .rows = 64, .n = 512, .k = 2048 }, .{ .rows = 100, .n = 256, .k = 768 } }, k256), .file = .nax, .pipe = "om_nax_bb_nt", .variant = 16 | 2, .control = .{ nrows, nrows_ctl }, .what = "drops a block row" },
    .{ .def = &gemm_def, .spec = gemmSpec("metal lfm3 omni om_nax_bf_nt", .f32, &.{.{ .rows = 100, .n = 512, .k = 768 }}, k256), .file = .nax, .pipe = "om_nax_bf_nt", .variant = 16 | 2 | 8, .control = .{ nrows, nrows_ctl }, .what = "drops a block row" },
    .{ .def = &gemm_def, .spec = gemmSpec("metal lfm3 omni om_nax_bf_nn", .f32, &.{.{ .rows = 256, .n = 400, .k = 256 }}, k256), .file = .nax, .pipe = "om_nax_bf_nn", .variant = 16 | 8, .control = .{ nrows, nrows_ctl }, .what = "drops a block row" },
    .{ .def = &gemm_def, .spec = gemmSpec("metal lfm3 omni om_nax_bf_tt", .f32, &.{ .{ .rows = 24, .n = 768, .k = 4096 }, .{ .rows = 64, .n = 256, .k = 2048 } }, k256), .file = .nax, .pipe = "om_nax_bf_tt", .variant = 16 | 1 | 2 | 8, .control = .{ nrows, nrows_ctl }, .what = "drops a block row" },
};

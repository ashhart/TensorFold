//! DeepSeek V4.1's indexer scores for the contract: inputs, the kernel's fp32 order as oracle, float64 reference.
const std = @import("std");
const contract = @import("contract");

const c = contract.spec;
const rows = contract.rows;

pub const heads = 32;
pub const dim = 128;
pub const block = 8;
pub const qrow = heads * dim / 2; // bytes of one query row: two FP4 codes a byte
pub const qsrow = heads * dim / 32; // e8m0 scales of one query row
pub const krow = dim / 2 + dim / 32; // bytes of one key: FP4 codes then four e8m0 scales
pub const w_scale: f32 = 0x1p-6; // softmax_scale * heads^-0.5 for 128 dims and 32 heads (model.py:513, 556)

/// What bounds a query row's keys: all of them, a causal prompt, a candidate mask, or one decode step.
pub const Kind = enum { dense, causal, masked, decode };

pub const Geo = struct { start: u32, ratio: u32, mask_stride: u32 };

/// The call's arguments: row r reaches (start + r) / ratio keys (model.py:565 prefill, 568 decode), then the mask.
pub fn geometry(kind: Kind, nkeys: usize) Geo {
    const k: u32 = @intCast(nkeys);
    return switch (kind) {
        .dense => .{ .start = k, .ratio = 1, .mask_stride = 0 },
        .causal => .{ .start = 1, .ratio = 4, .mask_stride = 0 },
        .masked => .{ .start = k, .ratio = 1, .mask_stride = (k + block - 1) / block + 3 },
        .decode => .{ .start = if (k * 128 > 130) k * 128 - 130 else 1, .ratio = 128, .mask_stride = 0 },
    };
}

pub fn e2m1(code: u8) f32 {
    const e: u32 = (code >> 1) & 3;
    const m: f32 = @floatFromInt(code & 1);
    const v: f32 = if (e == 0) m * 0.5 else std.math.ldexp(m + 2, @as(i32, @intCast(e)) - 2);
    return if (code & 8 != 0) -v else v;
}

pub fn e8m0(code: u8) f32 {
    if (code == 255) return std.math.nan(f32);
    return std.math.ldexp(@as(f32, 1), @as(i32, code) - 127);
}

/// ReLU keeping a NaN, as the kernel does (a max against zero would turn it into 0).
fn relu32(x: f32) f32 {
    return if (x != x or x > 0) x else 0;
}

fn relu64(x: f64) f64 {
    return if (x != x or x > 0) x else 0;
}

/// Longest chain of roundings: a 32-term group, the scale, the group tree, the product and the 32-head sum.
pub fn depth(_: c.Shape) usize {
    return 32 + 1 + 2 + 1 + heads;
}

pub const Out = struct { oracle: f32, exact: f64, mag: f64 };

/// Host inputs of one call: Q [rows][qrow], QS [rows][qsrow], W [rows][heads] bf16, IK [keys][krow], mask bytes.
pub const Ix = struct {
    a: std.mem.Allocator,
    shape: c.Shape, // rows, n = keys, k = dim
    kind: Kind,
    geo: Geo,
    q: []u8,
    qs: []u8,
    w: []u16,
    ik: []u8,
    cm: []u8,
    qf: []f32 = &.{}, // decoded query values times their group scale
    wf: []f32 = &.{}, // weights times w_scale, as the kernel forms them
    kv: []f32 = &.{}, // decoded key values
    ksc: []f32 = &.{}, // key group scales

    pub fn init(a: std.mem.Allocator, shape: c.Shape, kind: Kind) !Ix {
        const geo = geometry(kind, shape.n);
        const q = try a.alloc(u8, shape.rows * qrow);
        errdefer a.free(q);
        const qs = try a.alloc(u8, shape.rows * qsrow);
        errdefer a.free(qs);
        const w = try a.alloc(u16, shape.rows * heads);
        errdefer a.free(w);
        const ik = try a.alloc(u8, shape.n * krow);
        errdefer a.free(ik);
        const cm = try a.alloc(u8, @max(16, shape.rows * geo.mask_stride));
        return .{ .a = a, .shape = shape, .kind = kind, .geo = geo, .q = q, .qs = qs, .w = w, .ik = ik, .cm = cm };
    }

    pub fn deinit(f: Ix) void {
        for ([_][]const u8{ f.q, f.qs, f.ik, f.cm }) |s| f.a.free(s);
        f.a.free(f.w);
        for ([_][]const f32{ f.qf, f.wf, f.kv, f.ksc }) |s| f.a.free(s);
    }

    pub fn outputs(f: Ix) usize {
        return f.shape.rows * f.shape.n;
    }

    pub fn fill(f: *Ix, mode: rows.Fill, seed: u64) void {
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (mode == .stress) return f.stress(rng);
        for (f.q) |*b| b.* = rng.int(u8);
        for (f.qs) |*b| b.* = 124 + rng.uintLessThan(u8, 7);
        for (f.w) |*b| b.* = c.toBf16(rng.float(f32) * 2 - 0.5);
        for (f.ik, 0..) |*b, i| b.* = if (i % krow >= dim / 2) 124 + rng.uintLessThan(u8, 7) else rng.int(u8);
        f.fillMask(rng);
        if (mode == .nonfinite) f.plant();
    }

    /// Row r % 7 == 3 has no live block, r % 11 == 5 all of them, the rest about a third; the padding reads as live.
    fn fillMask(f: *Ix, rng: std.Random) void {
        @memset(f.cm, 0xff);
        const stride = f.geo.mask_stride;
        if (stride == 0) return;
        const blocks = (f.shape.n + block - 1) / block;
        for (0..f.shape.rows) |r| for (0..blocks) |b| {
            f.cm[r * stride + b] = if (r % 7 == 3) 0 else if (r % 11 == 5) 1 else @intFromBool(rng.uintLessThan(u8, 3) == 0);
        };
    }

    /// Gate 1: a NaN query scale, an infinite and a NaN-making weight, a NaN key scale (every row meets that key).
    fn plant(f: *Ix) void {
        f.qs[3 * 4 + 1] = 255; // row 0, head 3, group 1
        f.w[1 * heads + 2] = 0x7f80; // row 1: +inf
        f.w[2 * heads + 5] = 0xff80; // row 2: -inf
        f.w[3 * heads + 7] = 0x7f80; // row 3: +inf and -inf together
        f.w[3 * heads + 9] = 0xff80;
        f.ik[(f.shape.n / 3) * krow + dim / 2 + 2] = 255;
    }

    /// Gate 4's inputs: scales 1, FP4 magnitudes that make every sum exact in fp32 and half sums inexact, relu active.
    fn stress(f: *Ix, rng: std.Random) void {
        const big = [8]u8{ 1, 3, 5, 6, 7, 7, 7, 7 }; // 0.5 1.5 3 4 6 6 6 6
        @memset(f.qs, 127);
        for (0..f.shape.rows) |r| for (0..heads) |h| for (0..dim / 2) |i| {
            const sign: u8 = if (h % 5 == 4) 8 else 0; // a negative head: relu drops its score
            const lo = big[rng.uintLessThan(usize, 8)] | sign;
            const hi = big[rng.uintLessThan(usize, 8)] | sign;
            f.q[r * qrow + h * (dim / 2) + i] = lo | (hi << 4);
        };
        for (0..f.shape.n) |key| {
            for (0..dim / 2) |i| f.ik[key * krow + i] = big[rng.uintLessThan(usize, 8)] | (big[rng.uintLessThan(usize, 8)] << 4);
            @memset(f.ik[key * krow + dim / 2 ..][0 .. dim / 32], 127);
        }
        for (f.w) |*b| b.* = c.toBf16(@floatFromInt(1 + rng.uintLessThan(u32, 7)));
        f.fillMask(rng);
    }

    /// The decoded forms the oracle and the float64 reference read; built once a call.
    pub fn decode(f: *Ix) !void {
        if (f.qf.len != 0) return;
        const nr = f.shape.rows;
        f.qf = try f.a.alloc(f32, nr * heads * dim);
        f.wf = try f.a.alloc(f32, nr * heads);
        f.kv = try f.a.alloc(f32, f.shape.n * dim);
        f.ksc = try f.a.alloc(f32, f.shape.n * 4);
        for (0..nr) |r| for (0..heads * dim) |i| {
            const h = i / dim;
            const d = i % dim;
            const byte = f.q[r * qrow + h * (dim / 2) + d / 2];
            f.qf[r * heads * dim + i] = e2m1((byte >> @intCast(4 * (d % 2))) & 15) * e8m0(f.qs[r * qsrow + h * 4 + d / 32]);
        };
        for (f.wf, f.w) |*x, b| x.* = c.fromBf16(b) * w_scale;
        for (0..f.shape.n) |key| {
            for (0..dim) |d| f.kv[key * dim + d] = e2m1((f.ik[key * krow + d / 2] >> @intCast(4 * (d % 2))) & 15);
            for (0..4) |g| f.ksc[key * 4 + g] = e8m0(f.ik[key * krow + dim / 2 + g]);
        }
    }

    pub fn live(f: Ix, r: usize, key: usize) bool {
        if (key >= (@as(usize, f.geo.start) + r) / f.geo.ratio) return false;
        return f.geo.mask_stride == 0 or f.cm[r * f.geo.mask_stride + key / block] != 0;
    }

    /// Score (r, key) live: the kernel's fp32 order as oracle, the float64 sum, its terms' magnitude.
    pub fn one(f: Ix, r: usize, key: usize) Out {
        const kv = f.kv[key * dim ..][0..dim];
        const ks = f.ksc[key * 4 ..][0..4];
        var acc: f32 = 0;
        var exact: f64 = 0;
        var mag: f64 = 0;
        for (0..heads) |h| {
            const qh = f.qf[(r * heads + h) * dim ..][0..dim];
            var p: [4]f32 = @splat(0);
            var pe: [4]f64 = @splat(0);
            var pm: f64 = 0;
            for (0..4) |g| {
                var m: f64 = 0;
                for (g * 32..g * 32 + 32) |i| {
                    p[g] += qh[i] * kv[i];
                    pe[g] += @as(f64, qh[i]) * kv[i];
                    m += @abs(@as(f64, qh[i]) * kv[i]);
                }
                pm += m * ks[g];
            }
            const dot: f32 = ((p[0] * ks[0]) + (p[1] * ks[1])) + ((p[2] * ks[2]) + (p[3] * ks[3]));
            const dote: f64 = ((pe[0] * ks[0]) + (pe[1] * ks[1])) + ((pe[2] * ks[2]) + (pe[3] * ks[3]));
            const wh = f.wf[r * heads + h];
            acc += relu32(dot) * wh;
            exact += relu64(dote) * wh;
            mag += @abs(@as(f64, wh)) * pm;
        }
        return .{ .oracle = acc, .exact = exact, .mag = mag };
    }

    /// The oracle's scores as `want` (-inf where masked); each must sit inside the fp32 bound of float64.
    pub fn reference(f: *Ix, want: []f64, mag: []f64) !void {
        try f.decode();
        const d: f64 = @floatFromInt(depth(f.shape));
        const n = f.shape.n;
        for (0..f.shape.rows) |r| for (0..n) |key| {
            const at = r * n + key;
            if (!f.live(r, key)) {
                want[at] = -std.math.inf(f64);
                mag[at] = 0;
                continue;
            }
            const o = f.one(r, key);
            if (std.math.isFinite(o.exact) and std.math.isFinite(o.mag) and @abs(@as(f64, o.oracle) - o.exact) > c.gamma(d, 0x1p-24) * o.mag + 0x1p-149) return error.OracleDrift;
            want[at] = o.oracle;
            mag[at] = o.mag;
        };
    }

    /// On gate 4's inputs: each score as accumulators of `acc` leave it, every sum in `order` (heads, then dims).
    pub fn simulate(f: *Ix, acc: c.Accum, order: rows.Order, out: []f64) !void {
        try f.decode();
        const n = f.shape.n;
        var t: [dim]f64 = undefined;
        var hs: [heads]f64 = undefined;
        for (0..f.shape.rows) |r| for (0..n) |key| {
            if (!f.live(r, key)) {
                out[r * n + key] = -std.math.inf(f64);
                continue;
            }
            for (0..heads) |h| {
                const qh = f.qf[(r * heads + h) * dim ..][0..dim];
                for (0..dim) |i| t[i] = @as(f64, qh[i]) * f.kv[key * dim + i] * f.ksc[key * 4 + i / 32];
                const dot = rows.sumIn(acc, order, &t);
                hs[h] = relu64(dot) * f.wf[r * heads + h];
            }
            out[r * n + key] = rows.sumIn(acc, order, &hs);
        };
    }
};

test "FP4 and e8m0 codes decode to their values" {
    const want = [8]f32{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };
    for (want, 0..) |v, i| {
        try std.testing.expectEqual(v, e2m1(@intCast(i)));
        try std.testing.expectEqual(-v, e2m1(@intCast(i + 8)));
    }
    try std.testing.expectEqual(@as(f32, 1), e8m0(127));
    try std.testing.expectEqual(@as(f32, 0x1p-6), e8m0(121));
    try std.testing.expect(std.math.isNan(e8m0(255)));
}

//! idol_decoder.cu's steps under the contract: the stem's embed and cast, to_kv and the head norms, at the SDK's points
const std = @import("std");
const cuda = @import("cuda");
const dk = @import("idol_decoder_cuda");
const r = @import("contract_idol_ref.zig");

const c = r.c;
const Range = r.Range;
const Sum = r.Sum;
const Fill = r.g.Fill;
const H = r.hidden;
const D = r.D;

/// The rows the decoders call a step at: 32k's anchors, one parent, the fewest past the short form, 262k's and 1M's.
const served_rows = [_]usize{ 1024, 1, 9, 8192, 32768 };

fn servedAt(comptime n: usize, comptime k: usize) [served_rows.len]c.Shape {
    var out: [served_rows.len]c.Shape = undefined;
    for (&out, served_rows) |*s, rows| s.* = .{ .rows = rows, .n = n, .k = k };
    return out;
}

/// An integer in [lo, hi].
fn int(rng: std.Random, lo: i32, hi: i32) f64 {
    return @floatFromInt(rng.intRangeAtMost(i32, lo, hi));
}

fn sign(rng: std.Random) f64 {
    return if (rng.boolean()) 1 else -1;
}

/// The point embedding's frequencies, 2^linspace(0, 10, 170) as octree.pointFreqs's (to an ulp: the kernel reads any).
fn freqs(out: []f64) void {
    for (out, 0..) |*v, j| v.* = r.f(std.math.pow(f64, 2, 10 * @as(f64, @floatFromInt(j)) / 169.0));
}

// --- tf_idol_dec_embed: h = fp32(fma(x2, w2, fma(x1, w1, x0 w0)) + b) + pe, pe = sin or cos of fp32(fp32(x f) pi) ---

pub const Embed = struct {
    pub const kinds = [_]r.Kind{ .f32, .f32, .f32, .f32 };
    pub const out: r.Kind = .f32;
    pub const in_place: ?usize = null;
    const served = servedAt(H, 3);
    const pi32 = r.f(std.math.pi);

    pub fn accepts(s: c.Shape) bool {
        return s.rows > 0 and s.n == H and s.k == 3;
    }

    /// The three-input FMA chain, then the bias.
    fn depth(_: c.Shape) usize {
        return 4;
    }

    pub fn spec(name: []const u8, _: u32) c.Spec {
        return .{ .name = name, .accum = .f32, .depth = depth, .ops = 2 + r.fn_ulps, .out = .f32, .served = &served };
    }

    pub fn sizes(s: c.Shape) [4]usize {
        return .{ s.rows * 3, H * 3, H, 170 };
    }

    pub fn count(s: c.Shape) usize {
        return s.rows * H;
    }

    pub fn fill(rng: std.Random, mode: Fill, s: c.Shape, _: u32, v: *[4][]f64) void {
        if (mode == .stress) {
            // x0 w0 and x1 w1 near +-2^16 (past fp16's range) cancelling to a few hundred; no embedding (frequencies 0)
            for (0..s.rows) |row| {
                v[0][row * 3] = int(rng, 200, 300);
                v[0][row * 3 + 1] = v[0][row * 3];
                v[0][row * 3 + 2] = int(rng, 1, 8);
            }
            for (0..H) |o| {
                v[1][o * 3] = sign(rng) * int(rng, 220, 300);
                v[1][o * 3 + 1] = -v[1][o * 3] + int(rng, -1, 1);
                v[1][o * 3 + 2] = int(rng, -8, 8);
            }
            for (v[2]) |*x| x.* = int(rng, -8, 8);
            @memset(v[3], 0);
            return;
        }
        r.uniform(rng, v[0], 0, 1);
        r.uniform(rng, v[1], -1, 1);
        r.uniform(rng, v[2], -0.5, 0.5);
        freqs(v[3]);
        if (mode == .nonfinite) {
            for ([_]f64{ r.nan, r.inf, -r.inf }, 0..) |x, i| v[0][(i * 3 + i) % v[0].len] = x;
            v[2][9] = r.nan;
        }
    }

    pub fn launch(k: *const dk.Kernels, s: cuda.Stream, dev: [4]u64, o: u64, shape: c.Shape, _: u32) !void {
        try dk.embed(k, s, dev[0], dev[1], dev[2], dev[3], o, shape.rows);
    }

    pub fn ref(_: std.mem.Allocator, shape: c.Shape, _: u32, v: [4][]const f64, sum: Sum, e: *r.Ends, want: []f64, mag: []f64) !void {
        var t: [3]f64 = undefined;
        for (0..shape.rows) |row| for (0..H) |o| {
            const x = v[0][row * 3 ..][0..3];
            for (&t, x, v[1][o * 3 ..][0..3]) |*p, xv, wv| p.* = xv * wv;
            const s = sum.total(&t, &t, &t, 3);
            const b = v[2][o];
            var pe: f64 = 0;
            var per = Range.of(0);
            if (o < 1020) {
                // the kernel's own fp32 angle (two correctly rounded products), then sinf or cosf within 2 ulps
                const j = o % 340;
                const ang = r.f(r.f(x[o / 340] * v[3][j % 170]) * pi32);
                pe = if (j < 170) @sin(ang) else @cos(ang);
                per = if (std.math.isNan(pe)) Range.of(pe) else Range.of(pe).fp32(r.fn_ulps);
            }
            e.put(want, mag, row * H + o, s.v + b + pe, s.r.add(Range.of(b)).fp32(0.5).add(per).fp32(0.5));
        };
    }
};

// --- tf_idol_dec_half: o = fp16(y + b) by column -------------------------------------------------------------------

pub const Half = struct {
    pub const kinds = [_]r.Kind{ .f32, .f32 };
    pub const out: r.Kind = .f16;
    pub const in_place: ?usize = null;
    const served = servedAt(H, 1);

    pub fn accepts(s: c.Shape) bool {
        return s.rows > 0 and s.n == H;
    }

    fn none(_: c.Shape) usize {
        return 0;
    }

    pub fn spec(name: []const u8, _: u32) c.Spec {
        const why = "no sums: an fp32 bias and the fp16 cast, each inside its interval";
        return .{ .name = name, .accum = .none, .depth = none, .ops = 1, .out = .f16, .served = &served, .sums = .{ .f16 = 0, .bf16 = 0, .why = why } };
    }

    pub fn sizes(s: c.Shape) [2]usize {
        return .{ s.rows * H, H };
    }

    pub fn count(s: c.Shape) usize {
        return s.rows * H;
    }

    pub fn fill(rng: std.Random, mode: Fill, _: c.Shape, _: u32, v: *[2][]f64) void {
        r.uniform(rng, v[0], -8, 8);
        r.uniform(rng, v[1], -2, 2);
        if (mode == .nonfinite) {
            r.plant(v[0], &.{ 0, 13, 26 });
            v[1][6] = r.inf;
        }
    }

    pub fn launch(k: *const dk.Kernels, s: cuda.Stream, dev: [2]u64, o: u64, shape: c.Shape, _: u32) !void {
        try dk.toHalf(k, s, dev[0], dev[1], o, shape.rows, H);
    }

    pub fn ref(_: std.mem.Allocator, shape: c.Shape, _: u32, v: [2][]const f64, _: Sum, e: *r.Ends, want: []f64, mag: []f64) !void {
        for (0..shape.rows * H) |i| {
            const y = v[0][i] + v[1][i % H];
            e.put(want, mag, i, r.h(y), Range.of(y).fp32(0.5).half());
        }
    }
};

// --- tf_idol_dec_kv: kv = fp16(ctx w^T + b), the latent's 16 channels in fp32 FMA ----------------------------------

pub const Kv = struct {
    pub const kinds = [_]r.Kind{ .f16, .f16, .f16 };
    pub const out: r.Kind = .f16;
    pub const in_place: ?usize = null;
    const K = 16;
    const N = 2 * H;
    const served = [_]c.Shape{.{ .rows = r.latent, .n = N, .k = K }};

    pub fn accepts(s: c.Shape) bool {
        return s.rows > 0 and s.n == N and s.k == K;
    }

    /// The 16 inputs' FMA chain, then the bias.
    fn depth(_: c.Shape) usize {
        return K + 1;
    }

    pub fn spec(name: []const u8, _: u32) c.Spec {
        return .{ .name = name, .accum = .f32, .depth = depth, .ops = 1, .out = .f16, .served = &served };
    }

    pub fn sizes(s: c.Shape) [3]usize {
        return .{ s.rows * K, N * K, N };
    }

    pub fn count(s: c.Shape) usize {
        return s.rows * N;
    }

    pub fn fill(rng: std.Random, mode: Fill, s: c.Shape, _: u32, v: *[3][]f64) void {
        if (mode == .stress) {
            // products to 2^16 whose halves cancel but for the first: partial sums near 2^19, exact in fp32
            for (0..s.rows) |row| for (0..K / 2) |k| {
                v[0][row * K + k] = int(rng, 1, 255);
                v[0][row * K + K / 2 + k] = v[0][row * K + k] + @as(f64, if (k == 0) 1 else 0);
            };
            for (0..N) |o| for (0..K / 2) |k| {
                v[1][o * K + k] = sign(rng) * int(rng, 1, 255);
                v[1][o * K + K / 2 + k] = -v[1][o * K + k];
            };
            for (v[2]) |*x| x.* = int(rng, -8, 8);
            return;
        }
        r.uniform(rng, v[0], -3, 3);
        r.uniform(rng, v[1], -0.5, 0.5);
        r.uniform(rng, v[2], -0.5, 0.5);
        if (mode == .nonfinite) {
            for ([_]f64{ r.nan, r.inf, -r.inf }, 0..) |x, i| v[0][(i * K + 3 * i) % v[0].len] = x;
            v[2][9] = r.nan;
        }
    }

    pub fn launch(k: *const dk.Kernels, s: cuda.Stream, dev: [3]u64, o: u64, shape: c.Shape, _: u32) !void {
        try dk.kv(k, s, dev[0], dev[1], dev[2], o, shape.rows);
    }

    pub fn ref(_: std.mem.Allocator, shape: c.Shape, _: u32, v: [3][]const f64, sum: Sum, e: *r.Ends, want: []f64, mag: []f64) !void {
        var t: [K]f64 = undefined;
        for (0..shape.rows) |row| for (0..N) |o| {
            for (&t, v[0][row * K ..][0..K], v[1][o * K ..][0..K]) |*p, x, w| p.* = x * w;
            const s = sum.total(&t, &t, &t, K);
            const b = v[2][o];
            e.put(want, mag, row * N + o, r.h(s.v + b), s.r.add(Range.of(b)).fp32(0.5).half());
        };
    }
};

// --- tf_idol_dec_heads: MultiHeadRMSNorm in place, F.normalize(x) times gamma times 8, rounded ----------------------

pub const Heads = struct {
    pub const kinds = [_]r.Kind{ .f16, .f32, .f32 };
    pub const out: r.Kind = .f16;
    pub const in_place: ?usize = 0;
    /// Cross attention's q rows (1,024 wide), the latent's to_kv rows (k at 0 of 2,048), self attention's qkv rows.
    const served = [_]c.Shape{
        .{ .rows = 1024, .n = 3 * H, .k = D },  .{ .rows = 1, .n = H, .k = D },            .{ .rows = 9, .n = H, .k = D },
        .{ .rows = 4096, .n = H, .k = D },      .{ .rows = r.latent, .n = 2 * H, .k = D }, .{ .rows = 8192, .n = 3 * H, .k = D },
        .{ .rows = 32768, .n = 3 * H, .k = D },
    };

    pub fn accepts(s: c.Shape) bool {
        return s.rows > 0 and s.k == D and (s.n == H or s.n == 2 * H or s.n == 3 * H);
    }

    /// A lane's two squares and their sum, then the warp's 5 shuffles.
    fn depth(_: c.Shape) usize {
        return 7;
    }

    pub fn spec(name: []const u8, _: u32) c.Spec {
        const why = "an fp16 output follows the head's sum of squares as its square root: a bf16 sum is off a few ulps";
        return .{ .name = name, .accum = .f32, .depth = depth, .ops = 4, .out = .f16, .served = &served, .sums = .{ .bf16 = 0, .why = why } };
    }

    pub fn sizes(s: c.Shape) [3]usize {
        return .{ s.rows * s.n, H, H };
    }

    pub fn count(s: c.Shape) usize {
        return s.rows * s.n;
    }

    pub fn fill(rng: std.Random, mode: Fill, _: c.Shape, _: u32, v: *[3][]f64) void {
        // integer heads, whose squares sum to ~2^20 exactly
        if (mode == .stress) r.integers(rng, v[0], 128) else r.uniform(rng, v[0], -4, 4);
        r.uniform(rng, v[1], 0.25, 2);
        r.uniform(rng, v[2], 0.25, 2);
        if (mode == .nonfinite) {
            r.plant(v[0], &.{ 5, 3 * H + 2 * D + 9, 2 * 3 * H + 3 * D + 1 });
            v[1][4 * D + 6] = r.nan;
        }
    }

    /// In place over the output (the rows' copy): q's heads at column 0, and k's at 1,024 in qkv rows.
    pub fn launch(k: *const dk.Kernels, s: cuda.Stream, dev: [3]u64, o: u64, shape: c.Shape, _: u32) !void {
        try dk.rmsHeads(k, s, o, dev[1], o, shape.rows, shape.n, 0);
        if (shape.n == 3 * H) try dk.rmsHeads(k, s, o, dev[2], o, shape.rows, shape.n, H);
    }

    pub fn ref(_: std.mem.Allocator, shape: c.Shape, _: u32, v: [3][]const f64, sum: Sum, e: *r.Ends, want: []f64, mag: []f64) !void {
        for (0..shape.rows * shape.n) |i| e.put(want, mag, i, v[0][i], Range.of(v[0][i]));
        var sq: [D]f64 = undefined;
        var lo: [D]f64 = undefined;
        var hi: [D]f64 = undefined;
        const normed: usize = if (shape.n == 3 * H) 2 else 1;
        for (0..normed) |which| for (0..shape.rows) |row| for (0..r.heads) |hd| {
            const base = row * shape.n + which * H + hd * D;
            const x = v[0][base..][0..D];
            for (&sq, &lo, &hi, x) |*s, *l, *u, xv| {
                s.* = xv * xv;
                const t = Range.of(xv).square().fp32(0.5);
                l.* = t.lo;
                u.* = t.hi;
            }
            const s = sum.total(&sq, &lo, &hi, 6);
            const n = clampMin(@sqrt(s.v));
            const nr = s.r.up(sqrt).fp32(r.sqrt_ulps).up(clampMin);
            const gamma = v[1 + which][hd * D ..][0..D];
            for (x, gamma, 0..) |xv, g, j| {
                const wr = Range.of(xv).div(nr).fp32(r.div_ulps).mul(Range.of(g)).fp32(0.5).mul(Range.of(8)).fp32(0.5).half();
                e.put(want, mag, base + j, r.h(xv / n * g * 8), wr);
            }
        };
    }
};

fn sqrt(x: f64) f64 {
    return @sqrt(@max(x, 0));
}

/// F.normalize's clamp_min(1e-12) in fp32, NaN kept.
fn clampMin(x: f64) f64 {
    return if (x < r.f(1e-12)) r.f(1e-12) else x;
}

test "the embed's and the heads' ranges hold their float64 values" {
    const a = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(9);
    inline for (.{ .{ Embed, c.Shape{ .rows = 3, .n = H, .k = 3 } }, .{ Heads, c.Shape{ .rows = 2, .n = 3 * H, .k = D } }, .{ Kv, c.Shape{ .rows = 2, .n = 2 * H, .k = 16 } } }) |x| {
        const Op = x[0];
        const shape = x[1];
        var bufs: [Op.kinds.len][]f64 = undefined;
        for (&bufs, Op.sizes(shape)) |*b, n| b.* = try a.alloc(f64, n);
        defer for (bufs) |b| a.free(b);
        Op.fill(prng.random(), .random, shape, 0, &bufs);
        var e: r.Ends = .{ .a = a };
        defer e.deinit();
        try e.size(Op.count(shape));
        const want = try a.alloc(f64, Op.count(shape));
        defer a.free(want);
        const mag = try a.alloc(f64, Op.count(shape));
        defer a.free(mag);
        var vals: [Op.kinds.len][]const f64 = undefined;
        for (&vals, bufs) |*o, b| o.* = b;
        try Op.ref(a, shape, 0, vals, .{ .range = false }, &e, want, mag);
        for (want, e.lo, e.hi) |w, lo, hi| try std.testing.expect(lo <= w and w <= hi);
    }
}

test "the planted NaN gamma and NaN query make whole heads NaN, as clamp_min keeps them" {
    try std.testing.expect(std.math.isNan(clampMin(std.math.nan(f64))));
    try std.testing.expectEqual(r.f(1e-12), clampMin(0));
}

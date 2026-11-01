//! Idol's image encoder's row kernels under the contract: GroupNorm, attention's biases and softmax, the rows.
const std = @import("std");
const cuda = @import("cuda");
const vae = @import("idol_vae_cuda");
const r = @import("contract_idol_ref.zig");

const c = r.c;
const Range = r.Range;
const Sum = r.Sum;
const Fill = r.g.Fill;

fn exp(x: f64) f64 {
    return @exp(x);
}

fn sqrt(x: f64) f64 {
    return @sqrt(@max(x, 0));
}

const why = "no sums: each output is a few fp32 steps, every one inside its interval";

fn noSums(_: c.Shape) usize {
    return 0;
}

// --- tf_iv_group_part: GroupNorm's chunk sums, of the values (variant 0) or their squared deviations (1) -------------

pub const GroupPart = struct {
    pub const Kernels = vae.Kernels;
    pub const kinds = [_]r.Kind{ .f32, .f32 };
    pub const out: r.Kind = .f32;
    pub const in_place: ?usize = null;

    /// A thread's chunk of pixels in turn, then its group's channels (up to 16) in turn.
    fn depth(s: c.Shape) usize {
        return @min(s.k, vae.chunk) + s.n / 32;
    }

    /// Rows unused (1); n: channels (32 groups); k: pixels, summed in chunks.
    pub fn spec(name: []const u8, _: u32) c.Spec {
        return .{
            .name = name,
            .accum = .f32,
            .depth = depth,
            .out = .f32,
            .tiles = .{ .{}, .{ .size = 128, .step = 32, .min = 32 }, .{ .size = vae.chunk } },
            .served = &.{ .{ .rows = 1, .n = 512, .k = 16384 }, .{ .rows = 1, .n = 512, .k = 65536 }, .{ .rows = 1, .n = 256, .k = 262144 }, .{ .rows = 1, .n = 128, .k = 1048576 } },
        };
    }

    pub fn accepts(s: c.Shape) bool {
        return s.n > 0 and s.n % 32 == 0 and s.n <= 512 and s.k > 0;
    }

    pub fn sizes(s: c.Shape) [2]usize {
        return .{ s.k * s.n, 64 };
    }

    pub fn count(s: c.Shape) usize {
        return 32 * vae.chunks(s.k);
    }

    pub fn fill(rng: std.Random, mode: Fill, s: c.Shape, centered: u32, v: *[2][]f64) void {
        if (mode == .stress) {
            // integers: plain sums reach ~2^15 a channel and cancel; deviations within 18 keep squares' sums under 2^24
            for (0..32) |grp| v[1][grp] = @floatFromInt(rng.intRangeAtMost(i32, -4, 4));
            const half = vae.chunk / 2;
            for (0..s.k) |p| for (0..s.n) |ch| {
                const grp = ch / (s.n / 32);
                const in = p % vae.chunk;
                v[0][p * s.n + ch] = if (centered == 1)
                    v[1][grp] + @as(f64, @floatFromInt(rng.intRangeAtMost(i32, -18, 18)))
                else if (in < half) @as(f64, if (in % 2 == 0) 127 else 511) else -@as(f64, if ((in - half) % 2 == 0) 127 else 511) + @as(f64, if (in - half < 4) 1 else 0);
            };
        } else {
            r.uniform(rng, v[0], -3, 3);
            r.uniform(rng, v[1][0..32], -0.5, 0.5);
            if (mode == .nonfinite) r.plant(v[0], &.{ 3, s.n + 9, 5 * s.n + 2 });
        }
        r.uniform(rng, v[1][32..], 0.5, 2);
    }

    pub fn launch(k: *const Kernels, s: cuda.Stream, dev: [2]u64, o: u64, shape: c.Shape, centered: u32) !void {
        try vae.groupPart(k, s, dev[0], dev[1], o, shape.k, shape.n, centered == 1);
    }

    pub fn ref(a: std.mem.Allocator, shape: c.Shape, centered: u32, v: [2][]const f64, sum: Sum, e: *r.Ends, want: []f64, mag: []f64) !void {
        const size = shape.n / 32;
        const most = vae.chunk * size;
        const buf = try a.alloc(f64, 3 * most);
        defer a.free(buf);
        const chunks = vae.chunks(shape.k);
        for (0..32) |grp| for (0..chunks) |ck| {
            var n: usize = 0;
            const mean = v[1][grp];
            for (ck * vae.chunk..@min((ck + 1) * vae.chunk, shape.k)) |p| for (0..size) |i| {
                const x = v[0][p * shape.n + grp * size + i];
                const t = if (centered == 1) Range.of(x).sub(Range.of(mean)).fp32(0.5).square().fp32(0.5) else Range.of(x);
                buf[n] = if (centered == 1) (x - mean) * (x - mean) else x;
                buf[most + n] = t.lo;
                buf[2 * most + n] = t.hi;
                n += 1;
            };
            const s = sum.total(buf[0..n], buf[most..][0..n], buf[2 * most ..][0..n], depth(shape));
            e.put(want, mag, grp * chunks + ck, s.v, s.r);
        };
    }
};

// --- tf_iv_group_finish: the 32 means (variant 0) or 32 rstd (1) from the chunk sums --------------------------------

pub const GroupFinish = struct {
    pub const Kernels = vae.Kernels;
    pub const kinds = [_]r.Kind{ .f32, .f32 };
    pub const out: r.Kind = .f32;
    /// stats [64] starts as input 1: the kernel writes the means' half or the rstd's, the other half stays.
    pub const in_place: ?usize = 1;

    /// A lane's share of the chunks in turn, then the warp's 5 shuffles.
    fn depth(s: c.Shape) usize {
        return (vae.chunks(s.k) + 31) / 32 + 5;
    }

    /// Rows unused (1); n: channels; k: pixels (their chunk sums are the input).
    pub fn spec(name: []const u8, centered: u32) c.Spec {
        return .{
            .name = name,
            .accum = .f32,
            .depth = depth,
            .ops = if (centered == 1) 2 * r.div_ulps + r.sqrt_ulps + 1 else r.div_ulps,
            .out = .f32,
            .tiles = .{ .{}, .{}, .{ .size = 32 * vae.chunk } },
            .served = &.{ .{ .rows = 1, .n = 512, .k = 16384 }, .{ .rows = 1, .n = 256, .k = 262144 }, .{ .rows = 1, .n = 128, .k = 1048576 } },
        };
    }

    pub fn accepts(s: c.Shape) bool {
        return s.n > 0 and s.n % 32 == 0 and s.n <= 512 and s.k > 0;
    }

    pub fn sizes(s: c.Shape) [2]usize {
        return .{ 32 * vae.chunks(s.k), 64 };
    }

    pub fn count(_: c.Shape) usize {
        return 64;
    }

    pub fn fill(rng: std.Random, mode: Fill, _: c.Shape, centered: u32, v: *[2][]f64) void {
        if (mode == .stress) {
            // integers whose sums (at most 4096 chunks) stay under 2^23: plain ones mirror, squared ones are positive
            const chunks = v[0].len / 32;
            for (0..32) |grp| for (0..chunks) |i| {
                const at = if (i < chunks / 2) i else i - chunks / 2;
                const x: f64 = @floatFromInt(1 + (at * 37 + grp) % 2000);
                v[0][grp * chunks + i] = if (centered == 1) x else if (i < chunks / 2) x else -x + @as(f64, if (at < 3) 1 else 0);
            };
        } else if (centered == 1) r.uniform(rng, v[0], 0, 1000) else r.uniform(rng, v[0], -500, 500);
        r.uniform(rng, v[1], -1, 1);
        if (mode == .nonfinite) r.plant(v[0], &.{ 1, v[0].len / 32 + 3, 2 * (v[0].len / 32) + 5 });
    }

    pub fn launch(k: *const Kernels, s: cuda.Stream, dev: [2]u64, o: u64, shape: c.Shape, centered: u32) !void {
        try vae.groupFinish(k, s, dev[0], o, shape.k, shape.n, centered == 1, 1e-6);
    }

    pub fn ref(_: std.mem.Allocator, shape: c.Shape, centered: u32, v: [2][]const f64, sum: Sum, e: *r.Ends, want: []f64, mag: []f64) !void {
        const chunks = vae.chunks(shape.k);
        const n = r.f(@as(f64, @floatFromInt(shape.k)) * @as(f64, @floatFromInt(shape.n / 32)));
        const eps = r.f(1e-6);
        for (0..64) |i| e.put(want, mag, i, v[1][i], Range.of(v[1][i]));
        for (0..32) |g| {
            const part = v[0][g * chunks ..][0..chunks];
            const s = sum.total(part, part, part, depth(shape));
            const mr = (Range{ .lo = s.r.lo / n, .hi = s.r.hi / n }).fp32(r.div_ulps);
            if (centered == 0) {
                e.put(want, mag, g, s.v / n, mr);
                continue;
            }
            const root = mr.add(Range.of(eps)).fp32(0.5).up(sqrt).fp32(r.sqrt_ulps);
            e.put(want, mag, 32 + g, 1 / @sqrt(s.v / n + eps), Range.of(1).div(root).fp32(r.div_ulps));
        }
    }
};

// --- tf_iv_norm: GroupNorm by given statistics, then SiLU (variant 1) or not -----------------------------------------

pub const Norm = struct {
    pub const Kernels = vae.Kernels;
    pub const kinds = [_]r.Kind{ .f32, .f32, .f32, .f32 };
    pub const out: r.Kind = .f32;
    pub const in_place: ?usize = null;

    /// Rows: pixels; n: channels (32 groups of n / 32).
    pub fn spec(name: []const u8, silu: u32) c.Spec {
        return .{
            .name = name,
            .accum = .none,
            .depth = noSums,
            .ops = 4 + r.fn_ulps + r.div_ulps + 1,
            .out = .f32,
            .tiles = .{ .{ .size = 8 }, .{ .size = 128, .step = 32, .min = 32 }, .{} },
            .served = if (silu == 1) &.{ .{ .rows = 4100, .n = 128, .k = 1 }, .{ .rows = 4100, .n = 256, .k = 1 }, .{ .rows = 4100, .n = 512, .k = 1 } } else &.{.{ .rows = 4100, .n = 512, .k = 1 }},
            .sums = .{ .f16 = 0, .bf16 = 0, .why = why },
        };
    }

    pub fn accepts(s: c.Shape) bool {
        return s.rows > 0 and s.n > 0 and s.n % 32 == 0;
    }

    pub fn sizes(s: c.Shape) [4]usize {
        return .{ s.rows * s.n, 64, s.n, s.n };
    }

    pub fn count(s: c.Shape) usize {
        return s.rows * s.n;
    }

    pub fn fill(rng: std.Random, mode: Fill, s: c.Shape, _: u32, v: *[4][]f64) void {
        // values stay within +-60 after the norm, short of fp32 exp's overflow in the SiLU
        r.uniform(rng, v[0], -4, 4);
        r.uniform(rng, v[1][0..32], -1, 1);
        r.uniform(rng, v[1][32..], 0.2, 3);
        r.uniform(rng, v[2], -2, 2);
        r.uniform(rng, v[3], -1, 1);
        if (mode == .nonfinite) r.plant(v[0], &.{ 3, s.n + 7, 2 * s.n + 1 });
    }

    pub fn launch(k: *const Kernels, s: cuda.Stream, dev: [4]u64, o: u64, shape: c.Shape, silu: u32) !void {
        try vae.norm(k, s, dev[0], dev[1], dev[2], dev[3], o, shape.rows, shape.n, silu == 1);
    }

    pub fn ref(_: std.mem.Allocator, shape: c.Shape, silu: u32, v: [4][]const f64, _: Sum, e: *r.Ends, want: []f64, mag: []f64) !void {
        const size = shape.n / 32;
        for (v[0], 0..) |x, i| {
            const ch = i % shape.n;
            const grp = ch / size;
            const m = v[1][grp];
            const rs = v[1][32 + grp];
            const g = v[2][ch];
            const b = v[3][ch];
            const y = (x - m) * rs * g + b;
            var yr = Range.of(x).sub(Range.of(m)).fp32(0.5).mul(Range.of(rs)).fp32(0.5).mul(Range.of(g)).fp32(0.5).add(Range.of(b)).fp32(0.5);
            var w = y;
            if (silu == 1) {
                w = y / (1 + @exp(-y));
                const ex = (Range{ .lo = @exp(-yr.hi), .hi = @exp(-yr.lo) }).fp32(r.fn_ulps);
                yr = yr.div(ex.add(Range.of(1)).fp32(0.5)).fp32(r.div_ulps);
            }
            e.put(want, mag, i, w, yr);
        }
    }
};

// --- tf_iv_bias: an attention linear's bias by column or by row (v^T), and the residual -----------------------------

pub fn Bias(comptime by_row: bool, comptime add: bool) type {
    return struct {
        pub const Kernels = vae.Kernels;
        pub const kinds = [_]r.Kind{ .f32, .f32, .f32 };
        pub const out: r.Kind = .f32;
        pub const in_place: ?usize = null;

        /// Rows by n: q, k and the output's [16384, 512], v^T's [512, 16384] (fewer rows: rows only tile).
        pub fn spec(name: []const u8, _: u32) c.Spec {
            return .{
                .name = name,
                .accum = .none,
                .depth = noSums,
                .ops = 2,
                .out = .f32,
                .tiles = .{ .{ .size = 1 }, .{ .size = 256 }, .{} },
                .served = if (by_row) &.{.{ .rows = 512, .n = 4100, .k = 1 }} else &.{.{ .rows = 4100, .n = 512, .k = 1 }},
                .sums = .{ .f16 = 0, .bf16 = 0, .why = why },
            };
        }

        pub fn accepts(s: c.Shape) bool {
            return s.rows > 0 and s.n > 0;
        }

        pub fn sizes(s: c.Shape) [3]usize {
            return .{ s.rows * s.n, if (by_row) s.rows else s.n, if (add) s.rows * s.n else 0 };
        }

        pub fn count(s: c.Shape) usize {
            return s.rows * s.n;
        }

        pub fn fill(rng: std.Random, mode: Fill, _: c.Shape, _: u32, v: *[3][]f64) void {
            r.uniform(rng, v[0], -6, 6);
            r.uniform(rng, v[1], -1, 1);
            r.uniform(rng, v[2], -3, 3);
            if (mode == .nonfinite) for (v, 0..) |col, i| if (col.len > 0) r.plant(col, &.{ 11 + 7 * i, 2048 + i });
        }

        pub fn launch(k: *const Kernels, s: cuda.Stream, dev: [3]u64, o: u64, shape: c.Shape, _: u32) !void {
            try vae.bias(k, s, dev[0], dev[1], if (add) dev[2] else null, o, shape.rows, shape.n, by_row);
        }

        pub fn ref(_: std.mem.Allocator, shape: c.Shape, _: u32, v: [3][]const f64, _: Sum, e: *r.Ends, want: []f64, mag: []f64) !void {
            for (0..shape.rows * shape.n) |i| {
                const b = v[1][if (by_row) i / shape.n else i % shape.n];
                var y = v[0][i] + b;
                var yr = Range.of(v[0][i]).add(Range.of(b)).fp32(0.5);
                if (add) {
                    y += v[2][i];
                    yr = yr.add(Range.of(v[2][i])).fp32(0.5);
                }
                e.put(want, mag, i, y, yr);
            }
        }
    };
}

// --- tf_iv_softmax: the attention's softmax(s / sqrt(512)) of a row of scores, in place -----------------------------

pub const Softmax = struct {
    pub const Kernels = vae.Kernels;
    pub const kinds = [_]r.Kind{.f32};
    pub const out: r.Kind = .f32;
    pub const in_place: ?usize = 0;
    const scale: f32 = 1.0 / @sqrt(@as(f32, 512));

    /// A thread's share of the row in turn, the warp's 5 shuffles, then the 8 warps' partials in order.
    fn depth(s: c.Shape) usize {
        return (s.n + 255) / 256 + 5 + 8;
    }

    /// Rows: query rows (the engine's chunk is 4,096; rows only tile); n = k: the keys, 16,384.
    pub fn spec(name: []const u8, _: u32) c.Spec {
        return .{
            .name = name,
            .accum = .f32,
            .depth = depth,
            .ops = 2 + r.fn_ulps + r.div_ulps,
            .out = .f32,
            .tiles = .{ .{}, .{ .size = 256 }, .{} },
            .served = &.{ .{ .rows = 6, .n = 16384, .k = 16384 }, .{ .rows = 6, .n = 4096, .k = 4096 } },
        };
    }

    pub fn accepts(s: c.Shape) bool {
        return s.rows > 0 and s.n > 0 and s.n <= vae.softmax_max;
    }

    pub fn sizes(s: c.Shape) [1]usize {
        return .{s.rows * s.n};
    }

    pub fn count(s: c.Shape) usize {
        return s.rows * s.n;
    }

    pub fn fill(rng: std.Random, mode: Fill, s: c.Shape, _: u32, v: *[1][]f64) void {
        // gate 4's rows are equal scores: every exp is 1 and the denominator n, exact in fp32 in any order
        if (mode == .stress) return @memset(v[0], 3.0);
        r.uniform(rng, v[0], -200, 200);
        if (mode == .nonfinite) r.plant(v[0], &.{ 3, s.n + 5, 2 * s.n + 7 });
    }

    pub fn launch(k: *const Kernels, s: cuda.Stream, _: [1]u64, o: u64, shape: c.Shape, _: u32) !void {
        try vae.softmax(k, s, o, shape.rows, shape.n, scale);
    }

    pub fn ref(a: std.mem.Allocator, shape: c.Shape, _: u32, v: [1][]const f64, sum: Sum, e: *r.Ends, want: []f64, mag: []f64) !void {
        const n = shape.n;
        const buf = try a.alloc(f64, 3 * n);
        defer a.free(buf);
        const ex = buf[0..n];
        const lo = buf[n .. 2 * n];
        const hi = buf[2 * n ..];
        for (0..shape.rows) |row| {
            const x = v[0][row * n ..][0..n];
            // s scale and its difference from the row's maximum are single correctly rounded fp32 steps
            var m = -r.inf;
            for (x) |s| m = r.fmax(m, r.f(s * scale));
            for (x, ex, lo, hi) |s, *t, *l, *u| {
                const d = r.f(r.f(s * scale) - m);
                t.* = @exp(d);
                const er = Range.of(d).up(exp).fp32(r.fn_ulps);
                l.* = er.lo;
                u.* = er.hi;
            }
            const total = sum.total(ex, lo, hi, depth(shape));
            for (0..n) |i| {
                const er: Range = .{ .lo = lo[i], .hi = hi[i] };
                e.put(want, mag, row * n + i, ex[i] / total.v, er.div(total.r).fp32(r.div_ulps));
            }
        }
    }
};

// --- tf_iv_tokens: the posterior's sample, patchified and normalized by the batch norm's statistics -----------------

pub const Tokens = struct {
    pub const Kernels = vae.Kernels;
    pub const kinds = [_]r.Kind{ .f32, .f32, .f32, .f32 };
    pub const out: r.Kind = .f32;
    pub const in_place: ?usize = null;
    const pad = 5;

    /// Rows: the latent grid's rows; n: the 128 channels of a row; k: the grid's columns.
    pub fn spec(name: []const u8, _: u32) c.Spec {
        return .{
            .name = name,
            .accum = .none,
            .depth = noSums,
            .ops = r.fn_ulps + r.div_ulps + 3,
            .out = .f32,
            .tiles = .{ .{ .size = 2, .step = 2, .min = 2 }, .{}, .{ .size = 2, .step = 2, .min = 2 } },
            .served = &.{ .{ .rows = 128, .n = 128, .k = 128 }, .{ .rows = 6, .n = 128, .k = 10 } },
            .sums = .{ .f16 = 0, .bf16 = 0, .why = why },
        };
    }

    pub fn accepts(s: c.Shape) bool {
        return s.n == 128 and s.rows > 0 and s.k > 0 and s.rows % 2 == 0 and s.k % 2 == 0;
    }

    pub fn sizes(s: c.Shape) [4]usize {
        return .{ s.rows * s.k * 64, s.rows * s.k * 32, 128, 128 };
    }

    pub fn count(s: c.Shape) usize {
        return (pad + s.rows * s.k / 4) * 128;
    }

    pub fn fill(rng: std.Random, mode: Fill, _: c.Shape, _: u32, v: *[4][]f64) void {
        for (0..v[0].len / 64) |p| {
            r.uniform(rng, v[0][p * 64 ..][0..32], -3, 3);
            // logvars past both of the clamp's ends (-30, 20) as well as inside them
            r.uniform(rng, v[0][p * 64 + 32 ..][0..32], -40, 30);
        }
        r.uniform(rng, v[1], -3, 3);
        r.uniform(rng, v[2], -1, 1);
        r.uniform(rng, v[3], 0.5, 4);
        if (mode == .nonfinite) {
            r.plant(v[0], &.{ 0, 64 + 1, 2 * 64 + 2 });
            r.plant(v[1], &.{ 7, 32 + 8, 2 * 32 + 9 });
            v[0][5 * 64 + 40] = r.nan;
        }
    }

    pub fn launch(k: *const Kernels, s: cuda.Stream, dev: [4]u64, o: u64, shape: c.Shape, _: u32) !void {
        try vae.tokens(k, s, dev[0], dev[1], dev[2], dev[3], o, shape.rows, shape.k, pad);
    }

    /// torch.clamp(x, -30, 20): NaN stays NaN.
    fn clamp(x: f64) f64 {
        return if (std.math.isNan(x)) x else std.math.clamp(x, -30, 20);
    }

    pub fn ref(_: std.mem.Allocator, shape: c.Shape, _: u32, v: [4][]const f64, _: Sum, e: *r.Ends, want: []f64, mag: []f64) !void {
        for (0..pad * 128) |i| e.put(want, mag, i, 0, Range.of(0));
        const w2 = shape.k / 2;
        for (0..shape.rows * shape.k / 4) |patch| for (0..128) |k| {
            const ch = k / 4;
            const p = ((patch / w2) * 2 + (k % 4) / 2) * shape.k + (patch % w2) * 2 + k % 2;
            const mean = v[0][p * 64 + ch];
            const lv = clamp(v[0][p * 64 + 32 + ch]);
            const nz = v[1][p * 32 + ch];
            const sd = @exp(0.5 * lv);
            const val = (mean + sd * nz - v[2][k]) / v[3][k];
            const er = Range.of(0.5 * lv).up(exp).fp32(r.fn_ulps);
            const zr = er.mul(Range.of(nz)).fp32(0.5).add(Range.of(mean)).fp32(0.5);
            const vr = zr.sub(Range.of(v[2][k])).fp32(0.5).div(Range.of(v[3][k])).fp32(r.div_ulps);
            e.put(want, mag, (pad + patch) * 128 + k, val, vr);
        };
    }
};

test "the softmax reference's ranges hold its float64 values, and gate 4's equal rows give 1 / n exactly" {
    const a = std.testing.allocator;
    const shape: c.Shape = .{ .rows = 2, .n = 300, .k = 300 };
    const x = try a.alloc(f64, 600);
    defer a.free(x);
    var prng = std.Random.DefaultPrng.init(5);
    var vals = [1][]f64{x};
    Softmax.fill(prng.random(), .random, shape, 0, &vals);
    var ends: r.Ends = .{ .a = a };
    defer ends.deinit();
    try ends.size(600);
    const want = try a.alloc(f64, 600);
    defer a.free(want);
    const mag = try a.alloc(f64, 600);
    defer a.free(mag);
    try Softmax.ref(a, shape, 0, .{x}, .{ .range = false }, &ends, want, mag);
    for (want, ends.lo, ends.hi) |w, lo, hi| try std.testing.expect(lo <= w and w <= hi);
    Softmax.fill(prng.random(), .stress, shape, 0, &vals);
    try Softmax.ref(a, shape, 0, .{x}, .{ .range = true }, &ends, want, mag);
    for (want) |w| try std.testing.expectEqual(@as(f64, 1.0 / 300.0), w);
    // fp16 sums of 4096 ones stall at 2048: gate 4's margin
    const ones = try a.alloc(f64, 3 * 4096);
    defer a.free(ones);
    @memset(ones[0..4096], 3.0);
    try ends.size(4096);
    try Softmax.ref(a, .{ .rows = 1, .n = 4096, .k = 4096 }, 0, .{ones[0..4096]}, .{ .sim = .{ .acc = .f16, .order = .sequential } }, &ends, ones[4096..8192], ones[8192..]);
    try std.testing.expectEqual(@as(f64, 1.0 / 2048.0), ones[4096]);
}

//! Row operations for the contract (RMS and layer norms, softmax, GELU, sums, dots, maxima): float64 references, sums.
const std = @import("std");
const c = @import("contract.zig");
const norms_ref = @import("../norms_ref.zig");
const act_ref = @import("../act_ref.zig");

/// What a gate asks the inputs to hold: ordinary values, NaN and infinities at known places, or sums that test fp32.
pub const Fill = enum { random, nonfinite, stress };

/// The order a simulated accumulator adds terms in: one chain, contiguous halves, or 32 lanes then their halves.
pub const Order = enum { sequential, pairwise, strided };

pub const Op = enum { rms, softmax, sum, dot, max, layer_norm, gelu };

/// One row's terms summed in `acc` (each add rounded to it) in `order`.
pub fn sumIn(acc: c.Accum, order: Order, t: []const f64) f64 {
    return switch (order) {
        .sequential => chain(acc, t, 1, 0),
        .pairwise => halves(acc, t),
        .strided => blk: {
            var lanes: [32]f64 = @splat(0);
            for (&lanes, 0..) |*l, i| l.* = chain(acc, t, 32, i);
            break :blk halves(acc, &lanes);
        },
    };
}

fn round(acc: c.Accum, v: f64) f64 {
    return switch (acc) {
        .f64, .none => v,
        .f32 => @as(f32, @floatCast(v)),
        .f16 => @as(f16, @floatCast(v)),
        .bf16 => c.fromBf16(c.toBf16(@floatCast(v))),
    };
}

fn chain(acc: c.Accum, t: []const f64, stride: usize, first: usize) f64 {
    var s: f64 = 0;
    var i = first;
    while (i < t.len) : (i += stride) s = round(acc, s + round(acc, t[i]));
    return s;
}

fn halves(acc: c.Accum, t: []const f64) f64 {
    if (t.len == 0) return 0;
    if (t.len == 1) return round(acc, t[0]);
    const mid = t.len / 2;
    return round(acc, halves(acc, t[0..mid]) + halves(acc, t[mid..]));
}

/// A row op's host inputs: x [rows, k], w and b [k] as bf16 bits; the norms, softmax and GELU write k outputs a row.
pub const Rows = struct {
    a: std.mem.Allocator,
    op: Op,
    shape: c.Shape,
    x: []u16,
    w: []u16,
    b: []u16,
    eps: f32 = 1e-6,

    pub fn init(a: std.mem.Allocator, op: Op, shape: c.Shape) !Rows {
        const x = try a.alloc(u16, shape.rows * shape.k);
        errdefer a.free(x);
        const w = try a.alloc(u16, shape.k);
        errdefer a.free(w);
        const b = try a.alloc(u16, shape.k);
        @memset(b, 0);
        return .{ .a = a, .op = op, .shape = shape, .x = x, .w = w, .b = b, .eps = if (op == .layer_norm) 1e-5 else 1e-6 };
    }

    pub fn deinit(r: Rows) void {
        r.a.free(r.x);
        r.a.free(r.w);
        r.a.free(r.b);
    }

    pub fn outputs(r: Rows) usize {
        return r.shape.rows * (switch (r.op) {
            .rms, .softmax, .layer_norm, .gelu => r.shape.k,
            else => 1,
        });
    }

    pub fn fill(r: Rows, mode: Fill, seed: u64) void {
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const k = r.shape.k;
        for (r.w) |*v| v.* = c.toBf16(if (r.op == .dot) rng.float(f32) * 2 - 1 else 0.5 + rng.float(f32));
        if (r.op == .layer_norm) for (r.b) |*v| {
            v.* = c.toBf16(rng.float(f32) * 2 - 1);
        };
        for (r.x, 0..) |*v, i| v.* = c.toBf16(switch (r.op) {
            .softmax => rng.float(f32) * 16 - 8,
            .gelu => rng.float(f32) * 12 - 6,
            else => (rng.float(f32) * 2 - 1) * (if (i % 64 == 7) @as(f32, 20) else 1),
        });
        if (mode == .stress) return r.stress(rng);
        if (mode != .nonfinite) return;
        const nan = c.toBf16(std.math.nan(f32));
        const inf = c.toBf16(std.math.inf(f32));
        const ninf = c.toBf16(-std.math.inf(f32));
        const at = [_][2]usize{ .{ 0, k / 3 }, .{ 1, k / 2 }, .{ 2, k - 1 }, .{ 3, 0 }, .{ 3, k - 1 } };
        const put = [_]u16{ nan, inf, ninf, inf, ninf };
        for (at, put) |p, v| if (p[0] < r.shape.rows) {
            r.x[p[0] * k + p[1]] = v;
        };
    }

    /// Gate 4's rows: integer terms, the second half mirroring the first (8 nudged by one): big partials, small result.
    fn stress(r: Rows, rng: std.Random) void {
        const k = r.shape.k;
        const h = k / 2;
        if (r.op == .layer_norm) return r.mirror(rng);
        for (0..r.shape.rows) |row| {
            const x = r.x[row * k ..][0..k];
            for (0..h) |i| {
                const v: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 64));
                const nudge: f32 = if (i < 8) 1 else 0;
                x[i] = c.toBf16(if (r.op == .rms) v - 32 else v);
                x[h + i] = c.toBf16(switch (r.op) {
                    .sum => nudge - v,
                    .rms => @as(f32, @floatFromInt(rng.uintLessThan(u32, 65))) - 32,
                    else => v + nudge,
                });
            }
            if (k % 2 == 1) x[k - 1] = 0;
        }
        if (r.op != .dot) return;
        for (0..h) |i| {
            const w: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 7));
            r.w[i] = c.toBf16(w);
            r.w[h + i] = c.toBf16(-w);
        }
        if (k % 2 == 1) r.w[k - 1] = 0;
    }

    /// Layer norm's: integers 1 to 32 and their negations, so the mean is 0 and both passes are exact in fp32 sums.
    fn mirror(r: Rows, rng: std.Random) void {
        const k = r.shape.k;
        for (0..r.shape.rows) |row| {
            const x = r.x[row * k ..][0..k];
            for (0..k / 2) |i| {
                const v: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 32));
                x[i] = c.toBf16(v);
                x[k / 2 + i] = c.toBf16(-v);
            }
            if (k % 2 == 1) x[k - 1] = 0;
        }
    }

    fn bv(r: Rows, i: usize) f64 {
        return c.fromBf16(r.b[i]);
    }

    fn xv(r: Rows, row: usize, i: usize) f64 {
        return c.fromBf16(r.x[row * r.shape.k + i]);
    }

    fn wv(r: Rows, i: usize) f64 {
        return c.fromBf16(r.w[i]);
    }

    /// Row `row`'s summed terms (rms: squares, softmax: exp(x - max), dot: x w, sum: x) into `t`; max has none.
    fn terms(r: Rows, row: usize, t: []f64) f64 {
        var top: f64 = -std.math.floatMax(f64);
        if (r.op == .softmax) for (0..r.shape.k) |i| {
            const v = r.xv(row, i);
            if (std.math.isNan(v) or v > top) top = v;
        };
        for (t, 0..) |*o, i| {
            const v = r.xv(row, i);
            o.* = switch (r.op) {
                .rms => v * v,
                .softmax => @exp(v - top),
                .dot => v * r.wv(i),
                .sum, .layer_norm => v,
                .max, .gelu => 0,
            };
        }
        return top;
    }

    /// Float64 outputs and, per output, the magnitude whose gamma(depth) bounds the error the sums carry into it.
    pub fn reference(r: Rows, want: []f64, mag: []f64) !void {
        const k = r.shape.k;
        const t = try r.a.alloc(f64, k);
        defer r.a.free(t);
        for (0..r.shape.rows) |row| {
            const top = r.terms(row, t);
            var sum: f64 = 0;
            var abs: f64 = 0;
            for (t) |v| {
                sum += v;
                abs += @abs(v);
            }
            switch (r.op) {
                .rms => {
                    const xs = try r.a.alloc(f64, k);
                    defer r.a.free(xs);
                    const ws = try r.a.alloc(f64, k);
                    defer r.a.free(ws);
                    for (xs, ws, 0..) |*x, *w, i| {
                        x.* = r.xv(row, i);
                        w.* = r.wv(i);
                    }
                    norms_ref.rms(xs, ws, r.eps, want[row * k ..][0..k]);
                    for (mag[row * k ..][0..k], want[row * k ..][0..k]) |*m, y| m.* = @abs(y) / 2; // y ~ ss^-1/2
                },
                .softmax => for (0..k) |i| {
                    want[row * k + i] = @exp(r.xv(row, i) - top) / sum;
                    mag[row * k + i] = @abs(want[row * k + i]);
                },
                .sum, .dot => {
                    want[row] = sum;
                    mag[row] = abs;
                },
                .layer_norm => r.layerNorm(row, sum, want[row * k ..][0..k], mag[row * k ..][0..k], abs),
                .gelu => for (0..k) |i| {
                    want[row * k + i] = gelu(r.xv(row, i));
                    mag[row * k + i] = 0;
                },
                .max => {
                    var m: f64 = -std.math.inf(f64);
                    for (0..k) |i| {
                        const v = r.xv(row, i);
                        if (std.math.isNan(v) or std.math.isNan(m)) m = std.math.nan(f64) else m = @max(m, v);
                    }
                    want[row] = m;
                    mag[row] = 0;
                },
            }
        }
    }

    /// Affine layer norm of row `row` (its terms' sum given), biased variance; mag bounds what the two sums carry.
    fn layerNorm(r: Rows, row: usize, sum: f64, want: []f64, mag: []f64, abs: f64) void {
        const k = r.shape.k;
        const kf: f64 = @floatFromInt(k);
        const mean = sum / kf;
        var q: f64 = 0;
        for (0..k) |i| q += (r.xv(row, i) - mean) * (r.xv(row, i) - mean);
        const inv = 1 / @sqrt(q / kf + r.eps);
        for (want, mag, 0..) |*w, *m, i| {
            const d = r.xv(row, i) - mean;
            w.* = d * inv * r.wv(i) + r.bv(i);
            // the mean's sum moves every d by its error; the variance's moves inv by half its relative error
            m.* = @abs(r.wv(i)) * inv * (abs / kf + @abs(d));
        }
    }

    /// The outputs if every row's sums ran in `acc` in `order`, the rest of the arithmetic exact.
    pub fn simulate(r: Rows, acc: c.Accum, order: Order, out: []f64) !void {
        const k = r.shape.k;
        const t = try r.a.alloc(f64, k);
        defer r.a.free(t);
        for (0..r.shape.rows) |row| {
            const top = r.terms(row, t);
            const s = sumIn(acc, order, t);
            switch (r.op) {
                .rms => {
                    const inv = 1 / @sqrt(s / @as(f64, @floatFromInt(k)) + r.eps);
                    for (0..k) |i| out[row * k + i] = r.xv(row, i) * inv * r.wv(i);
                },
                .softmax => for (0..k) |i| {
                    out[row * k + i] = @exp(r.xv(row, i) - top) / s;
                },
                .sum, .dot => out[row] = s,
                .layer_norm => {
                    // the variance's squares about the simulated mean, summed the same way
                    const mean = s / @as(f64, @floatFromInt(k));
                    for (t, 0..) |*o, i| o.* = (r.xv(row, i) - mean) * (r.xv(row, i) - mean);
                    const q = sumIn(acc, order, t);
                    const inv = 1 / @sqrt(q / @as(f64, @floatFromInt(k)) + r.eps);
                    for (0..k) |i| out[row * k + i] = (r.xv(row, i) - mean) * inv * r.wv(i) + r.bv(i);
                },
                .max, .gelu => return error.NoSums,
            }
        }
    }
};

/// The exact GELU x / 2 (1 + erf(x / sqrt 2)) in float64, as the kernel's formula (NaN where -inf meets 0).
pub fn gelu(x: f64) f64 {
    return 0.5 * x * (1 + act_ref.erf(x / std.math.sqrt2));
}

/// What an fp32 erff of 2 ulps adds to |gelu(x) - gelu64(x)|: x / 2 times its absolute error near +-1.
pub fn geluSlack(x: f64) f64 {
    return 0.5 * @abs(x) * 0x1p-22;
}

test "layer norm stress rows: the mean is 0 and both passes are exact in fp32 sums, bf16 sums miss" {
    const a = std.testing.allocator;
    const r = try Rows.init(a, .layer_norm, .{ .rows = 2, .n = 4096, .k = 4096 });
    defer r.deinit();
    r.fill(.stress, 5);
    const want = try a.alloc(f64, r.outputs());
    defer a.free(want);
    const mag = try a.alloc(f64, r.outputs());
    defer a.free(mag);
    const got = try a.alloc(f64, r.outputs());
    defer a.free(got);
    try r.reference(want, mag);
    for ([_]Order{ .sequential, .pairwise, .strided }) |o| {
        try r.simulate(.f32, o, got);
        try std.testing.expectEqualSlices(f64, want, got);
        try r.simulate(.bf16, o, got);
        try std.testing.expect(!std.mem.eql(f64, want, got));
    }
    try std.testing.expect(std.math.isNan(gelu(-std.math.inf(f64))) and gelu(std.math.inf(f64)) == std.math.inf(f64));
}

test "the stress rows: fp32 sums exact in every order, fp16 and bf16 sums far off a small result" {
    const a = std.testing.allocator;
    for ([_]Op{ .dot, .sum }) |op| {
        const r = try Rows.init(a, op, .{ .rows = 3, .n = 1, .k = 4096 });
        defer r.deinit();
        r.fill(.stress, 3);
        var want: [3]f64 = undefined;
        var mag: [3]f64 = undefined;
        try r.reference(&want, &mag);
        var got: [3]f64 = undefined;
        for ([_]Order{ .sequential, .pairwise, .strided }) |o| {
            try r.simulate(.f32, o, &got);
            try std.testing.expectEqualSlices(f64, &want, &got);
            for ([_]c.Accum{ .f16, .bf16 }) |acc| {
                try r.simulate(acc, o, &got);
                var most: f64 = 0; // the worst row's miss in half bf16 ulps of its result
                for (got, want) |g, w| most = @max(most, if (std.math.isFinite(g)) @abs(g - w) / (0x1p-8 * @abs(w) + 0x1p-133) else std.math.inf(f64));
                try std.testing.expect(most >= 100);
            }
        }
        for (want) |w| try std.testing.expect(@abs(w) <= 8 * 7 and w != 0 or op == .sum);
        try std.testing.expect(mag[0] > 0x1p17 and mag[0] < 0x1p23);
    }
}

test "planted non-finites reach the reference: NaN, both infinities, and an infinity cancelling to NaN" {
    const a = std.testing.allocator;
    const r = try Rows.init(a, .max, .{ .rows = 5, .n = 1, .k = 96 });
    defer r.deinit();
    r.fill(.nonfinite, 9);
    var want: [5]f64 = undefined;
    var mag: [5]f64 = undefined;
    try r.reference(&want, &mag);
    try std.testing.expectEqual(c.Class.nan, c.classOf(want[0]));
    try std.testing.expectEqual(c.Class.pos_inf, c.classOf(want[1]));
    try std.testing.expectEqual(c.Class.finite, c.classOf(want[2]));
    try std.testing.expectEqual(c.Class.pos_inf, c.classOf(want[3]));
    try std.testing.expectEqual(c.Class.finite, c.classOf(want[4]));
}

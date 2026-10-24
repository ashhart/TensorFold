//! Independent float64 tower points/ranges at the BF16 nodes, and exact-sum precision fixtures.
const std = @import("std");
const core = @import("core");
const bf = core.bf16_round;
const c = core.contract.spec;
const sums = core.contract.rows;

pub const Kind = enum { qkv, gelu, pool };
pub const heads = 16;
pub const dim = 72;
pub const width = heads * dim;

pub const Input = struct {
    kind: Kind,
    shape: c.Shape,
    values: [5][]const u16,
    exact_sums: bool = false,
};

// The contract's k is the reduction length (nine); the served 266-token grid is 14 by19 blocks.
pub fn poolWidth(tokens: usize) usize {
    return if (tokens % 19 == 0) 19 else 1;
}

fn gamma(n: f64) f64 {
    return c.gamma(n, c.unit(.f32));
}

fn f(x: u16) f64 {
    return bf.value(x);
}

fn pair(a: f64, b: f64) [2]f64 {
    if (std.math.isNan(a) or std.math.isNan(b)) return @splat(std.math.nan(f64));
    return .{ @min(a, b), @max(a, b) };
}

fn roundedProduct(lo: f64, hi: f64, weight: f64) [2]f64 {
    return pair(bf.bf16(lo * weight), bf.bf16(hi * weight));
}

fn norm(x: []const u16, weight: ?[]const u16, acc: c.Accum, order: sums.Order, exact: bool, point: []f64, low: []f64, high: []f64) void {
    var terms: [dim]f64 = undefined;
    for (&terms, x) |*t, v| t.* = f(v) * f(v); // BF16 products fit exactly in FP32 at these fixture magnitudes.
    const ss = sums.sumIn(acc, order, &terms);
    const inv = 1 / @sqrt(ss / dim + @as(f64, @as(f32, 1e-6)));
    // Up to eight sum roundings and six inverse/product roundings. Stress proves the sums exact in every order.
    const spread = gamma(if (exact) 8 else 16);
    for (x, point, low, high, 0..) |v, *p, *l, *h, i| {
        const w = if (weight) |weights| f(weights[i]) else 1;
        const z = f(v) * inv * w;
        p.* = bf.bf16(z);
        const range = pair(bf.bf16(z * (1 - spread)), bf.bf16(z * (1 + spread)));
        l.* = range[0];
        h.* = range[1];
    }
}

/// Each output: an independent FP64 point and endpoints allowing only FP32 operations at the declared BF16 nodes.
pub fn each(in: Input, acc: c.Accum, order: sums.Order, ctx: anytype, comptime emit: fn (@TypeOf(ctx), usize, f64, f64, f64) anyerror!void) !void {
    const values = in.values;
    switch (in.kind) {
        .qkv => for (0..in.shape.rows) |row| {
            for (0..3) |part| for (0..heads) |head| {
                const at = (row * 3 + part) * width + head * dim;
                var n: [dim]f64 = undefined;
                var lo: [dim]f64 = undefined;
                var hi: [dim]f64 = undefined;
                norm(values[0][at..][0..dim], if (part == 2) null else values[1 + part], acc, order, in.exact_sums, &n, &lo, &hi);
                for (0..dim) |d| {
                    var point = n[d];
                    var low = lo[d];
                    var high = hi[d];
                    if (part != 2) {
                        const half = dim / 2;
                        const quarter = half / 2;
                        const j = d % half;
                        const other = d / half * half + (if (j < quarter) j + quarter else j - quarter);
                        const sign: f64 = if (j < quarter) -1 else 1;
                        const cosine = f(values[3][row * dim + d]);
                        const sine = f(values[4][row * dim + d]);
                        const a = roundedProduct(lo[d], hi[d], cosine);
                        const b = if (sign < 0) roundedProduct(-hi[other], -lo[other], sine) else roundedProduct(lo[other], hi[other], sine);
                        point = bf.bf16(bf.bf16(n[d] * cosine) + bf.bf16(sign * n[other] * sine));
                        // The two rounded BF16 products are added in FP32 before the final BF16 store.
                        const error_ = gamma(1) * (@max(@abs(a[0]), @abs(a[1])) + @max(@abs(b[0]), @abs(b[1])));
                        low = bf.bf16(a[0] + b[0] - error_);
                        high = bf.bf16(a[1] + b[1] + error_);
                    }
                    try emit(ctx, at + d, point, low, high);
                }
            };
        },
        .pool => for (0..in.shape.rows) |token| {
            const bw = poolWidth(in.shape.rows);
            for (0..width) |ch| {
                var terms: [9]f64 = undefined;
                for (0..3) |dy| for (0..3) |dx| {
                    const at = ((3 * (token / bw) + dy) * (3 * bw) + 3 * (token % bw) + dx) * width + ch;
                    // Freeze the producer's explicitly rounded FP32 multiply, then sum independently in FP64.
                    terms[3 * dy + dx] = @as(f32, @floatCast(f(values[0][at]) * @as(f64, @as(f32, 1.0 / 9.0))));
                };
                const total = sums.sumIn(acc, order, &terms);
                var magnitude: f64 = 0;
                for (terms) |term| magnitude += @abs(term);
                const error_ = if (in.exact_sums) 0 else gamma(9) * magnitude;
                const root: f64 = @as(f32, @floatCast(@sqrt(@as(f64, width))));
                const bias = f(values[1][ch]);
                const scale = f(values[2][ch]);
                const mean = bf.bf16(total);
                const mean_lo = bf.bf16(total - error_);
                const mean_hi = bf.bf16(total + error_);
                const ends = pair((mean_lo * root - bias) * scale, (mean_hi * root - bias) * scale);
                const post = gamma(3) * (@max(@abs(mean_lo), @abs(mean_hi)) * root + @abs(bias)) * @abs(scale);
                const point = bf.bf16((mean * root - bias) * scale);
                try emit(ctx, token * width + ch, point, bf.bf16(ends[0] - post), bf.bf16(ends[1] + post));
            }
        },
        .gelu => for (values[0], values[1], 0..) |x, up, i| {
            const point = gelu(f(x), f(up));
            try emit(ctx, i, point, point, point);
        },
    }
}

/// The special-value rule follows the same formula in FP64: in particular gelu(-Inf) is NaN, not zero.
pub fn gelu(x: f64, up: f64) f64 {
    const beta: f64 = @as(f32, @floatCast(std.math.sqrt2 * std.math.two_sqrtpi * 0.5));
    const inner = beta * (x + @as(f64, @as(f32, 0.044715)) * x * x * x);
    return bf.bf16(bf.bf16(0.5 * x * (1 + std.math.tanh(inner))) * up);
}

pub fn fillStress(kind: Kind, tokens: usize, values: [5][]u16, rng: std.Random) void {
    switch (kind) {
        .qkv => {
            for (values[0]) |*x| x.* = bf.bits(@floatFromInt(128 + rng.uintLessThan(u32, 128)));
            for (values[1..3]) |weights| {
                for (weights) |*w| w.* = bf.bits(0.5 + rng.float(f32) * 2);
            }
            @memset(values[3], bf.bits(1));
            @memset(values[4], bf.bits(0));
        },
        .pool => {
            const terms = [_]f32{ 0x1p20, 1, 1, 1, -0x1p20, 1, 1, 1, 1 };
            const bw = poolWidth(tokens);
            for (0..tokens) |token| for (0..width) |ch| for (0..3) |dy| for (0..3) |dx| {
                const at = ((3 * (token / bw) + dy) * (3 * bw) + 3 * (token % bw) + dx) * width + ch;
                values[0][at] = bf.bits(9 * terms[(3 * dy + dx + ch) % 9]);
            };
            @memset(values[1], bf.bits(0));
            @memset(values[2], bf.bits(1));
        },
        .gelu => {},
    }
}

const Write = struct { point: []f64, low: ?[]f64, high: ?[]f64 };
fn write(w: Write, i: usize, point: f64, low: f64, high: f64) !void {
    w.point[i] = point;
    if (w.low) |v| v[i] = low;
    if (w.high) |v| v[i] = high;
}

pub fn outputs(in: Input, acc: c.Accum, order: sums.Order, point: []f64, low: ?[]f64, high: ?[]f64) !void {
    try each(in, acc, order, Write{ .point = point, .low = low, .high = high }, write);
}

pub fn inside(point: f64, low: f64, high: f64, bits: u16) bool {
    const got = f(bits);
    if (!std.math.isFinite(point)) return c.classOf(point) == c.classOf(got);
    return std.math.isFinite(got) and got >= low and got <= high;
}

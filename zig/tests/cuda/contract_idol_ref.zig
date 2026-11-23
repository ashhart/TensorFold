//! Idol's CUDA kernel tests' shared parts: float64 values at the kernels' fp16 points, and every value they may take.
const std = @import("std");
const core = @import("core");

pub const c = core.contract.spec;
pub const g = core.contract.gates;
pub const rows = core.contract.rows;

pub const hidden = 1024;
pub const heads = 16;
pub const D = 64;
pub const ffn = 4096;
/// The served row counts: the joint rows, the noise refiner's latent rows, the condition refiner's rows.
pub const joint = 12294;
pub const latent = 8192;
pub const cond = 4101;

// nvcc's error bounds under the glue flags: IEEE division and square root; expf, tanhf, cosf and sinf 2 ulps
pub const div_ulps = 0.5;
pub const sqrt_ulps = 0.5;
pub const fn_ulps = 2;

/// An input's element type on the device.
pub const Kind = enum { f16, f32 };

/// A value as fp32, then fp16, round to nearest even (overflow to infinity).
pub fn h(v: f64) f64 {
    return @as(f16, @floatCast(@as(f32, @floatCast(v))));
}

/// A value as fp32.
pub fn f(v: f64) f64 {
    return @as(f32, @floatCast(v));
}

/// fmaxf: a NaN operand gives the other.
pub fn fmax(a: f64, b: f64) f64 {
    if (std.math.isNan(a)) return b;
    if (std.math.isNan(b)) return a;
    return @max(a, b);
}

/// Every value the kernel may hold at a step: its fp32 steps widen both ends, its fp16 points round them.
pub const Range = struct {
    lo: f64,
    hi: f64,

    pub fn of(v: f64) Range {
        return .{ .lo = v, .hi = v };
    }
    /// An fp32 step good to `ulps` units in the last place (a correctly rounded one: 0.5); infinities stay.
    pub fn fp32(a: Range, ulps: f64) Range {
        const u = ulps * 0x1p-23;
        return .{ .lo = if (std.math.isFinite(a.lo)) a.lo - u * @abs(a.lo) - 0x1p-149 else a.lo, .hi = if (std.math.isFinite(a.hi)) a.hi + u * @abs(a.hi) + 0x1p-149 else a.hi };
    }
    pub fn half(a: Range) Range {
        return .{ .lo = h(a.lo), .hi = h(a.hi) };
    }
    pub fn add(a: Range, b: Range) Range {
        return .{ .lo = a.lo + b.lo, .hi = a.hi + b.hi };
    }
    pub fn sub(a: Range, b: Range) Range {
        return .{ .lo = a.lo - b.hi, .hi = a.hi - b.lo };
    }
    pub fn mul(a: Range, b: Range) Range {
        const p = [4]f64{ a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi };
        return .{ .lo = @min(@min(p[0], p[1]), @min(p[2], p[3])), .hi = @max(@max(p[0], p[1]), @max(p[2], p[3])) };
    }
    /// a / b for b of one sign.
    pub fn div(a: Range, b: Range) Range {
        return a.mul(.{ .lo = 1 / b.hi, .hi = 1 / b.lo });
    }
    /// An increasing function at both ends.
    pub fn up(a: Range, comptime fun: fn (f64) f64) Range {
        return .{ .lo = fun(a.lo), .hi = fun(a.hi) };
    }
    /// fmaxf against a constant.
    pub fn atLeast(a: Range, v: f64) Range {
        return .{ .lo = @max(a.lo, v), .hi = @max(a.hi, v) };
    }
    /// x^2 over the range.
    pub fn square(a: Range) Range {
        const lo = if (a.lo <= 0 and a.hi >= 0) 0 else @min(a.lo * a.lo, a.hi * a.hi);
        return .{ .lo = lo, .hi = @max(a.lo * a.lo, a.hi * a.hi) };
    }
    /// cos or sin (`sine`) over the range, its extrema included when one lies inside.
    pub fn trig(a: Range, sine: bool) Range {
        if (!(std.math.isFinite(a.lo) and std.math.isFinite(a.hi)) or a.hi - a.lo >= 2 * std.math.pi) return .{ .lo = -1, .hi = 1 };
        const fun: *const fn (f64) f64 = if (sine) sinf64 else cosf64;
        var lo = @min(fun(a.lo), fun(a.hi));
        var hi = @max(fun(a.lo), fun(a.hi));
        // extrema at k pi (cos) or pi / 2 + k pi (sin)
        const shift: f64 = if (sine) std.math.pi / 2.0 else 0;
        var k = @ceil((a.lo - shift) / std.math.pi);
        while (k * std.math.pi + shift <= a.hi) : (k += 1) {
            const v = fun(k * std.math.pi + shift);
            lo = @min(lo, v);
            hi = @max(hi, v);
        }
        return .{ .lo = lo, .hi = hi };
    }
};

fn sinf64(x: f64) f64 {
    return @sin(x);
}
fn cosf64(x: f64) f64 {
    return @cos(x);
}

/// How a reference forms its sums: in float64 with their fp32 range (exact for gate 4's inputs), or in an accumulator.
pub const Sum = union(enum) {
    range: bool,
    sim: struct { acc: c.Accum, order: g.Order },

    /// The terms' sum and its range, `lo` and `hi` each term's own ends, `depth` fp32 roundings deep.
    pub fn total(s: Sum, terms: []const f64, lo: []const f64, hi: []const f64, depth: usize) struct { v: f64, r: Range } {
        switch (s) {
            .sim => |x| {
                const v = rows.sumIn(x.acc, x.order, terms);
                return .{ .v = v, .r = Range.of(v) };
            },
            .range => |exact| {
                var v: f64 = 0;
                var a: f64 = 0;
                var b: f64 = 0;
                var m: f64 = 0;
                for (terms, lo, hi) |t, l, u| {
                    v += t;
                    a += l;
                    b += u;
                    m += @max(@abs(l), @abs(u));
                }
                if (exact) return .{ .v = v, .r = Range.of(v) };
                const e = c.gamma(@floatFromInt(depth), c.unit(.f32)) * m;
                return .{ .v = v, .r = .{ .lo = a - e, .hi = b + e } };
            },
        }
    }
};

/// Each output's float64 value at the kernel's rounding points, and its range, kept for the bounds and intervals.
pub const Ends = struct {
    a: std.mem.Allocator,
    lo: []f64 = &.{},
    hi: []f64 = &.{},

    pub fn deinit(e: *Ends) void {
        e.a.free(e.lo);
        e.a.free(e.hi);
        e.lo = &.{};
        e.hi = &.{};
    }

    pub fn size(e: *Ends, n: usize) !void {
        e.deinit();
        e.lo = try e.a.alloc(f64, n);
        e.hi = try e.a.alloc(f64, n);
    }

    pub fn put(e: *Ends, want: []f64, mag: []f64, i: usize, w: f64, r: Range) void {
        want[i] = w;
        mag[i] = if (std.math.isFinite(w)) @abs(w) else 0;
        e.lo[i] = r.lo;
        e.hi[i] = r.hi;
    }

    /// The allowance about the float64 value: its range's farther end (a non-finite value is checked by class).
    pub fn bounds(e: *const Ends, want: []const f64, allow: []f64) void {
        for (allow, want, e.lo, e.hi) |*l, w, lo, hi| l.* = if (std.math.isFinite(w) and std.math.isFinite(lo) and std.math.isFinite(hi)) @max(@abs(w - lo), @abs(hi - w)) else 0;
    }

    pub fn intervals(e: *const Ends, low: []f64, high: []f64) void {
        @memcpy(low, e.lo);
        @memcpy(high, e.hi);
    }
};

/// Values uniform in [lo, hi].
pub fn uniform(rng: std.Random, out: []f64, lo: f64, hi: f64) void {
    for (out) |*v| v.* = lo + rng.float(f64) * (hi - lo);
}

/// Integers in [-bound, bound].
pub fn integers(rng: std.Random, out: []f64, bound: u32) void {
    for (out) |*v| v.* = @as(f64, @floatFromInt(rng.uintLessThan(u32, 2 * bound + 1))) - @as(f64, @floatFromInt(bound));
}

pub const nan = std.math.nan(f64);
pub const inf = std.math.inf(f64);

/// Plants NaN, +inf and -inf at `at` (indices taken modulo the length).
pub fn plant(out: []f64, at: []const usize) void {
    const vals = [_]f64{ nan, inf, -inf };
    for (at, 0..) |i, j| out[i % out.len] = vals[j % 3];
}

test "fp16 points round to nearest even, ranges keep the exact value inside" {
    try std.testing.expectEqual(@as(f64, 1.0), h(1.0 + 1.0 / 4096.0));
    try std.testing.expectEqual(inf, h(70000));
    const r = Range.of(0.3).fp32(0.5).mul(Range.of(1.7)).fp32(0.5).half();
    try std.testing.expect(r.lo <= h(0.3 * 1.7) and h(0.3 * 1.7) <= r.hi);
    const t = (Range{ .lo = 3.1, .hi = 3.2 }).trig(false);
    try std.testing.expectEqual(@as(f64, -1), t.lo);
}

test "a sum's range holds the exact sum, and is the sum itself for gate 4's exact inputs" {
    const t = [_]f64{ 1.5, -2.25, 3 };
    const s = (Sum{ .range = false }).total(&t, &t, &t, 3);
    try std.testing.expect(s.r.lo < 2.25 and 2.25 < s.r.hi);
    const e = (Sum{ .range = true }).total(&t, &t, &t, 3);
    try std.testing.expectEqual(e.r.lo, e.r.hi);
}

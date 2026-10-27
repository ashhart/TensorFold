//! MOSS-Image's kernel tests' shared parts: float64 values at bf16 points, their ranges, the launches' buffers.
const std = @import("std");
const mtl = @import("metal");
pub const tf = @import("tensorfold");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");

pub const c = contract.spec;
pub const g = contract.gates;
pub const ks = tf.moss_image.kernels;
pub const cfg = tf.moss_image.config;
pub const eps: f64 = cfg.eps;
pub const D = cfg.head_dim;

pub fn noSums(_: c.Shape) usize {
    return 0;
}

/// A value as fp32 then bf16 round it.
pub fn bf(v: f64) f64 {
    return c.fromBf16(c.toBf16(@floatCast(v)));
}

pub fn kref(r: rig_.Ref) ks.Ref {
    return .{ .buf = r.buf, .off = r.off };
}

pub fn kat(r: rig_.Ref, bytes: usize) ks.Ref {
    return .{ .buf = r.buf, .off = r.off + bytes };
}

/// Every value the kernel may hold at a step: its fp32 steps widen both ends, its bf16 points round them.
pub const Range = struct {
    lo: f64,
    hi: f64,
    pub fn of(v: f64) Range {
        return .{ .lo = v, .hi = v };
    }
    /// An fp32 step good to `ulps` units in the last place (a correctly rounded one: 0.5).
    pub fn fp32(a: Range, ulps: f64) Range {
        const u = ulps * 0x1p-23;
        return .{ .lo = a.lo - u * @abs(a.lo) - 0x1p-149, .hi = a.hi + u * @abs(a.hi) + 0x1p-149 };
    }
    pub fn bf16(a: Range) Range {
        return .{ .lo = bf(a.lo), .hi = bf(a.hi) };
    }
    pub fn add(a: Range, b: Range) Range {
        return .{ .lo = a.lo + b.lo, .hi = a.hi + b.hi };
    }
    pub fn mul(a: Range, b: Range) Range {
        const p = [4]f64{ a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi };
        return .{ .lo = @min(@min(p[0], p[1]), @min(p[2], p[3])), .hi = @max(@max(p[0], p[1]), @max(p[2], p[3])) };
    }
    /// An increasing function at both ends.
    pub fn up(a: Range, comptime f: fn (f64) f64) Range {
        return .{ .lo = f(a.lo), .hi = f(a.hi) };
    }
    /// 1 / sqrt(a), a positive.
    pub fn rsqrt(a: Range) Range {
        return .{ .lo = 1 / @sqrt(a.hi), .hi = 1 / @sqrt(a.lo) };
    }
    /// 1 / a, a positive.
    pub fn recip(a: Range) Range {
        return .{ .lo = 1 / a.hi, .hi = 1 / a.lo };
    }
};

pub fn tanh(x: f64) f64 {
    return std.math.tanh(x);
}

// Metal's error bounds with the engine's options (fast math on, the kernels' own `precise::` calls).
pub const div_ulps = 2.5;
pub const rsqrt_ulps = 2;
pub const exp_ulps = 4;
pub const tanh_ulps = 5;

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
        for (allow, want, e.lo, e.hi) |*l, w, lo, hi| l.* = if (std.math.isFinite(w)) @max(@abs(w - lo), @abs(hi - w)) else 0;
    }

    pub fn intervals(e: *const Ends, low: []f64, high: []f64) void {
        @memcpy(low, e.lo);
        @memcpy(high, e.hi);
    }
};

pub fn randomBf16(rng: std.Random, out: []u16, lo: f32, hi: f32) void {
    for (out) |*v| v.* = c.toBf16(lo + rng.float(f32) * (hi - lo));
}

pub fn integers(rng: std.Random, out: []u16, bound: u32) void {
    for (out) |*v| v.* = c.toBf16(@as(f32, @floatFromInt(rng.uintLessThan(u32, 2 * bound + 1))) - @as(f32, @floatFromInt(bound)));
}

pub const nan16 = 0x7fc0;
pub const inf16 = 0x7f80;
pub const ninf16 = 0xff80;

/// A row's sum of squares in float64 and its range from fp32 sums `depth` deep (exact for gate 4's integers).
pub fn squares(x: []const u16, depth: usize, exact: bool) struct { ss: f64, r: Range } {
    var ss: f64 = 0;
    for (x) |v| {
        const f: f64 = c.fromBf16(v);
        ss += f * f;
    }
    if (exact) return .{ .ss = ss, .r = Range.of(ss) };
    const e = c.gamma(@floatFromInt(depth), c.unit(.f32)) * ss;
    return .{ .ss = ss, .r = .{ .lo = ss - e, .hi = ss + e } };
}

/// rsqrt(ss / n + eps) in float64 and its range through the kernel's division, add and precise rsqrt.
pub fn invNorm(ss: f64, r: Range, n: f64) struct { inv: f64, r: Range } {
    const mean = (Range{ .lo = r.lo / n, .hi = r.hi / n }).fp32(div_ulps);
    return .{ .inv = 1 / @sqrt(ss / n + eps), .r = mean.add(Range.of(eps)).fp32(0.5).rsqrt().fp32(rsqrt_ulps) };
}

//! The edit encoder suite's shared parts: fp32 step ranges and their ends, fills, launch constants and binding.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");

pub const c = contract.spec;
pub const g = contract.gates;

/// Metal's error bounds in ulps with the encoder's options (safe math, precise functions).
pub const div_ulps = 2.5;
pub const sqrt_ulps = 3;
pub const exp_ulps = 4;

/// GroupNorm's eps as the launcher passes it (picture_vae.zig's group_eps), and the pixels a part sums.
pub const eps: f32 = 1e-6;
pub const chunk = 256;

pub const nan = std.math.nan(f32);
pub const inf = std.math.inf(f32);

/// GroupNorm's parts for `pixels` (image_encoder.zig's `chunks`).
pub fn chunks(pixels: usize) usize {
    return (pixels + chunk - 1) / chunk;
}

/// The kernels' HieGroup constants for a pass over `pixels` of `channels`.
pub const HieGroup = extern struct { pixels: u32, channels: u32, chunk: u32, chunks: u32, centered: u32, eps: f32 };

pub fn group(channels: usize, pixels: usize, centered: bool) HieGroup {
    return .{ .pixels = @intCast(pixels), .channels = @intCast(channels), .chunk = chunk, .chunks = @intCast(chunks(pixels)), .centered = @intFromBool(centered), .eps = eps };
}

pub fn bind(e: mtl.ComputeEncoder, r: rig_.Ref, index: usize) void {
    e.setBuffer(r.buf, r.off, index);
}

pub fn uniform(rng: std.Random, out: []f32, lo: f32, hi: f32) void {
    for (out) |*v| v.* = lo + rng.float(f32) * (hi - lo);
}

pub fn int(rng: std.Random, lo: i32, hi: i32) f32 {
    return @floatFromInt(rng.intRangeAtMost(i32, lo, hi));
}

pub fn noSums(_: c.Shape) usize {
    return 0;
}

/// Every value an fp32 step may give: each step widens both ends by its ulps.
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
    pub fn add(a: Range, b: Range) Range {
        return .{ .lo = a.lo + b.lo, .hi = a.hi + b.hi };
    }
    pub fn mul(a: Range, b: Range) Range {
        const p = [4]f64{ a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi };
        return .{ .lo = @min(@min(p[0], p[1]), @min(p[2], p[3])), .hi = @max(@max(p[0], p[1]), @max(p[2], p[3])) };
    }
    /// 1 / a, a positive.
    pub fn recip(a: Range) Range {
        return .{ .lo = 1 / a.hi, .hi = 1 / a.lo };
    }
    /// sqrt(a), a not negative.
    pub fn sqrt(a: Range) Range {
        return .{ .lo = @sqrt(a.lo), .hi = @sqrt(a.hi) };
    }
};

/// v / (1 + exp(-v)) over v's range through the kernel's exp, add and division.
pub fn silu(v: Range) Range {
    const e = (Range{ .lo = @exp(-v.hi), .hi = @exp(-v.lo) }).fp32(exp_ulps);
    return v.mul(e.add(Range.of(1)).fp32(0.5).recip()).fp32(div_ulps);
}

/// Each output's float64 value and its range, kept for the bounds and intervals hooks.
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

/// A sum and its terms' magnitude, the magnitude 0 where it is not finite (the gate checks those by class).
pub fn total(terms: []const f64) struct { sum: f64, mag: f64 } {
    var s: f64 = 0;
    var m: f64 = 0;
    for (terms) |v| {
        s += v;
        m += @abs(v);
    }
    return .{ .sum = s, .mag = if (std.math.isFinite(m)) m else 0 };
}

//! Independent decoded-weight and BF16-node reference for Transformers PR #51050 at 6173a2f6.
const std = @import("std");
const u = @import("inkling2_util.zig");

pub const Format = enum {
    nvfp4,
    mxfp8,

    pub fn block(f: Format) usize {
        return if (f == .nvfp4) 16 else 32;
    }
};

pub fn fp32(x: f64) f64 {
    const r: f32 = @floatCast(x);
    return r;
}

const fp4_grid = [_]f64{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };

pub fn e2m1(bits: u8) f64 {
    const mag = fp4_grid[bits & 7];
    return if (bits & 8 == 0) mag else -mag;
}

pub fn e4m3(bits: u8) f64 {
    const v = switch (bits & 127) {
        127 => std.math.nan(f64),
        0...7 => @as(f64, @floatFromInt(bits & 7)) / 512,
        else => std.math.ldexp(@as(f64, @floatFromInt(8 + (bits & 7))), @as(i32, (bits & 127) / 8) - 10),
    };
    return if (bits & 128 == 0) v else -v;
}

// The converter evaluates exp2(byte - 127) in FP32; the reserved byte overflows to +Inf.
pub fn e8m0(bits: u8) f64 {
    return if (bits == 255) std.math.inf(f64) else std.math.ldexp(@as(f64, 1), @as(i32, bits) - 127);
}

/// At BF16 load the F32 metadata tensors are cast separately; the converter divides them in FP32.
pub fn factor(amax: f32, denominator: f32) f64 {
    return fp32(u.bf(amax) / u.bf(denominator));
}

/// The GEMM's operands are BF16, not the unrounded products of the checkpoint codes and scales.
pub fn weight(format: Format, code: u8, scale: u8, outer: f64) f64 {
    const decoded = switch (format) {
        .nvfp4 => fp32(e2m1(code) * e4m3(scale)) * outer,
        .mxfp8 => e4m3(code) * e8m0(scale),
    };
    return u.bf(fp32(decoded));
}

pub const Range = struct {
    lo: f64,
    hi: f64,

    pub fn point(v: f64) Range {
        return .{ .lo = v, .hi = v };
    }

    pub fn add(a: Range, b: Range) Range {
        return .{ .lo = fp32(a.lo + b.lo), .hi = fp32(a.hi + b.hi) };
    }

    pub fn mul(r: Range, v: f64) Range {
        const a = fp32(r.lo * v);
        const b = fp32(r.hi * v);
        return if (v < 0) .{ .lo = b, .hi = a } else .{ .lo = a, .hi = b };
    }

    pub fn bf16(r: Range) Range {
        return .{ .lo = u.bf(r.lo), .hi = u.bf(r.hi) };
    }
};

fn capped(v: f64, cap: f64) f64 {
    return std.math.tanh(v / cap) * cap;
}

/// The two projection stores have already rounded the inputs; the sigmoid sees the uncapped gate.
pub fn swiglu(gate: f64, up: f64, gate_cap: f64, up_cap: f64) f64 {
    return (capped(gate, gate_cap) / (1 + @exp(-gate))) * capped(up, up_cap);
}

pub fn activation(gate: f64, up: f64, gate_cap: f64, up_cap: f64) Range {
    const v = swiglu(gate, up, gate_cap, up_cap);
    if (!std.math.isFinite(v)) return .point(v);
    // Bound FP32 elementary and transcendental rounding before the BF16 store.
    const err = @abs(v) * (64 * 0x1p-24);
    return (Range{ .lo = v - err, .hi = v + err }).bf16();
}

test "Mini storage encodings preserve sign, subnormal and maximum codes" {
    try std.testing.expectEqual(@as(u64, 1) << 63, @as(u64, @bitCast(e2m1(8))));
    try std.testing.expectEqual(@as(f64, -0.5), e2m1(9));
    try std.testing.expectEqual(@as(f64, 0x1p-9), e4m3(1));
    try std.testing.expectEqual(@as(f64, -448), e4m3(0xfe));
    try std.testing.expect(std.math.isNan(e4m3(0x7f)));
    try std.testing.expectEqual(@as(f64, 0x1p-127), e8m0(0));
    try std.testing.expectEqual(@as(f64, 0x1p-119), weight(.mxfp8, 0x78, 0, 1));
    try std.testing.expectEqual(std.math.inf(f64), e8m0(255));
    try std.testing.expect(std.math.isNan(weight(.mxfp8, 0, 255, 1)));
}

test "Mini NVFP4 load rounds operands but not the outer factor" {
    const a = factor(1.003, 2693);
    try std.testing.expectEqual(fp32(1.0 / 2688.0), a);
    try std.testing.expect(a != u.bf(a));
    try std.testing.expectEqual(u.bf(fp32(1536.0 / 2688.0)), weight(.nvfp4, 7, 0x78, a));
    const fifth = factor(1.003, 5.003);
    try std.testing.expectEqual(@as(f64, 0.224609375), weight(.nvfp4, 2, 0x39, fifth));
    try std.testing.expectEqual(@as(f64, 0.2255859375), weight(.nvfp4, 2, 0x39, u.bf(fifth)));
}

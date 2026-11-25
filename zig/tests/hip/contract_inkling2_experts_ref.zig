//! The HIP experts' host side: loop_model.zig's expert arithmetic as the device must reproduce it, bit for bit.
const std = @import("std");
const lm = @import("inkling2_loop_model");
const bf = @import("bf16_round");

/// Every NaN the kernels write (experts.hpp kNan).
pub const nan_bits: u16 = 0x7fc0;

fn softcap(x: f32, cap: f32) f32 {
    return cap * std.math.tanh(x / cap);
}

/// softcap4(g) * sigmoid(g): loop_model.expertActivation's first product, with its f32 std.math.tanh and @exp.
pub fn gateFactor(g: f32) f32 {
    const sigmoid: f32 = 1 / (1 + @exp(-g));
    return softcap(g, 4) * sigmoid;
}

/// softcap25(u): expertActivation's last factor.
pub fn upFactor(u: f32) f32 {
    return softcap(u, 25);
}

/// A float64 reference value's bf16 bits as the kernels store it: NaN as nan_bits, else ties to even.
pub fn bits(v: f64) u16 {
    return if (std.math.isNan(v)) nan_bits else bf.bits(@floatCast(v));
}

/// Two fp32 outputs agree: both NaN, or the same bits.
pub fn same32(x: f32, y: f32) bool {
    return (std.math.isNan(x) and std.math.isNan(y)) or @as(u32, @bitCast(x)) == @as(u32, @bitCast(y));
}

test "loop_model's activation is one fp32 product of the two factors, rounded once: all 65,536 g by 64 u, and back" {
    var prng = std.Random.DefaultPrng.init(0x1a2b);
    var samples: [64]u16 = undefined;
    for (&samples, 0..) |*s, i| s.* = if (i < 8) ([_]u16{ 0, 0x8000, 0x7f80, 0xff80, 0x7fc0, 0x3f80, 0xc120, 0x0001 })[i] else prng.random().int(u16);
    for (0..65536) |i| for (samples) |j| for ([_][2]u16{ .{ @intCast(i), j }, .{ j, @intCast(i) } }) |p| {
        const g = bf.value(p[0]);
        const u = bf.value(p[1]);
        const want = bits(lm.expertActivation(g, u));
        const got = if (std.math.isNan(gateFactor(g) * upFactor(u))) nan_bits else bf.bits(gateFactor(g) * upFactor(u));
        if (want != got) {
            std.debug.print("g 0x{x:0>4} u 0x{x:0>4}: loop_model 0x{x:0>4}, factors 0x{x:0>4}\n", .{ p[0], p[1], want, got });
            return error.TestUnexpectedResult;
        }
    };
}

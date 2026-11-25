//! The HIP loop kernels' host model: the Zig f32 results their probe is held to.
const std = @import("std");
const K = @import("inkling2_loop_hip");

/// Two fp32 outputs agree: both NaN, or the same bits.
pub fn same32(x: f32, y: f32) bool {
    return (std.math.isNan(x) and std.math.isNan(y)) or @as(u32, @bitCast(x)) == @as(u32, @bitCast(y));
}

/// The baselines' f32 functions the probe runs on the device: Zig 0.17's std.math.tanh, @exp and 1 / @sqrt.
pub fn zig(op: K.Math, x: f32) f32 {
    return switch (op) {
        .tanh, .device_tanh => std.math.tanh(x),
        .exp, .device_exp => @exp(x),
        .inv_sqrt, .device_rsqrt => 1 / @sqrt(x),
    };
}

//! Activation rows in float64, the reference every backend's activation kernels are checked against.
const std = @import("std");

pub fn sigmoid(x: f64) f64 {
    return 1 / (1 + @exp(-x));
}

/// A 4-bit affine table's value: q * scale + bias.
pub fn affine4(q: u4, scale: f64, bias: f64) f64 {
    return @as(f64, @floatFromInt(q)) * scale + bias;
}

/// A bf16 output of fp32 arithmetic through expf: half a bf16 ulp plus 2^-18 of |want|.
pub fn tolerance(want: f64) f64 {
    return @abs(want) * (std.math.pow(f64, 2, -8) + std.math.pow(f64, 2, -18)) + 1e-30;
}

test "activations against hand values" {
    try std.testing.expectEqual(@as(f64, 0.5), sigmoid(0));
    try std.testing.expectEqual(@as(f64, 7 * 0.5 - 1), affine4(7, 0.5, -1));
    try std.testing.expect(tolerance(1) > tolerance(0.5));
}

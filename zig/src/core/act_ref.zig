//! Activation rows in float64, the reference every backend's activation kernels are checked against.
const std = @import("std");

pub fn sigmoid(x: f64) f64 {
    return 1 / (1 + @exp(-x));
}

/// silu(gate) * up.
pub fn swiglu(gate: f64, up: f64) f64 {
    return gate * sigmoid(gate) * up;
}

/// DFlash2's dynamic conv at one value: x k0 + prev k1 with the taps as rounded (k_t = bf16(base_t + dyn_t)).
pub fn dconv(x: f64, prev: f64, k0: f64, k1: f64) f64 {
    return x * k0 + prev * k1;
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
    try std.testing.expectApproxEqRel(@as(f64, 2 * sigmoid(2) * 3), swiglu(2, 3), 1e-15);
    try std.testing.expect(tolerance(1) > tolerance(0.5));
}

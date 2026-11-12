//! 1-D rotary inverse frequencies as HF's default rope computes them in float32; angles and cos/sin are device work.
const std = @import("std");

/// Frequencies a head of `dim` dims takes: one a rotated pair.
pub fn count(dim: usize) usize {
    return dim / 2;
}

/// out[i] = 1 / theta^(2i / dim): the exponent an f32 quotient, the power of the f32 operands rounded once to f32.
pub fn inverse(theta: f64, dim: usize, out: []f32) void {
    std.debug.assert(out.len == count(dim));
    const base: f32 = @floatCast(theta);
    for (out, 0..) |*v, i| {
        const e: f32 = @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(dim));
        const pw: f32 = @floatCast(std.math.pow(f64, base, e));
        v.* = @as(f32, 1.0) / pw;
    }
}

test "dots' inverse frequencies (theta 1e6, head dim 128) in float32" {
    var got: [64]f32 = undefined;
    inverse(1e6, 128, &got);
    for (got, 0..) |v, i| {
        const e = @as(f32, @floatFromInt(i * 2)) / 128.0;
        const want = @as(f32, 1.0) / @as(f32, @floatCast(std.math.pow(f64, @as(f32, 1e6), e)));
        try std.testing.expectEqual(@as(u32, @bitCast(want)), @as(u32, @bitCast(v)));
    }
    try std.testing.expectEqual(@as(f32, 1), got[0]);
    try std.testing.expectEqual(@as(f32, 1e-3), got[32]); // theta^(1/2): 1000 exactly, its f32 reciprocal
    try std.testing.expect(got[63] < got[62] and got[63] > 1e-6);
}

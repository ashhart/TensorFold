//! NVFP4 (W4A16, group 16) host dequantization: the recipe the golden fixture proved bit-exact
//! against modelopt's own unpack (nvfp4_tensor.py). Two e2m1 values per byte — even nibble first —
//! an e4m3 scale per 16-element block along the input dim, one f32 scale_2 for the tensor.
const std = @import("std");

/// The signed e2m1 table by nibble: {0, .5, 1, 1.5, 2, 3, 4, 6} and their negations.
pub const e2m1 = [16]f32{ 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0 };

/// e4m3fn byte to f32: sign, 4-bit exponent (bias 7, zero = subnormal), 3-bit mantissa; S.1111.111 = NaN.
pub fn e4m3ToFloat(bits: u8) f32 {
    const sign: f32 = if (bits & 0x80 != 0) -1.0 else 1.0;
    const exp: u8 = (bits >> 3) & 0xF;
    const mant: u8 = bits & 0x7;
    if (exp == 0) return sign * @as(f32, @floatFromInt(mant)) / 8.0 * 0.015625;
    if (exp == 15 and mant == 7) return std.math.nan(f32);
    return sign * (1.0 + @as(f32, @floatFromInt(mant)) / 8.0) * std.math.pow(f32, 2, @as(f32, @floatFromInt(exp)) - 7.0);
}

/// Dequantize a packed nvfp4 matrix into `dst` [rows, in] f32, row-major: `q` is rows x in/2 bytes,
/// `scales` rows x in/16 e4m3 bytes, `scale_2` the tensor's f32 scale. Shapes are checked loudly.
pub fn dequantNvfp4(q: []const u8, rows: usize, in: usize, scales: []const u8, scale_2: f32, dst: []f32) !void {
    const packed_cols = in / 2;
    const blocks = in / 16;
    if (q.len != rows * packed_cols or scales.len != rows * blocks or dst.len != rows * in) return error.ShapeMismatch;
    for (0..rows) |r| {
        for (0..blocks) |blk| {
            const s = e4m3ToFloat(scales[r * blocks + blk]) * scale_2;
            for (0..8) |pair| {
                const byte = q[r * packed_cols + blk * 8 + pair];
                dst[r * in + blk * 16 + pair * 2] = e2m1[byte & 0x0F] * s;
                dst[r * in + blk * 16 + pair * 2 + 1] = e2m1[byte >> 4] * s;
            }
        }
    }
}

/// torch .to(bfloat16): round-to-nearest-even on the low 16 bits (the fixture goldens store bf16 as f32).
pub fn bf16(x: f32) f32 {
    const u: u32 = @bitCast(x);
    if ((u & 0x7FFF_FFFF) > 0x7F80_0000) return x;
    const lsb: u32 = (u >> 16) & 1;
    return @bitCast((u + 0x7FFF + lsb) & 0xFFFF_0000);
}

test "e2m1 and e4m3 decode the documented values, subnormals and NaN" {
    try std.testing.expectEqual(@as(f32, 6.0), e2m1[7]);
    try std.testing.expectEqual(@as(f32, -6.0), e2m1[15]);
    try std.testing.expectEqual(@as(f32, 0.0), e2m1[8]); // -0.0 keeps the sign bit only
    try std.testing.expectEqual(@as(f32, 1.0), e4m3ToFloat(0x38)); // exp 7, mant 0
    try std.testing.expectEqual(@as(f32, 448.0), e4m3ToFloat(0x7E)); // the largest finite
    try std.testing.expectEqual(@as(f32, 0.001953125), e4m3ToFloat(0x01)); // smallest subnormal: 2^-9
    try std.testing.expectEqual(@as(f32, 0.015625), e4m3ToFloat(0x08)); // exp 0 mant 0: 2^-6
    try std.testing.expect(std.math.isNan(e4m3ToFloat(0x7F)));
    try std.testing.expectEqual(@as(f32, -448.0), e4m3ToFloat(0xFE));
}

test "dequantNvfp4 rejects mismatched shapes and decodes a known block" {
    var q: [8]u8 = @splat(0);
    var scales: [1]u8 = @splat(0);
    var dst: [16]f32 = undefined;
    // one block of 16: nibbles 0..15 across 8 bytes; scale 0x38 (1.0) x scale_2 2.0
    for (&q, 0..) |*b, i| b.* = @intCast(i * 2 | (i * 2 + 1) << 4);
    scales[0] = 0x38;
    try std.testing.expectError(error.ShapeMismatch, dequantNvfp4(q[0..7], 1, 16, scales[0..], 2.0, dst[0..]));
    try dequantNvfp4(q[0..], 1, 16, scales[0..], 2.0, dst[0..]);
    try std.testing.expectEqual(@as(f32, 0.0), dst[0]);
    try std.testing.expectEqual(@as(f32, 1.0), dst[1]); // nibble 1 x 2.0
    try std.testing.expectEqual(@as(f32, -12.0), dst[15]); // nibble 15 (high of byte 7) = -6, x 2.0
    // the odd nibble is the high one: byte 0 = nibbles {0, 1}
    try std.testing.expectEqual(@as(f32, 0.0), dst[0]);
    try std.testing.expectEqual(@as(f32, 1.0), dst[1]);
    try std.testing.expectEqual(@as(f32, 2.0), dst[2]); // byte 1 low = nibble 2
    try std.testing.expectEqual(@as(f32, 3.0), dst[3]);
}

test "bf16 rounds to nearest even and passes NaN and infinity through" {
    try std.testing.expectEqual(@as(f32, 1.0), bf16(1.0));
    try std.testing.expectEqual(@as(f32, 0.0), bf16(0.0));
    try std.testing.expect(std.math.isNan(bf16(std.math.nan(f32))));
    try std.testing.expect(std.math.isInf(bf16(std.math.inf(f32))));
    // a value with the low half exactly at the rounding midpoint rounds to even
    try std.testing.expectEqual(@as(u32, 0x3F80_0000), @as(u32, @bitCast(bf16(@as(f32, @bitCast(@as(u32, 0x3F80_7FFF)))))));
    try std.testing.expectEqual(@as(u32, 0x3F81_0000), @as(u32, @bitCast(bf16(@as(f32, @bitCast(@as(u32, 0x3F80_8001)))))));
}

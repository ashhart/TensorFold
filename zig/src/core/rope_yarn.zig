//! transformers' static YaRN rope for 64 or 128 rotary dims: the frequencies, and the factor cos and sin carry.
const std = @import("std");

/// The table for `dims` rotary dims: dims / 2 inverse frequencies as fp32, and the factor the cosines and sines carry.
pub fn Table(comptime dims: usize) type {
    return struct { freq: [dims / 2]f32, scale: f32 };
}

/// _compute_yarn_parameters (beta 32, 1; truncated ends) in fp32 over `dims` dims; flashnext keeps its own hashed copy.
pub fn yarnOf(comptime dims: usize, base: f64, factor: f64, trained: usize) Table(dims) {
    const dim: f64 = @floatFromInt(dims);
    const tp: f64 = @floatFromInt(trained);
    const dimOf = struct {
        fn f(rot: f64, b: f64, max_pos: f64, d: f64) f64 {
            return d * @log(max_pos / (rot * 2 * std.math.pi)) / (2 * @log(b));
        }
    }.f;
    const low = @max(@floor(dimOf(32, base, tp, dim)), 0);
    var high = @min(@ceil(dimOf(1, base, tp, dim)), dim - 1);
    if (low == high) high += 0.001;
    var out: Table(dims) = .{ .freq = undefined, .scale = @floatCast(0.1 * @log(factor) + 1) };
    for (&out.freq, 0..) |*f, k| {
        const kf: f64 = @floatFromInt(k);
        const pos_freq: f32 = @floatCast(std.math.pow(f64, base, @as(f64, @as(f32, @floatCast(2 * kf / dim)))));
        const extra: f32 = 1.0 / pos_freq;
        const inter: f32 = 1.0 / (@as(f32, @floatCast(factor)) * pos_freq);
        const ef: f32 = 1 - @as(f32, @floatCast(std.math.clamp((kf - low) / (high - low), 0, 1)));
        f.* = inter * (1 - ef) + extra * ef;
    }
    return out;
}

/// The 64-dim table (Qwen3.5's heads): the frequencies and the attention factor.
pub fn yarn(base: f64, factor: f64, trained: usize) Table(64) {
    return yarnOf(64, base, factor, trained);
}

test "Qwen3.6's YaRN table (theta 1e7, factor 4 over 262,144) equals transformers 5.19.0's, word for word" {
    // transformers 5.19.0's "yarn" rope init under torch 2.14.1 (the sglang-node:v0.5.22 image), fp32 words
    const torch = [32]u32{
        0x3f800000, 0x3f1ab32b, 0x3ebaf81b, 0x3e61f835, 0x3e088d77, 0x3da50956, 0x3d47763f, 0x3cf11177,
        0x3c91ad39, 0x3c301052, 0x3bd4ca15, 0x3b80967d, 0x3b1b690d, 0x3abbd3ed, 0x3a6301e2, 0x39f8a364,
        0x3986b53e, 0x3910058b, 0x38975c0d, 0x381b7e06, 0x379ac368, 0x3712f6ed, 0x36812dab, 0x361c1fc4,
        0x35bcb0c1, 0x35640cc6, 0x3509cf4a, 0x34a68e4c, 0x34494c57, 0x33f3499d, 0x3393048e, 0x3331af44,
    };
    const y = yarn(10000000, 4, 262144);
    for (y.freq, torch) |got, want| try std.testing.expectEqual(want, @as(u32, @bitCast(got)));
    try std.testing.expectEqual(@as(f32, @floatCast(1.138629436111989)), y.scale);
}

test "Qwen2.5's 128-dim YaRN table (theta 1e4, factor 4 over 8,192) equals transformers 5.22.2's, word for word" {
    // transformers 5.22.2's _compute_yarn_parameters under torch 2.13.0 (Qwen2Config, rope_parameters yarn), fp32 words
    const torch = [64]u32{
        0x3f800000, 0x3f5dafd7, 0x3f3ff911, 0x3f263de0, 0x3f0ff59a, 0x3ef953cf, 0x3ed7e89b, 0x3ebaf81b,
        0x3ea1e89b, 0x3e8c3504, 0x3e72d423, 0x3e5247ed, 0x3e361887, 0x3e1db040, 0x3e088d77, 0x3dec7fd6,
        0x3dcccccd, 0x3db15978, 0x3d99940d, 0x3d84fe4d, 0x3d6655c2, 0x3d47763f, 0x3d2cba15, 0x3d159348,
        0x3d0186e3, 0x3ce054d2, 0x3cbc6f5e, 0x3c9e219d, 0x3c8490d5, 0x3c5e067f, 0x3c39b628, 0x3c1b24c1,
        0x3c016f01, 0x3bd7a830, 0x3bb36127, 0x3b94f3dc, 0x3b76eb4c, 0x3b4c3fbe, 0x3b2894d9, 0x3b0ace50,
        0x3ae3f7a3, 0x3abaa4c8, 0x3a984d59, 0x3a77a07a, 0x3a48737a, 0x3a217916, 0x3a0157b6, 0x39cdd95b,
        0x39a2877f, 0x397e3f8e, 0x3944948c, 0x392a3b44, 0x39136a16, 0x38ff4fac, 0x38dd1725, 0x38bf74d7,
        0x38a5cb60, 0x388f9272, 0x3878a815, 0x385753e4, 0x383a7753, 0x38217916, 0x380bd472, 0x37f22ce2,
    };
    const y = yarnOf(128, 10000, 4, 8192);
    for (y.freq, torch) |got, want| try std.testing.expectEqual(want, @as(u32, @bitCast(got)));
    try std.testing.expectEqual(@as(u32, 0x3f91be9c), @as(u32, @bitCast(y.scale)));
}

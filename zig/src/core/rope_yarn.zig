//! transformers' static YaRN rope for 64 rotary dims: the frequencies and the attention factor cos and sin carry.
const std = @import("std");

/// _compute_yarn_parameters (beta 32, 1; truncated ends) in fp32; flashnext keeps its copy, which its receipts hash.
pub fn yarn(base: f64, factor: f64, trained: usize) struct { freq: [32]f32, scale: f32 } {
    const dim: f64 = 64;
    const tp: f64 = @floatFromInt(trained);
    const dimOf = struct {
        fn f(rot: f64, b: f64, max_pos: f64) f64 {
            return 64 * @log(max_pos / (rot * 2 * std.math.pi)) / (2 * @log(b));
        }
    }.f;
    const low = @max(@floor(dimOf(32, base, tp)), 0);
    var high = @min(@ceil(dimOf(1, base, tp)), dim - 1);
    if (low == high) high += 0.001;
    var out: [32]f32 = undefined;
    for (0..32) |k| {
        const kf: f64 = @floatFromInt(k);
        const pos_freq: f32 = @floatCast(std.math.pow(f64, base, @as(f64, @as(f32, @floatCast(2 * kf / dim)))));
        const extra: f32 = 1.0 / pos_freq;
        const inter: f32 = 1.0 / (@as(f32, @floatCast(factor)) * pos_freq);
        const ef: f32 = 1 - @as(f32, @floatCast(std.math.clamp((kf - low) / (high - low), 0, 1)));
        out[k] = inter * (1 - ef) + extra * ef;
    }
    return .{ .freq = out, .scale = @floatCast(0.1 * @log(factor) + 1) };
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

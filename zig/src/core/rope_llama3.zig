//! transformers' Llama 3 rope (rope_type llama3): the default fp32 table, long wavelengths stretched by the factor.
const std = @import("std");
const rope_freq = @import("rope_freq.zig");

/// rope_scaling's llama3 fields: the stretch, the factors that bound its band of wavelengths, the trained window.
pub const Llama3 = struct { factor: f64, low: f64, high: f64, original: usize };

/// out[i]: the default fp32 table's over `factor` past original / low wavelengths, kept under original / high, blended.
pub fn inverseFrequencies(theta: f64, dim: usize, s: Llama3, out: []f32) void {
    rope_freq.inverseFrequencies(theta, dim, out);
    const original: f64 = @floatFromInt(s.original);
    const factor: f32 = @floatCast(s.factor);
    const low_wavelength: f32 = @floatCast(original / s.low);
    const high_wavelength: f32 = @floatCast(original / s.high);
    const span: f32 = @floatCast(s.high - s.low);
    const two_pi: f32 = @floatCast(2 * std.math.pi);
    for (out) |*f| {
        const wavelength = two_pi / f.*;
        const kept = if (wavelength > low_wavelength) f.* / factor else f.*;
        const medium = !(wavelength < high_wavelength) and !(wavelength > low_wavelength);
        if (!medium) {
            f.* = kept;
            continue;
        }
        const smooth = (@as(f32, @floatCast(original)) / wavelength - @as(f32, @floatCast(s.low))) / span;
        f.* = (1 - smooth) * kept / factor + smooth * kept;
    }
}

/// transformers 5.3.0's LlamaRotaryEmbedding (torch 2.15.0, CPU, arm64) on Llama 3.1's published config: fp32 words.
pub const llama31_words = [64]u32{
    0x3f800000, 0x3f508ac1, 0x3f29e1c6, 0x3f0a6384, 0x3ee177bc, 0x3eb7ab7d, 0x3e959ee3, 0x3e73c461,
    0x3e4693b0, 0x3e21c3a0, 0x3e03c6a0, 0x3dd6b19c, 0x3daee4ad, 0x3d8e7898, 0x3d681e67, 0x3d3d1684,
    0x3d1a08c8, 0x3cfaf53f, 0x3ccc6f49, 0x3ca68939, 0x3c87a9c3, 0x3c5d06ec, 0x3c340d6d, 0x3c12ac7f,
    0x3beef74f, 0x3bc2aa76, 0x3b9e9402, 0x3b812e35, 0x3b527720, 0x3b0dfd06, 0x3ab3d11d, 0x3a60979e,
    0x3a0995d2, 0x39a3f108, 0x393b2dd2, 0x38c86886, 0x38a3418d, 0x3884fdbf, 0x3858ac81, 0x38308199,
    0x380fc8f8, 0x37ea426f, 0x37bed4f4, 0x379b7475, 0x377d45c3, 0x374e51f5, 0x3728126b, 0x3708ea0f,
    0x36df10c4, 0x36b5b687, 0x369406cb, 0x36712b80, 0x36447610, 0x36200a69, 0x36025f34, 0x35d46808,
    0x35ad07a7, 0x358cf400, 0x3565a54d, 0x353b12c7, 0x351864a7, 0x34f848c2, 0x34ca41b0, 0x34a4c2ff,
};

/// The same on Llama 3.2's (factor 32): the high band equal to 3.1's, the rest stretched four times as far.
pub const llama32_words = [64]u32{
    0x3f800000, 0x3f508ac1, 0x3f29e1c6, 0x3f0a6384, 0x3ee177bc, 0x3eb7ab7d, 0x3e959ee3, 0x3e73c461,
    0x3e4693b0, 0x3e21c3a0, 0x3e03c6a0, 0x3dd6b19c, 0x3daee4ad, 0x3d8e7898, 0x3d681e67, 0x3d3d1684,
    0x3d1a08c8, 0x3cfaf53f, 0x3ccc6f49, 0x3ca68939, 0x3c87a9c3, 0x3c5d06ec, 0x3c340d6d, 0x3c12ac7f,
    0x3beef74f, 0x3bc2aa76, 0x3b9e9402, 0x3b812e35, 0x3b527720, 0x3b0ad4f8, 0x3aa9279b, 0x3a47e547,
    0x39e13620, 0x39699599, 0x38cb98f7, 0x37c86886, 0x37a3418d, 0x3784fdbf, 0x3758ac81, 0x37308199,
    0x370fc8f8, 0x36ea426f, 0x36bed4f4, 0x369b7475, 0x367d45c3, 0x364e51f5, 0x3628126b, 0x3608ea0f,
    0x35df10c4, 0x35b5b687, 0x359406cb, 0x35712b80, 0x35447610, 0x35200a69, 0x35025f34, 0x34d46808,
    0x34ad07a7, 0x348cf400, 0x3465a54d, 0x343b12c7, 0x341864a7, 0x33f848c2, 0x33ca41b0, 0x33a4c2ff,
};

const llama31: Llama3 = .{ .factor = 8, .low = 1, .high = 4, .original = 8192 };
const llama32: Llama3 = .{ .factor = 32, .low = 1, .high = 4, .original = 8192 };

fn expectWords(want: []const u32, got: []const f32) !void {
    for (want, got) |w, g| try std.testing.expectEqual(w, @as(u32, @bitCast(g)));
}

test "Llama 3.1's and 3.2's tables (theta 500,000 over 128 dims) equal transformers' words" {
    var got: [64]f32 = undefined;
    inverseFrequencies(500000, 128, llama31, &got);
    try expectWords(&llama31_words, &got);
    inverseFrequencies(500000, 128, llama32, &got);
    try expectWords(&llama32_words, &got);
}

test "the bands: short wavelengths keep the default table, long ones divide by the factor, six blend between" {
    var plain: [64]f32 = undefined;
    rope_freq.inverseFrequencies(500000, 128, &plain);
    var got: [64]f32 = undefined;
    inverseFrequencies(500000, 128, llama31, &got);
    var blended: usize = 0;
    for (plain, got) |p, g| {
        if (g == p) continue;
        if (g == p / 8) continue;
        try std.testing.expect(g < p and g > p / 8);
        blended += 1;
    }
    try std.testing.expectEqual(@as(usize, 6), blended);
    try std.testing.expectEqual(plain[28], got[28]);
    try std.testing.expectEqual(plain[35] / 8, got[35]);
}

test "planted: a formula slip misses transformers' words (the blend's band unblended, the factor's ends swapped)" {
    var got: [64]f32 = undefined;
    inverseFrequencies(500000, 128, .{ .factor = 8, .low = 1, .high = 1.0000001, .original = 8192 }, &got);
    var differ: usize = 0;
    for (llama31_words, got) |w, g| differ += @intFromBool(w != @as(u32, @bitCast(g)));
    try std.testing.expect(differ >= 5);
    inverseFrequencies(500000, 128, .{ .factor = 8, .low = 4, .high = 1, .original = 8192 }, &got);
    differ = 0;
    for (llama31_words, got) |w, g| differ += @intFromBool(w != @as(u32, @bitCast(g)));
    try std.testing.expect(differ >= 5);
}

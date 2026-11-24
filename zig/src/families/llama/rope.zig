//! A YaRN checkpoint's rotation table: each position's bf16 cos and sin, made once on the host for ll_prep_table.
const std = @import("std");
const bf16_round = @import("bf16_round");
const cfg = @import("config.zig");

const hd = cfg.head_dim;

/// The table's bytes over the window, 0 for a checkpoint without YaRN (its rotation is ll_prep's, on the GPU).
pub fn bytes(c: cfg.Config) usize {
    return if (c.yarn != null) c.context * hd * 2 else 0;
}

/// Positions first.. into `out`, a row each: bf16(fp32(cos or sin of the fp32 angle) * the factor), cosines first.
pub fn table(c: cfg.Config, first: usize, out: []u16) void {
    const half = hd / 2;
    var inv: [half]f32 = undefined;
    c.inverseFrequencies(&inv);
    for (0..out.len / hd) |r| {
        const position: f32 = @floatFromInt(first + r);
        const row = out[r * hd ..][0..hd];
        for (inv, 0..) |f, i| {
            const angle: f32 = position * f;
            const cs: f32 = @floatCast(@cos(@as(f64, angle)));
            const sn: f32 = @floatCast(@sin(@as(f64, angle)));
            row[i] = bf16_round.bits(cs * c.ropeScale());
            row[half + i] = bf16_round.bits(sn * c.ropeScale());
        }
    }
}

test "Bimo-7B-CyberAgent's rotation table equals transformers 5.22.2's over the whole window but for 19 bf16 ties" {
    const c = try cfg.parse(std.testing.allocator, @embedFile("fixtures/bimo-config.json"));
    const t = try std.testing.allocator.alloc(u16, c.context * hd);
    defer std.testing.allocator.free(t);
    table(c, 0, t);
    // Qwen2RotaryEmbedding's word (torch 2.15.0, CPU) where torch's fp32 cos or sin is an ulp off, on a bf16 tie
    const hf_words = [_][3]u32{
        .{ 777, 14, 0xbf91 },
        .{ 887, 37, 0xbf88 },
        .{ 2007, 63, 0x3f91 },
        .{ 3180, 51, 0x3f7d },
        .{ 5176, 41, 0x3f07 },
        .{ 6934, 85, 0xbf91 },
        .{ 7406, 27, 0x3c9a },
        .{ 7717, 97, 0x3f11 },
        .{ 9475, 78, 0x3f22 },
        .{ 10741, 62, 0x3f89 },
        .{ 11223, 27, 0xbf91 },
        .{ 13113, 5, 0xbeb6 },
        .{ 13174, 85, 0x3f30 },
        .{ 16883, 33, 0xbeeb },
        .{ 17971, 93, 0xbf91 },
        .{ 21966, 14, 0x3eb9 },
        .{ 22252, 80, 0x3f6e },
        .{ 23242, 68, 0x3f67 },
        .{ 25610, 35, 0xbf8f },
    };
    for (hf_words) |w| {
        const ours = t[w[0] * hd + w[1]];
        try std.testing.expectEqual(@as(u32, 1), @abs(@as(i32, ours) - @as(i32, @intCast(w[2]))));
        t[w[0] * hd + w[1]] = @intCast(w[2]);
    }
    // with them, every one of the 32,768 positions' cos[:64] | sin[:64] bf16 words is HF's
    var sum: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(t), &sum, .{});
    try std.testing.expectEqualStrings("d628b48fe05715468273774b8a287e90a4b2ef4554382601965b0624b1fc0b21", &std.fmt.bytesToHex(sum, .lower));
    // a window from a later position is the same rows
    var tail: [2 * hd]u16 = undefined;
    table(c, c.context - 2, &tail);
    try std.testing.expectEqualSlices(u16, t[(c.context - 2) * hd ..], &tail);
}

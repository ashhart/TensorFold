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

test "Bimo-7B-CyberAgent's rotation table is transformers 5.22.2's at its rounding points, cos and sin exactly rounded" {
    const c = try cfg.parse(std.testing.allocator, @embedFile("fixtures/bimo-config.json"));
    const t = try std.testing.allocator.alloc(u16, c.context * hd);
    defer std.testing.allocator.free(t);
    table(c, 0, t);
    // the raw table's hash, as tools/zig/llama_rope_table.py prints it from transformers' own frequencies and factor
    var sum: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(t), &sum, .{});
    try std.testing.expectEqualStrings("44da14ee331db23c2376ec16f7680cfd7fd74e58d0caf267b27d82195ca96a92", &std.fmt.bytesToHex(sum, .lower));
    // a window from a later position is the same rows
    var tail: [2 * hd]u16 = undefined;
    table(c, c.context - 2, &tail);
    try std.testing.expectEqualSlices(u16, t[(c.context - 2) * hd ..], &tail);
}

//! DeepSeek-V4.1's Metal image preparation: a thin image's short side, and degenerate vision config refusals.
const std = @import("std");
const vision = @import("dsv41_metal").vision;
const prep = @import("dsv41_metal").vision_prep;
const vprep = @import("dsv41_metal").vision_prepare;

test "prepare: a one-pixel image keeps a one-pixel short side, never a zero side" {
    const a = std.testing.allocator;
    const cfg: vision.Config = .{};
    // 1x84 and 8192x1 at budget 5, 1x2604 at 64: each used to fit a zero side, which resample.into divided by.
    const cases = [_]struct { u32, u32, u32 }{ .{ 1, 84, 5 }, .{ 8192, 1, 5 }, .{ 1, 2604, 64 } };
    for (cases) |c| {
        const png = try vprep.solidPng(a, c[0], c[1], .{ 90, 120, 150 });
        defer a.free(png);
        const p = try vprep.prepare(std.testing.io, a, png, c[2], cfg);
        defer a.free(p.pixels);
        try std.testing.expect(p.gh >= 1 and p.gw >= 1);
        try std.testing.expect(p.tokens >= 1 and p.tokens <= c[2]);
    }
    // contain itself keeps both sides at one pixel or more at every aspect a one-pixel image can take.
    for ([_]u32{ 1, 42, 126, 8192 }) |bh| {
        for ([_]u32{ 1, 42, 126, 8192 }) |bw| {
            const f = prep.contain(1, @max(84, 2 * bh), bw, bh);
            try std.testing.expect(f[0] >= 1 and f[1] >= 1);
            const g = prep.contain(@max(84, 2 * bw), 1, bw, bh);
            try std.testing.expect(g[0] >= 1 and g[1] >= 1);
        }
    }
}

test "check: a degenerate vision config is refused by a named error" {
    const a = std.testing.allocator;
    try std.testing.expectError(error.ZeroHeads, vision.Config.parse(a, "{\"vision_config\": {\"num_hidden_layers\": 32, \"hidden_size\": 1024, \"num_attention_heads\": 0, \"intermediate_size\": 2816, \"patch_size\": 14, \"rope_theta\": 10000, \"downsample_ratio\": 3, \"max_image_tokens\": 1024, \"min_pixels\": 295936, \"max_wh_ratio\": null}}", 5120));
    try std.testing.expectError(error.LowDownsampleRatio, vision.Config.parse(a, "{\"vision_config\": {\"num_hidden_layers\": 32, \"hidden_size\": 1024, \"num_attention_heads\": 16, \"intermediate_size\": 2816, \"patch_size\": 14, \"rope_theta\": 10000, \"downsample_ratio\": 1, \"max_image_tokens\": 1024, \"min_pixels\": 295936, \"max_wh_ratio\": null}}", 5120));
    try std.testing.expectError(error.ZeroMaxWhRatio, vision.Config.parse(a, "{\"vision_config\": {\"num_hidden_layers\": 32, \"hidden_size\": 1024, \"num_attention_heads\": 16, \"intermediate_size\": 2816, \"patch_size\": 14, \"rope_theta\": 10000, \"downsample_ratio\": 3, \"max_image_tokens\": 1024, \"min_pixels\": 295936, \"max_wh_ratio\": 0}}", 5120));
    for ([_]u32{ 1, 2, 3, 4 }) |b| {
        const png = try vprep.solidPng(a, 40, 30, .{ 1, 2, 3 });
        defer a.free(png);
        try std.testing.expectError(error.TooFewTokens, vprep.prepare(std.testing.io, a, png, b, .{}));
    }
}

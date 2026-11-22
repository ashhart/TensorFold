//! gfx1201's prompt choices in the mlx tiles' registry: the dot2 tiles with the matrix cores off, its matrix tiles on.

const std = @import("std");
const registry = @import("core").registry;
const entries = @import("mlx_entries.zig");
const rows12 = @import("matrix_rows.zig");

const Shape = registry.Shape;
const kernelsOn = entries.kernelsOn;
const envOn = entries.envOn;

test "gfx1201 takes the K-parallel tile for short routed prompts, up to its items and pairs per item" {
    const r = kernelsOn("gfx1201");
    const env = envOn("gfx1201", true, true, false);
    // each model's served plans on both sides of a bound: n, k, group, items, pairs and the tile
    const want = [_]struct { u32, u32, u16, u32, u32, []const u8 }{
        .{ 1024, 2048, 64, 401, 1152, "kp3" }, // Qwen3.6, 128 tokens
        .{ 1024, 2048, 64, 545, 1161, "gemm" }, // 129
        .{ 1856, 2688, 64, 224, 768, "kp3" }, // Nemotron, 128 tokens, as before
        .{ 1856, 2688, 64, 320, 1116, "kp3" }, // 186
        .{ 1856, 2688, 64, 320, 1122, "gemm" }, // 187
        .{ 1280, 2560, 32, 507, 451, "kp3" }, // Flash Next, 41 rows
        .{ 1280, 2560, 32, 519, 462, "gemm" }, // 42
    };
    for (want) |w| {
        const shape: Shape = .{ .m = 128, .n = w[0], .k = w[1], .bits = 4, .group = w[2], .fp16 = false, .items = w[3], .plan_pairs = w[4] };
        try std.testing.expect(std.mem.endsWith(u8, r.select(env, .mlx, .routed, .prefill, shape).?.id, w[5]));
    }
}

test "gfx1201 takes its matrix tiles for a prompt at every row count, the dot2 tiles where they do not fit" {
    const r = kernelsOn("gfx1201");
    const on = envOn("gfx1201", true, true, true);
    var shape: Shape = .{ .m = 1, .n = 4096, .k = 4096, .bits = 4, .group = 64, .fp16 = false };
    for ([_]struct { u32, []const u8 }{ .{ 1, "rows16" }, .{ 32, "rows16" }, .{ 33, "rows64" }, .{ 64, "rows64" }, .{ 65, "matrix12" }, .{ 2048, "matrix12" } }) |w| {
        shape.m = w[0];
        try std.testing.expect(std.mem.endsWith(u8, r.select(on, .mlx, .project, .prefill, shape).?.id, w[1]));
    }
    // Flash Next's gate and up as launched (11 slots a row, 513 experts, m = @min(tile, rows)): 2.0-6.8 pairs an item
    for ([_]struct { u32, []const u8 }{ .{ 1, "rows16" }, .{ 128, "rows16" }, .{ 1024, "rows32" }, .{ 2048, "rows64" } }) |w| {
        const pairs = w[0] * 11;
        const routed: Shape = .{ .m = @min(w[0], 128), .n = 1280, .k = 2560, .bits = 2, .group = 32, .fp16 = false, .items = @min(pairs, 513) + pairs / 8, .plan_pairs = pairs };
        try std.testing.expect(std.mem.endsWith(u8, r.select(on, .mlx, .routed, .prefill, routed).?.id, w[1]));
    }
    // a router that fills its items to 20 rows each: the 64-row block
    const full: Shape = .{ .m = 128, .n = 1280, .k = 2560, .bits = 2, .group = 32, .fp16 = false, .items = 512, .plan_pairs = 10240 };
    try std.testing.expect(std.mem.endsWith(u8, r.select(on, .mlx, .routed, .prefill, full).?.id, "rows64"));
    // words the matrix tiles cannot load wide, or the matrix cores switched off: the dot2 tiles by rows
    shape.m = 2;
    shape.words_aligned = false;
    try std.testing.expect(std.mem.endsWith(u8, r.select(on, .mlx, .project, .prefill, shape).?.id, "block"));
    shape.words_aligned = true;
    try std.testing.expect(std.mem.endsWith(u8, r.select(envOn("gfx1201", true, true, false), .mlx, .project, .prefill, shape).?.id, "kp0"));
    // a decode call keeps the dot2 tiles: gfx12's matrix tile has no decode entry
    shape.m = 32;
    try std.testing.expect(std.mem.endsWith(u8, r.select(on, .mlx, .project, .decode, shape).?.id, ".decode.gemm"));
}

test "a plain product takes a shorter tier only while its 128-row blocks leave compute units idle" {
    if (!std.mem.eql(u8, @import("../kernels.zig").arch, "gfx1201")) return error.SkipZigTest; // measured there only
    // 32 columns of 128: 256 rows fill 64 blocks, 128 rows fill them on 64-row blocks, 64 rows need 32-row blocks
    try std.testing.expectEqual(@as(?usize, null), rows12.fillTier(256, 4096));
    try std.testing.expectEqual(@as(?usize, 0), rows12.fillTier(128, 4096));
    try std.testing.expectEqual(@as(?usize, 1), rows12.fillTier(64, 4096));
    // 10 columns: 2,048 rows keep the 128-row block, 512 take 64 rows, 256 take 32
    try std.testing.expectEqual(@as(?usize, null), rows12.fillTier(2048, 1280));
    try std.testing.expectEqual(@as(?usize, 0), rows12.fillTier(512, 1280));
    try std.testing.expectEqual(@as(?usize, 1), rows12.fillTier(256, 1280));
}

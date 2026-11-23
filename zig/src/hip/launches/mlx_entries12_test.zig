//! gfx1201's prompt choices in the mlx registry: the dot2 tiles by default, its matrix tiles with TF_HIP_GFX12_WMMA=1.

const std = @import("std");
const registry = @import("core").registry;
const entries = @import("mlx_entries.zig");
const rows12 = @import("matrix_rows.zig");

const Shape = registry.Shape;
const kernelsOn = entries.kernelsOn;
const envOn = entries.envOn;

/// gfx1201 with TF_HIP_GFX12_WMMA=1 (Kernels.env): its matrix tiles, a family of their own.
fn envOn12(matrix_on: bool) registry.Env {
    var e = envOn("gfx1201", true, true, matrix_on);
    e.matrix = true;
    e.matrix_item = false;
    e.matrix_on = matrix_on;
    return e;
}

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
    const on = envOn12(true);
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
    // Nemotron's bucketed plan (128 experts, 6 a row): its capacity doubles past 1,024 rows, so the expert rows bound
    for ([_]struct { u32, []const u8 }{ .{ 341, "rows16" }, .{ 342, "rows32" }, .{ 513, "rows32" }, .{ 1025, "rows64" } }) |w| {
        const size = std.math.ceilPowerOfTwo(u32, w[0]) catch unreachable;
        var bucketed: Shape = .{ .m = 128, .n = 1856, .k = 2688, .bits = 4, .group = 64, .fp16 = false };
        bucketed.items = 128 + size * 6 / 8;
        bucketed.plan_pairs = w[0] * 6;
        bucketed.expert_rows = (w[0] * 6 + 127) / 128;
        try std.testing.expect(std.mem.endsWith(u8, r.select(on, .mlx, .routed, .prefill, bucketed).?.id, w[1]));
    }
    // words the matrix tiles cannot load wide, or the matrix cores switched off: the dot2 tiles by rows
    shape.m = 2;
    shape.words_aligned = false;
    try std.testing.expect(std.mem.endsWith(u8, r.select(on, .mlx, .project, .prefill, shape).?.id, "block"));
    shape.words_aligned = true;
    try std.testing.expect(std.mem.endsWith(u8, r.select(envOn12(false), .mlx, .project, .prefill, shape).?.id, "kp0"));
    // a decode call keeps the dot2 tiles: gfx12's matrix tile has no decode entry
    shape.m = 32;
    try std.testing.expect(std.mem.endsWith(u8, r.select(on, .mlx, .project, .decode, shape).?.id, ".decode.gemm"));
}

test "a plain product takes a shorter tier only while its 128-row blocks leave compute units idle" {
    if (!std.mem.eql(u8, @import("../kernels.zig").arch, "gfx1201")) return error.SkipZigTest; // measured there only
    // 32 columns of 128: 256 rows fill 64 blocks, 128 rows fill them on 64-row blocks, 64 rows need 32-row blocks
    try std.testing.expectEqual(@as(?usize, null), rows12.fillTier(256, 4096));
    try std.testing.expectEqual(@as(?usize, 0), rows12.fillTier(128, 4096));
    // fillTier alone: no launch reaches it, the registry gives a plain product of 64 rows rows64
    try std.testing.expectEqual(@as(?usize, 1), rows12.fillTier(64, 4096));
    // 10 columns: 2,048 rows keep the 128-row block, 512 take 64 rows, 256 take 32
    try std.testing.expectEqual(@as(?usize, null), rows12.fillTier(2048, 1280));
    try std.testing.expectEqual(@as(?usize, 0), rows12.fillTier(512, 1280));
    try std.testing.expectEqual(@as(?usize, 1), rows12.fillTier(256, 1280));
}

test "gfx1201 keeps a routed prompt in the matrix12 family over its items, pairs and expert rows" {
    const r = kernelsOn("gfx1201");
    const on = envOn12(true);
    // Qwen3.6, Nemotron and Flash Next's routed products; past 64 expert rows or 1,024 rows the 128-row tile
    const models = [_]struct { u32, u32, u16, u8 }{
        .{ 1024, 2048, 64, 4 },
        .{ 1856, 2688, 64, 4 },
        .{ 1280, 2560, 32, 2 },
    };
    for (models) |w| {
        const base: Shape = .{ .m = 1, .n = w[0], .k = w[1], .bits = w[3], .group = w[2], .fp16 = false };
        var expert_rows: u32 = 1;
        while (expert_rows <= 300) : (expert_rows += 7) {
            for ([_]u32{ 1, 16, 128, 1024, 1025 }) |m| {
                for ([_]u32{ 64, 400, 513, 2000 }) |items| {
                    for ([_]u32{ 1, 3, 6, 7, 40 }) |per_item| {
                        var shape = base;
                        shape.m = m;
                        shape.items = items;
                        shape.plan_pairs = items * per_item;
                        shape.expert_rows = expert_rows;
                        const e = r.select(on, .mlx, .routed, .prefill, shape).?;
                        try std.testing.expectEqual(registry.Family.matrix12, e.family);
                    }
                }
            }
        }
    }
}

/// Every gfx12 pick over the served models' products, rows and plans, and the policy's switches, as one hash.
fn gfx12Picks(gfx: []const u8) [2]u64 {
    const r = kernelsOn(gfx);
    var h = std.hash.Wyhash.init(0);
    var count: u64 = 0;
    const models = [_]struct { u32, u32, u16, u8 }{ .{ 1024, 2048, 64, 4 }, .{ 1856, 2688, 64, 4 }, .{ 1280, 2560, 32, 2 }, .{ 4096, 4096, 64, 4 }, .{ 2048, 512, 32, 8 }, .{ 6144, 2048, 128, 3 } };
    const rows = [_]u32{ 1, 2, 3, 4, 8, 9, 15, 16, 17, 32, 33, 64, 65, 128, 129, 512, 1024, 2048 };
    for (0..8) |bits| {
        const env = envOn(gfx, bits & 1 != 0, bits & 2 != 0, bits & 4 != 0);
        for ([_]registry.Op{ .project, .group, .routed, .routed_act }) |op| for ([_]registry.Path{ .decode, .prefill }) |path| for (models) |md| for (rows) |m| for ([_]bool{ true, false }) |aligned| {
            const base: Shape = .{ .m = m, .n = md[0], .k = md[1], .bits = md[3], .group = md[2], .fp16 = false, .words_aligned = aligned };
            const plain = op == .project or op == .group;
            const item_counts: []const u32 = if (plain) &.{0} else &.{ 8, 64, 300, 401, 507, 513, 545, 2000 };
            for (item_counts) |items| for (if (plain) &[_]u32{0} else &[_]u32{ 1, 2, 3, 4, 6, 7, 12 }) |per_item| for (if (plain) &[_]u32{0} else &[_]u32{ 1, 5, 16, 17, 27, 28, 64, 300 }) |expert_rows| {
                var s = base;
                s.items = items;
                s.plan_pairs = items * per_item;
                s.expert_rows = expert_rows;
                const id = if (r.select(env, .mlx, op, path, s)) |e| e.id else "-";
                h.update(id);
                h.update("\n");
                count += 1;
            };
        };
    }
    return .{ h.final(), count };
}

test "without TF_HIP_GFX12_WMMA, gfx1200 and gfx1201 pick exactly main's tiles" {
    // main's picks (281d81b8b) over the same sweep, in the same order
    for ([_][]const u8{ "gfx1200", "gfx1201" }) |gfx| try std.testing.expectEqual([2]u64{ 0xbd8557c4188b8b63, 3103488 }, gfx12Picks(gfx));
}

test "with TF_HIP_GFX12_WMMA=1 a lane round and a prompt each keep one family on gfx1201" {
    const r = kernelsOn("gfx1201");
    for ([_]bool{ true, false }) |matrix_on| try r.verify(envOn12(matrix_on), .mlx);
}

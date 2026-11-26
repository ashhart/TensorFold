//! Unit tests split out of attention.zig plus M3 Ultra 2048-row prompt layout coverage.
const std = @import("std");
const attn = @import("attention.zig");
const mem_layout = @import("attention_memory.zig").Layout;
const prompt_rows = @import("prompt_rows.zig");
const planning = attn.planning;
const Plan = attn.Plan;

test "cache and pack descriptors match native pointer and six-word ABI" {
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(attn.CachePtrs));
    try std.testing.expectEqual(@as(usize, 32), @sizeOf(attn.PagedPtrs));
    try std.testing.expectEqual(@as(usize, 24), @sizeOf(attn.PackArgs));
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(planning.KeepRow));
}

test "plan fingerprint changes for an ancestor mutation rather than only counts" {
    var plan = try Plan.init(std.testing.allocator, &.{.{ .parents = &.{ -1, 0, 0 }, .start = 63, .capacity = 512 }}, .{});
    defer plan.deinit();
    const before = attn.fingerprint(plan);
    plan.paths[2 * planning.path_width + 1] = 1;
    try std.testing.expect(!std.mem.eql(u8, &before, &attn.fingerprint(plan)));
}

test "cache span includes padded head strides but no unused trailing padding" {
    const cache = attn.Cache{ .keys = undefined, .values = undefined, .capacity = 8, .key_stride = 4096, .value_stride = 8192 };
    try std.testing.expectEqual(@as(usize, 28672), attn.cacheBytes(cache, true, .{}));
    try std.testing.expectEqual(@as(usize, 53248), attn.cacheBytes(cache, false, .{}));
    const contiguous = attn.Cache{ .keys = undefined, .values = undefined, .capacity = 8 };
    try std.testing.expectEqual(@as(usize, 16384), attn.cacheBytes(contiguous, true, .{}));
}

test "descriptor identity includes numeric geometry and limits still refuse new shaders" {
    var plan = try Plan.init(std.testing.allocator, &.{.{ .parents = &.{-1}, .start = 0, .capacity = 8 }}, .{});
    defer plan.deinit();
    const before = attn.fingerprint(plan);
    plan.geometry.heads = 20; // GQA 5: no shader build takes it
    try std.testing.expect(!std.mem.eql(u8, &before, &attn.fingerprint(plan)));
    try std.testing.expectError(error.TargetAttentionShape, attn.validLimits(.{ .geometry = plan.geometry }));
}

test "M3 Ultra unlocks 2048-row linear chains without 4x pob over-allocation" {
    try std.testing.expectEqual(@as(u32, 2048), prompt_rows.requested(17, false, null));
    try std.testing.expectEqual(@as(u32, 512), prompt_rows.requested(15, false, null));
    try std.testing.expectEqual(@as(u32, 2048), prompt_rows.requestedModel(15, false, true, null));
    try std.testing.expectEqual(@as(u32, 512), prompt_rows.requestedModel(15, false, false, null));
    try std.testing.expectEqual(@as(u32, 512), prompt_rows.requestedModel(14, false, true, null));

    const g_m3 = planning.Geometry{ .heads = 16, .kv_heads = 4, .shared_prefix = false, .fuse_chain = true, .wide_prompt = true };
    try std.testing.expectEqual(@as(u32, 2048), g_m3.promptRows(2048));
    try std.testing.expectEqual(@as(u32, 128), (planning.Geometry{ .shared_prefix = false, .fuse_chain = true }).promptRows(2048));
    try std.testing.expectEqual(@as(u32, 128), (planning.Geometry{ .shared_prefix = false, .fuse_chain = false, .wide_prompt = true }).promptRows(2048));

    var chain: [@import("gpu_frame.zig").max_rows]i32 = undefined;
    for (&chain, 0..) |*p, r| p.* = @as(i32, @intCast(r)) - 1;
    var p = try Plan.init(std.testing.allocator, &.{.{ .parents = &chain, .start = 0, .capacity = 4096 }}, g_m3);
    defer p.deinit();
    try std.testing.expectEqual(@as(u32, 2048), p.rows);
    try std.testing.expect(!p.fused);

    var tree: [129]i32 = undefined;
    for (&tree, 0..) |*q, r| q.* = if (r == 0) -1 else 0;
    try std.testing.expectError(error.AttentionCapacity, Plan.init(std.testing.allocator, &.{.{ .parents = &tree, .start = 0, .capacity = 4096 }}, g_m3));

    const lim_m3 = attn.Limits{ .rows = 2048, .streams = 1, .keys = 32768, .shared_prefix = false, .geometry = g_m3 };
    const lay_m3 = mem_layout.init(lim_m3);
    const pob_idx = @backingInt(attn.Field.pob);
    const old_b = @as(usize, g_m3.kv_heads) * lay_m3.tail_chunks * lim_m3.rows * 16 * g_m3.dim * 4;
    try std.testing.expect(lay_m3.sizes[pob_idx] < old_b / 2);
    try std.testing.expect(lay_m3.sizes[pob_idx] >= @as(usize, g_m3.kv_heads) * lay_m3.chunks * p.tiles * 16 * g_m3.dim * 4);

    const lim_m5 = attn.Limits{ .rows = 2048, .streams = 1, .keys = 32768, .shared_prefix = true, .geometry = .{ .heads = 16, .kv_heads = 4, .wide_prompt = true } };
    const lay_m5 = mem_layout.init(lim_m5);
    const expected_m5_b = @as(usize, 4) * 3 * lim_m5.rows * 16 * 256 * 4;
    try std.testing.expectEqual(expected_m5_b, lay_m5.sizes[pob_idx]);
    const lim_default = attn.Limits{ .rows = 2048, .streams = 1, .keys = 32768, .shared_prefix = true, .geometry = .{ .heads = 16, .kv_heads = 4 } };
    try std.testing.expectEqual(@as(usize, 4) * 3 * 2048 * 16 * 256 * 4, mem_layout.init(lim_default).sizes[pob_idx]);
}

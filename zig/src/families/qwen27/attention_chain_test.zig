//! Long prompt chains use absolute keys while branching windows retain their bounded ancestor maps.
const std = @import("std");
const p = @import("attention_plan.zig");
const frame_rows = @import("gpu_frame.zig").max_rows;
const a = std.testing.allocator;
const pre_m5 = p.Geometry{ .shared_prefix = false };

fn parents() [frame_rows]i32 {
    var chain: [frame_rows]i32 = undefined;
    for (&chain, 0..) |*parent, r| parent.* = @as(i32, @intCast(r)) - 1;
    return chain;
}

test "attention chain: pre-M5 prompt choices follow every admitted frame and request" {
    const requested = @import("prompt_rows.zig");
    for ([_]u32{ 13, 14, 15, 16 }) |generation| {
        try std.testing.expectEqual(@as(u32, 512), pre_m5.promptRows(requested.requested(generation, false, null)));
        for ([_][]const u8{ "256", "512", "1024", "2048" }, [_]u32{ 256, 512, 1024, 2048 }) |text, want| {
            try std.testing.expectEqual(want, pre_m5.promptRows(requested.requested(generation, false, text)));
        }
    }
    var frame: u32 = 16;
    while (frame <= frame_rows) : (frame += 16) {
        for ([_]u32{ 1, 16, 127, 128, 129, 256, 512, 1024, 2048, 4096 }) |request| {
            try std.testing.expectEqual(@min(request, frame), pre_m5.promptRows(@min(request, frame)));
        }
    }
    try std.testing.expectEqual(@as(u32, frame_rows), pre_m5.promptRows(4096));
    try std.testing.expectEqual(@as(u32, 128), (p.Geometry{ .heads = 16, .kv_heads = 4, .shared_prefix = false }).promptRows(512));
    try std.testing.expectEqual(@as(u32, 128), (p.Geometry{ .heads = 32, .kv_heads = 2, .shared_prefix = false }).promptRows(512));
    try std.testing.expectEqual(@as(u32, 128), (p.Geometry{ .shared_prefix = false, .fuse_chain = false }).promptRows(512));
}

test "attention chain: long tail positions and causal physical keys equal one-row decode" {
    const chain = parents();
    for ([_]usize{ 127, 128, 129, 255, 256, 511, 512, 1023, 1024, 2048 }) |rows| {
        for ([_]u32{ 0, 63, 64, 127, 128, 511, 512, 8063, 8191, 8192, 8193 }) |start| {
            var plan = try p.Plan.init(a, &.{.{ .parents = chain[0..rows], .start = start, .capacity = 16384 }}, pre_m5);
            defer plan.deinit();
            try std.testing.expect(!plan.fused);
            try std.testing.expectEqual(@as(i32, @intFromBool(rows > p.path_width)), plan.meta[p.global_words + 10]);
            for (0..rows) |r| {
                const position = start + @as(u32, @intCast(r));
                try std.testing.expectEqual(@as(i32, @intCast(position)), plan.positions[r]);
                try std.testing.expectEqual(@as(i32, @intCast(p.chunkFor(position))), plan.nodes[r * p.node_words + 2]);
                for ([_]u32{ 0, start, position, position + 1 }) |key| {
                    const want: ?u32 = if (key <= position) key else null;
                    try std.testing.expectEqual(want, try plan.physical(r, key));
                }
            }
            try std.testing.expectEqual(@as(u32, 0), plan.prefix_chunks);
            try std.testing.expect(plan.tail_chunks <= 65);
        }
    }
}

test "attention chain: keep validates long chain identity without reading past the ancestor map" {
    const chain = parents();
    var plan = try p.Plan.init(a, &.{.{ .parents = &chain, .start = 127, .capacity = 4096 }}, pre_m5);
    defer plan.deinit();
    var kept: [frame_rows]u32 = undefined;
    for (&kept, 0..) |*row, r| row.* = @intCast(r);
    for ([_]usize{ 0, 127, 128, 129, 512, 1024, 2048 }) |n| {
        const map = try plan.keepRows(a, &.{kept[0..n]});
        defer a.free(map);
        for (map, 0..) |row, r| try std.testing.expectEqual(p.KeepRow{ .stream = 0, .source = @intCast(r), .destination = @intCast(r), .row = @intCast(r) }, row);
    }
    kept[129] = 0;
    try std.testing.expectError(error.AttentionKeep, plan.keepRows(a, &.{kept[0..512]}));
}

test "attention chain: branching, multiple streams and disabled chains keep their old limits" {
    var chain = parents();
    const window = p.Window{ .parents = chain[0..129], .start = 0, .capacity = 4096 };
    try std.testing.expectError(error.AttentionCapacity, p.Plan.init(a, &.{window}, .{ .shared_prefix = false, .fuse_chain = false }));
    try std.testing.expectError(error.AttentionCapacity, p.Plan.init(a, &.{ window, window }, pre_m5));
    const small = p.Window{ .parents = chain[0..65], .start = 0, .capacity = 4096 };
    try std.testing.expectError(error.AttentionRows, p.Plan.init(a, &.{ small, small }, pre_m5));
    chain[128] = 0;
    try std.testing.expectError(error.AttentionCapacity, p.Plan.init(a, &.{window}, pre_m5));
    var short = try p.Plan.init(a, &.{.{ .parents = &.{ -1, 0, 0 }, .start = 511, .capacity = 4096 }}, pre_m5);
    defer short.deinit();
    try std.testing.expectEqual(@as(i32, 0), short.meta[p.global_words + 10]);
    try std.testing.expectEqual(@as(?u32, null), try short.physical(2, 513));
    try std.testing.expectEqual(@as(?u32, 513), try short.physical(2, 512));
}

test "attention chain: every ragged width through the frame limit has the same final causal row" {
    const chain = parents();
    const limit = @import("attention_memory.zig").Layout.init(.{ .rows = @as(u32, frame_rows), .streams = @as(u32, 1), .keys = @as(u32, 16384), .shared_prefix = false, .geometry = pre_m5 });
    for (1..frame_rows + 1) |rows| {
        var plan = try p.Plan.init(a, &.{.{ .parents = chain[0..rows], .start = 8191, .capacity = 16384 }}, pre_m5);
        defer plan.deinit();
        const last: u32 = @intCast(8191 + rows - 1);
        try std.testing.expectEqual(@as(?u32, last), try plan.physical(rows - 1, last));
        try std.testing.expectEqual(@as(?u32, 8191), try plan.physical(rows - 1, 8191));
        try std.testing.expectEqual(@as(?u32, null), try plan.physical(rows - 1, last + 1));
        try std.testing.expect(plan.tiles <= limit.tiles and plan.tail_chunks <= limit.tail_chunks);
    }
}

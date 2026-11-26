//! Attention metadata reuse checked against the allocating planner, without a GPU.
const std = @import("std");
const planning = @import("families/qwen27/attention_plan.zig");
const Scratch = @import("families/qwen27/attention_scratch.zig").Scratch;
const t = std.testing;

fn equal(a: planning.Plan, b: planning.Plan) !void {
    try t.expectEqual(a.geometry, b.geometry);
    inline for (.{ "streams", "rows", "tiles", "prefix_chunks", "tail_chunks", "fused", "pairs" }) |field|
        try t.expectEqual(@field(a, field), @field(b, field));
    try t.expectEqualSlices(i32, &a.meta, &b.meta);
    inline for (.{ "tile_stream", "q_rows", "nodes", "paths", "positions", "mates" }) |field|
        try t.expectEqualSlices(i32, @field(a, field), @field(b, field));
}

test "qwen27 plan scratch: warmed one/four-stream plans make no child allocations" {
    var counted: t.FailingAllocator = .init(t.allocator, .{});
    const a = counted.allocator();
    var scratch: Scratch = .{};
    defer scratch.deinit();
    var windows: [4]planning.Window = undefined;
    for (&windows, 0..) |*w, i| w.* = .{ .parents = &.{ -1, 0, 0, 1, 2 }, .start = @intCast(8190 + i), .capacity = 32768 };
    for (0..3) |_| {
        var p = try scratch.plan(a, &windows, .{});
        p.deinit();
    }
    const before = counted.allocations;
    for (0..40) |i| {
        windows[0].start = @intCast(600 + i);
        var p = try scratch.plan(a, windows[0..if (i % 2 == 0) 1 else 4], .{});
        p.deinit();
    }
    try t.expectEqual(before, counted.allocations);
}

test "qwen27 plan scratch: exact metadata after trees chains chunks joins and geometry changes" {
    var scratch: Scratch = .{};
    defer scratch.deinit();
    var chain: [2048]i32 = undefined;
    for (&chain, 0..) |*p, i| p.* = @as(i32, @intCast(i)) - 1;
    var windows: [4]planning.Window = undefined;
    for ([_]bool{ false, true }) |shared| for ([_]bool{ false, true }) |fuse| for ([_]u32{ 1, 4, 16, 127, 128, 512, 2048 }) |rows| {
        const geometry = planning.Geometry{ .shared_prefix = shared, .fuse_chain = fuse, .solo_prompt = true };
        for ([_]usize{ 1, 4 }) |streams| {
            if (rows > 128 and (streams != 1 or !fuse)) continue;
            if (rows * streams > (if (shared and fuse) @as(u32, 2048) else @as(u32, 128)) and streams != 1) continue;
            for ([_]u32{ 0, 63, 64, 511, 8191, 8192, 32768 }) |start| {
                for (windows[0..streams], 0..) |*w, i| w.* = .{ .parents = chain[0..rows], .start = start + @as(u32, @intCast(i)), .capacity = 65536 };
                var reference = try planning.Plan.init(t.allocator, windows[0..streams], geometry);
                defer reference.deinit();
                var actual = try scratch.plan(t.allocator, windows[0..streams], geometry);
                defer actual.deinit();
                reference.pair(&.{});
                actual.pair(&.{});
                try equal(reference, actual);
            }
        }
    };
    for (0..8) |i| {
        const parents: []const i32 = if (i % 2 == 0) &.{ -1, 0, 0, 1, 2 } else &.{ -1, 0 };
        windows[0] = .{ .parents = parents, .start = @intCast(600 + i), .capacity = 1024 };
        var reference = try planning.Plan.init(t.allocator, windows[0..1], .{});
        defer reference.deinit();
        var actual = try scratch.plan(t.allocator, windows[0..1], .{});
        defer actual.deinit();
        try equal(reference, actual);
    }
}

test "qwen27 plan scratch: refusal and allocation failure leave the next plan usable" {
    var scratch: Scratch = .{};
    defer scratch.deinit();
    try t.expectError(error.AttentionParents, scratch.plan(t.allocator, &.{.{ .parents = &.{ -1, 2 }, .start = 0, .capacity = 16 }}, .{}));
    var failed: t.FailingAllocator = .init(t.allocator, .{ .fail_index = 0 });
    // A fresh owner deliberately has no buffer to reuse.
    var empty: Scratch = .{};
    defer empty.deinit();
    try t.expectError(error.OutOfMemory, empty.plan(failed.allocator(), &.{.{ .parents = &.{-1}, .start = 0, .capacity = 16 }}, .{}));
    var a = try scratch.plan(t.allocator, &.{.{ .parents = &.{-1}, .start = 0, .capacity = 16 }}, .{});
    defer a.deinit();
    try t.expectEqualSlices(i32, &.{0}, a.positions);
}

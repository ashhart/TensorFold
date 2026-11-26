//! Real runner attention metadata and ownership without a GPU or shader compiler.
const std = @import("std");
const q = @import("tensorfold").qwen27;
test "qwen27 plan owner: reusable runner plans preserve pairing and independently owned plans" {
    const a = std.testing.allocator;
    var model: q.model.Model = undefined;
    model.kernels.prompt = true;
    var runner: q.decode_round.Runner = undefined;
    runner.allocator = a;
    runner.model = &model;
    runner.kv_step = null;
    runner.capacity = 65536;
    runner.geometry = .{ .shared_prefix = false, .solo_prompt = true };
    runner.attention.join = @splat(@import("metal").Pipeline{ .id = @ptrFromInt(8) });
    runner.plan_scratch = .{};
    defer runner.plan_scratch.deinit();
    var round = try q.round_plan.Round.init(a, &.{
        .{ .slot = 0, .start = 600, .capacity = 65536, .ids = &.{ 1, 2, 3 } },
        .{ .slot = 1, .start = 8190, .capacity = 65536, .ids = &.{ 4, 5 } },
    }, 4, 2, 16);
    defer round.deinit();
    var owned = try runner.attentionPlan(&round);
    defer owned.deinit();
    const positions = try a.dupe(i32, owned.positions);
    defer a.free(positions);
    for ([_]bool{ false, true }) |prompt| {
        model.kernels.prompt = prompt;
        var current = try q.attention_scratch.build(&runner, &round, &runner.plan_scratch);
        defer current.deinit();
        var reference = try runner.attentionPlan(&round);
        defer reference.deinit();
        try std.testing.expectEqual(reference.geometry, current.geometry);
        try std.testing.expectEqual(reference.pairs, current.pairs);
        try std.testing.expect(current.pairs > 0);
        try std.testing.expectEqualSlices(i32, &reference.meta, &current.meta);
        inline for (.{ "tile_stream", "q_rows", "nodes", "paths", "positions", "mates" }) |field|
            try std.testing.expectEqualSlices(i32, @field(reference, field), @field(current, field));
        try std.testing.expectEqualSlices(i32, positions, owned.positions);
    }
}

fn scratchFailureCycle(a: std.mem.Allocator) !void {
    var scratch: q.attention_scratch.Scratch = .{};
    defer scratch.deinit();
    var parents: [2048]i32 = undefined;
    for (&parents, 0..) |*p, i| p.* = @as(i32, @intCast(i)) - 1;
    for ([_]usize{ 1, 128, 2048, 1, 2048, 1 }) |rows| {
        var p = try scratch.plan(a, &.{.{ .parents = parents[0..rows], .start = 8191, .capacity = 65536 }}, .{});
        defer p.deinit();
        try std.testing.expectEqual(@as(i32, @intCast(8191 + rows - 1)), p.positions[rows - 1]);
    }
    _ = scratch.arena.?.reset(.{ .retain_with_limit = 2 << 20 });
    try std.testing.expect(scratch.arena.?.queryCapacity() <= 2 << 20);
}

test "qwen27 plan owner: scratch growth and failed preheating release every allocation" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, scratchFailureCycle, .{});
}

test "qwen27 plan owner: another runner's metadata and held plan remain untouched" {
    const a = std.testing.allocator;
    var first: q.attention_scratch.Scratch = .{};
    defer first.deinit();
    var second: q.attention_scratch.Scratch = .{};
    defer second.deinit();
    var p = try first.plan(a, &.{.{ .parents = &.{ -1, 0, 0 }, .start = 600, .capacity = 1024 }}, .{});
    defer p.deinit();
    const positions = try a.dupe(i32, p.positions);
    defer a.free(positions);
    for (0..12) |i| {
        var other = try second.plan(a, &.{.{ .parents = &.{-1}, .start = @intCast(8190 + i), .capacity = 65536 }}, .{});
        defer other.deinit();
        try std.testing.expectEqualSlices(i32, positions, p.positions);
    }
}

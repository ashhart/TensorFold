//! Early production runner guards execute on the CPU without creating a device or touching a buffer.
const std = @import("std");
const q = @import("tensorfold").qwen27;

test "qwen27 bank boundary: a pending keep rejects resizing and trimming" {
    var runner: q.decode_round.Runner = undefined;
    runner.failed = false;
    runner.active = @splat(0x17);
    runner.gdn.before = null;
    runner.gdn.before_slots = 7;
    const round: q.round_plan.Round = undefined;
    try std.testing.expectError(error.RoundNotReady, runner.verify(&round));
    try std.testing.expectError(error.RoundNotReady, runner.advance(&round, .all));
    runner.trimBefore();
    try std.testing.expectEqual(@as(usize, 7), runner.gdn.before_slots);
    runner.active = null;
    runner.trimBefore();
    try std.testing.expectEqual(@as(usize, 0), runner.gdn.before_slots);
}

fn unusedTail(ptr: *anyopaque, _: @import("metal").ComputeEncoder) !void {
    const called: *bool = @ptrCast(@alignCast(ptr));
    called.* = true;
}
test "qwen27 bank boundary: a verify tail cannot run on prompt heads or accepted chains" {
    var model: q.model.Model = undefined;
    model.frame.capacity = 16;
    var runner: q.decode_round.Runner = undefined;
    runner.failed = false;
    runner.active = null;
    runner.model = &model;
    var called = false;
    runner.tail = .{ .ptr = &called, .encode = unusedTail };
    var round: q.round_plan.Round = undefined;
    round.ids = @constCast(&[_]u32{0});
    try std.testing.expectError(error.BadVerifyTail, runner.verifyHead(&round, .none));
    try std.testing.expectError(error.BadVerifyTail, runner.advance(&round, .all));
    try std.testing.expect(!called);
}

fn unusedSeamPart(_: *anyopaque) q.forward.Seam {
    unreachable;
}
fn unusedSeamStage(_: *anyopaque, _: *@import("metal").ComputeEncoder, _: u32, _: q.projection.Ref, _: u32, _: u32, _: u32) void {
    unreachable;
}
fn unusedSeamGather(_: *anyopaque, _: *@import("metal").ComputeEncoder, _: u32, _: q.projection.Ref, _: q.projection.Ref, _: u32, _: u32, _: u32) void {
    unreachable;
}
fn unusedSeamFailed(_: *anyopaque) bool {
    unreachable;
}
test "qwen27 bank boundary: a rank's verify never runs a tail" {
    var model: q.model.Model = undefined;
    model.frame.capacity = 16;
    var runner: q.decode_round.Runner = undefined;
    runner.failed = false;
    runner.active = null;
    runner.model = &model;
    var called = false;
    runner.tail = .{ .ptr = &called, .encode = unusedTail };
    runner.reduce = .{ .ptr = undefined, .part = unusedSeamPart, .stage = unusedSeamStage, .gather = unusedSeamGather, .failed = unusedSeamFailed };
    var round: q.round_plan.Round = undefined;
    round.ids = @constCast(&[_]u32{0});
    try std.testing.expectError(error.TailOnRank, runner.verifyHead(&round, .all));
    try std.testing.expect(!called);
}

test "qwen27 bank boundary: snapshot drain preserves the picked bonus while reset forgets it" {
    var g: q.dflash.generation.Generation = undefined;
    g.keep = null;
    g.pending = 17;
    try g.finishKeep();
    try std.testing.expectEqual(@as(?u32, 17), g.pending);
    try g.settle();
    try std.testing.expectEqual(@as(?u32, null), g.pending);
}

test "qwen27 bank boundary: decode rows retain the pre-M5 limit while solo prompt rows follow the frame" {
    var model: q.model.Model = undefined;
    var runner: q.decode_round.Runner = undefined;
    runner.model = &model;
    for ([_]u32{ 128, 512, 2048 }) |capacity| {
        model.frame.capacity = capacity;
        runner.geometry = .{ .shared_prefix = false, .solo_prompt = true };
        try std.testing.expectEqual(@as(u32, 128), runner.decodeRows());
        try std.testing.expectEqual(capacity, runner.promptRows(2048));
        runner.geometry.solo_prompt = false;
        try std.testing.expectEqual(@as(u32, 128), runner.promptRows(2048));
        runner.geometry.shared_prefix = true;
        try std.testing.expectEqual(capacity, runner.decodeRows());
    }
}

test "qwen27 bank boundary: prompt partial scratch fits packed rows and rejects one tile past its bytes" {
    const planning = q.attention.planning;
    var parents: [2048]i32 = undefined;
    for (&parents, 0..) |*p, i| p.* = @as(i32, @intCast(i)) - 1;
    for ([_]u32{ 128, 512, 1024, 2048 }) |rows| {
        const limits = q.attention.Limits{ .rows = rows, .streams = 1, .keys = 65536, .shared_prefix = false, .geometry = .{ .shared_prefix = false, .solo_prompt = true } };
        const layout = q.attention.Layout.init(limits);
        const rp = @as(usize, layout.tiles) * 16;
        const want = @max(@as(usize, @min(rows, planning.max_rows)) * 16, rp);
        try std.testing.expectEqual(want, layout.chunk_rows);
        try std.testing.expectEqual(@as(usize, 4) * layout.tail_chunks * want * 256 * 4, layout.sizes[10]);
        try std.testing.expectEqual(@as(usize, 4) * @min(rows, planning.max_rows) * 16 * 256 * 2, layout.sizes[6]);
        var scratch: q.attention.Scratch = undefined;
        scratch.limits = limits;
        scratch.tiles = layout.tiles;
        scratch.chunks = layout.chunks;
        scratch.tail_chunks = layout.tail_chunks;
        scratch.chunk_rows = layout.chunk_rows;
        var plan = try planning.Plan.init(std.testing.allocator, &.{.{ .parents = parents[0..rows], .start = 16384, .capacity = 65536 }}, limits.geometry);
        defer plan.deinit();
        const chunks: u32 = @intCast(@divFloor(16384 + rows + planning.chunk - 1, planning.chunk));
        try scratch.checkPrompt(chunks, plan.tiles, false);
        scratch.tiles = @intCast(layout.chunk_rows / 16 + 1);
        try std.testing.expectError(error.AttentionLimits, scratch.checkPrompt(layout.tail_chunks, scratch.tiles, false));
    }
}

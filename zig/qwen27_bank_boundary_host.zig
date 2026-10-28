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

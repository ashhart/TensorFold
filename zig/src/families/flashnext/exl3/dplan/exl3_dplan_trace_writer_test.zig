const std = @import("std");
const trace = @import("exl3_dplan_trace.zig");
const replay = @import("exl3_dplan_replay.zig");

test "exl3: CUDA writer bytes replay all 1200 rounds and terminal words exactly on CPU" {
    var snapshot = try trace.Snapshot.decode(std.testing.allocator, @embedFile("dplan_trace_fixture"));
    defer snapshot.deinit();
    const view = try snapshot.view();
    const result = try replay.verifyTrace(std.testing.allocator, view);
    try std.testing.expectEqual(@as(usize, 1200), result.rounds);
    try std.testing.expectEqual(@as(usize, 1200), result.matched_transitions);
    try std.testing.expectEqual(@as(u64, 0), result.delta.overflow);
    try std.testing.expect(result.delta.mapped_picks > 0);
    try std.testing.expect(result.delta.copies > 0);
    try std.testing.expect(result.delta.evictions > 0);
}

test "exl3: CUDA ownership checkpoints replay every segment and final transferred state" {
    var snapshot = try trace.Snapshot.decode(std.testing.allocator, @embedFile("dplan_checkpoint_fixture"));
    defer snapshot.deinit();
    const view = try snapshot.view();
    const result = try replay.verifyTraceWithFinal(std.testing.allocator, view, view.terminalSeed());
    try std.testing.expectEqual(@as(usize, 1024), result.rounds);
    try std.testing.expectEqual(@as(usize, 1024), result.matched_transitions);
    try std.testing.expectEqual(@as(usize, 4), view.checkpoints());
    try std.testing.expectEqual(@as(usize, 0), (try view.checkpoint(0)).roundIndex());
    try std.testing.expectEqual(@as(usize, 512), (try view.checkpoint(1)).roundIndex());
    try std.testing.expectEqual(@as(usize, 768), (try view.checkpoint(2)).roundIndex());
    try std.testing.expectEqual(@as(usize, 1024), (try view.checkpoint(3)).roundIndex());
    try std.testing.expect(result.delta.copies > 0);
}

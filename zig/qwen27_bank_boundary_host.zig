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

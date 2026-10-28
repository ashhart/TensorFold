//! A progressing long prefill is not a stalled native call; stopped and idle scopes never expire.
const std = @import("std");
const progress = @import("round_progress.zig");
const t = std.testing;

test "round progress: idle waits and completed work never trigger; nested native waits retain ownership" {
    var p: progress.Progress = .{ .io = t.io };
    try t.expect(p.stalledAt(10, 1_000_000) == null);
    var outer = p.beginAt(.round, 100);
    var inner = p.beginAt(.follower, 101);
    try t.expect(p.stalledAt(10, 109) == null);
    try t.expectEqual(progress.Phase.follower, p.stalledAt(10, 110).?);
    inner.end();
    try t.expectEqual(progress.Phase.round, p.stalledAt(10, 110).?);
    outer.end();
    outer.end();
    try t.expect(p.stalledAt(10, 1_000_000) == null);
    try t.expectEqual(@as(u32, 0), p.active);
}
test "round progress: advancing prefill can exceed the timeout many times without a false exit" {
    var p: progress.Progress = .{ .io = t.io };
    var scope = p.beginAt(.round, 0);
    defer scope.end();
    for (1..1000) |i| {
        const at = i * 9;
        try t.expect(p.stalledAt(10, at) == null);
        p.touchAt(at);
    }
    try t.expectEqual(progress.Phase.round, p.stalledAt(10, 999 * 9 + 10).?);
    try t.expect(p.stalledAt(0, 1_000_000) == null);
}
test "round progress: the next operation owns a fresh deadline and stale touches cannot reverse progress" {
    var p: progress.Progress = .{ .io = t.io };
    var first = p.beginAt(.round, 10);
    p.touchAt(25);
    p.touchAt(20);
    try t.expect(p.stalledAt(10, 34) == null);
    first.end();
    var next = p.beginAt(.follower, 1000);
    defer next.end();
    try t.expect(p.stalledAt(10, 1009) == null);
    try t.expectEqual(progress.Phase.follower, p.stalledAt(10, 1010).?);
}
test "round progress: stall timeout is opt-in, checked and independent of health configuration" {
    try t.expectEqual(@as(u64, 0), try progress.timeoutNs(null));
    try t.expectEqual(@as(u64, 0), try progress.timeoutNs("0"));
    try t.expectEqual(@as(u64, 300) * std.time.ns_per_s, try progress.timeoutNs("300"));
    for ([_][]const u8{ "", "-1", "no", "18446744074", "18446744073709551616" }) |text|
        try t.expectError(error.BadStallTimeout, progress.timeoutNs(text));
}

//! Decisions' label scores reach a lanes backend that gives them, on the engine thread, and nowhere else.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;

/// The fake backend's entries with a score of its own: label i scores prompt.len + i, the mass 7.5.
const Scorer = struct {
    var engine_thread: ?std.Thread.Id = null;
    var calls: usize = 0;

    fn score(_: *anyopaque, prompt: []const u32, labels: []const u32, logits: []f64, mass: bool) anyerror!f64 {
        engine_thread = std.Thread.getCurrentId();
        calls += 1;
        if (labels.len == 0) return error.EmptyLabels;
        for (logits, 0..) |*v, i| v.* = @floatFromInt(prompt.len + i);
        return if (mass) 7.5 else 0;
    }
};

test "an engine whose backend gives no score refuses decisions as unsupported" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{}, 1, 0);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 1 });
    var logits: [2]f64 = undefined;
    try std.testing.expectError(error.Unsupported, host.engine().score(&.{ 1, 2 }, &.{ 3, 4 }, &logits, true));
}

test "a backend's score runs on the engine thread, a failure comes back as Failed, and a stopped host fails" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{}, 1, 0);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    const plain = target.backend();
    var vt = plain.vtable.*;
    vt.score = Scorer.score;
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, .{ .ptr = plain.ptr, .vtable = &vt }, clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 1 });
    try host.start();
    const engine = host.engine();
    Scorer.calls = 0;
    var logits: [3]f64 = undefined;
    const lse = try engine.score(&.{ 1, 2, 3, 4 }, &.{ 9, 8, 7 }, &logits, true);
    try std.testing.expectEqual(@as(f64, 7.5), lse);
    try std.testing.expectEqualSlices(f64, &.{ 4, 5, 6 }, &logits);
    try std.testing.expect(Scorer.engine_thread.? != std.Thread.getCurrentId());
    try std.testing.expectError(error.Failed, engine.score(&.{1}, &.{}, logits[0..0], true));
    try std.testing.expectEqual(@as(usize, 2), Scorer.calls);
    host.stop();
    try std.testing.expectError(error.Failed, engine.score(&.{1}, &.{2}, logits[0..1], false));
    try std.testing.expectEqual(@as(usize, 2), Scorer.calls);
}

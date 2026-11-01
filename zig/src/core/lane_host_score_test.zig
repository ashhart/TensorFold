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

/// A fake decider: label j of a branch scores trunk.len + branch.len + j; a branch of token 0 is refused.
const Decides = struct {
    var engine_thread: ?std.Thread.Id = null;
    var calls: usize = 0;

    fn run(_: *anyopaque, d: *api.Decision) anyerror!void {
        engine_thread = std.Thread.getCurrentId();
        calls += 1;
        var at: usize = 0;
        for (d.branches, d.counts) |b, n| {
            if (b[0] == 0) {
                d.problem = "branch zero";
                return error.Refused;
            }
            for (d.logits[at..][0..n], 0..) |*v, j| v.* = @floatFromInt(d.trunk.len + b.len + j);
            at += n;
        }
        d.read = d.trunk.len;
    }
};

test "a host's decider answers whole decisions on the engine thread, refusals keep their words, a stopped host fails" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{}, 1, 0);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 1 });
    var unused: u8 = 0;
    host.decider = .{ .ctx = &unused, .run = Decides.run };
    try host.start();
    const engine = host.engine();
    try std.testing.expect(engine.vtable.decide != null);
    Decides.calls = 0;
    var logits: [3]f64 = undefined;
    var d: api.Decision = .{ .trunk = &.{ 1, 2, 3 }, .branches = &.{ &.{ 5, 6 }, &.{7} }, .counts = &.{ 2, 1 }, .labels = &.{ 9, 8, 7 }, .logits = &logits };
    try engine.decide(&d);
    try std.testing.expectEqualSlices(f64, &.{ 5, 6, 4 }, &logits);
    try std.testing.expect(Decides.engine_thread.? != std.Thread.getCurrentId());
    var refused: api.Decision = .{ .trunk = &.{1}, .branches = &.{&.{0}}, .counts = &.{1}, .labels = &.{9}, .logits = logits[0..1] };
    try std.testing.expectError(error.Refused, engine.decide(&refused));
    try std.testing.expectEqualStrings("branch zero", refused.problem);
    host.stop();
    try std.testing.expectError(error.Failed, engine.decide(&d));
    try std.testing.expectEqual(@as(usize, 2), Decides.calls);
}

//! Decisions' label scores reach a lanes backend that gives them, on the engine thread, and nowhere else.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const scoring = @import("lane_host_score.zig");

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
    const log_level = std.testing.log_level;
    std.testing.log_level = .err;
    defer std.testing.log_level = log_level;
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

/// Closes its host during the first score, as a drain's halt does while the engine thread answers a queue.
const Closer = struct {
    var host: ?*LaneHost = null;
    var calls: usize = 0;

    fn score(_: *anyopaque, _: []const u32, _: []const u32, logits: []f64, _: bool) anyerror!f64 {
        calls += 1;
        const h = host.?;
        h.lock();
        h.closing = true;
        h.unlock();
        @memset(logits, 1);
        return 0;
    }
};

test "a queue answered while the host closes runs the started score and fails the rest" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{}, 1, 0);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    const plain = target.backend();
    var vt = plain.vtable.*;
    vt.score = Closer.score;
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, .{ .ptr = plain.ptr, .vtable = &vt }, clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 1 });
    defer host.scores.asks.deinit(gpa);
    Closer.host = &host;
    Closer.calls = 0;
    var logits: [3][1]f64 = .{ .{0}, .{0}, .{0} };
    var asks: [3]scoring.Ask = undefined;
    for (&asks, &logits) |*ask, *row| {
        ask.* = .{ .prompt = &.{1}, .labels = &.{2}, .logits = row, .mass = false };
        try host.scores.asks.append(gpa, ask);
    }
    scoring.answer(&host);
    try std.testing.expectEqual(@as(usize, 1), Closer.calls);
    try std.testing.expect(!asks[0].failed and asks[0].done);
    try std.testing.expectEqual(@as(f64, 1), logits[0][0]);
    for (asks[1..]) |ask| try std.testing.expect(ask.failed and ask.done);
}

/// The fake decision: branch b's label i scores trunk.len + branch.len + i; no labels is refused, no trunk fails.
const Decider = struct {
    var calls: usize = 0;
    var engine_thread: ?std.Thread.Id = null;

    fn decide(_: *anyopaque, trunk: []const u32, branches: []const []const u32, counts: []const usize, labels: []const u32, logits: []f64, problem: *[]const u8) anyerror!void {
        calls += 1;
        engine_thread = std.Thread.getCurrentId();
        if (labels.len == 0) {
            problem.* = "no labels to score";
            return error.Refused;
        }
        if (trunk.len == 0) return error.DeviceLost;
        var at: usize = 0;
        for (branches, counts) |b, n| {
            for (logits[at..][0..n], 0..) |*v, i| v.* = @floatFromInt(trunk.len + b.len + i);
            at += n;
        }
    }
};

fn hosted(gpa: std.mem.Allocator, target: *lanes.fake.Fake, vt: *lanes.backend.Backend.VTable, with_decide: bool) lanes.backend.Backend {
    vt.* = target.backend().vtable.*;
    vt.score = Scorer.score;
    if (with_decide) vt.decide = Decider.decide;
    _ = gpa;
    return .{ .ptr = target.backend().ptr, .vtable = vt };
}

test "a backend with score and decide gets Engine.decide; score alone keeps the per-question path" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{}, 1, 0);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var vt: lanes.backend.Backend.VTable = undefined;
    var core = lanes.Engine.init(gpa, &cfg, hosted(gpa, &target, &vt, false), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 1 });
    try std.testing.expect(host.engine().vtable.score != null and host.engine().vtable.decide == null);
    vt.decide = Decider.decide;
    try std.testing.expect(host.engine().vtable.decide != null);
    // a backend that decides but gives no score gets neither hook
    vt.score = null;
    try std.testing.expect(host.engine().vtable.score == null and host.engine().vtable.decide == null);
}

test "a decision runs on the engine thread: each branch's logits, the tokens read; refusals keep their reason" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{}, 1, 0);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var vt: lanes.backend.Backend.VTable = undefined;
    var core = lanes.Engine.init(gpa, &cfg, hosted(gpa, &target, &vt, true), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 1 });
    try host.start();
    const engine = host.engine();
    Decider.calls = 0;
    var logits: [3]f64 = undefined;
    var d: api.Decision = .{ .trunk = &.{ 1, 2, 3 }, .branches = &.{ &.{4}, &.{ 5, 6 } }, .counts = &.{ 2, 1 }, .labels = &.{ 7, 8, 9 }, .logits = &logits };
    try engine.decide(&d);
    try std.testing.expectEqualSlices(f64, &.{ 4, 5, 5 }, &logits);
    try std.testing.expectEqual(@as(usize, 6), d.read);
    try std.testing.expect(Decider.engine_thread.? != std.Thread.getCurrentId());
    // the backend's refusal is the client's error with the backend's reason; any other error is the engine's failure
    var empty: api.Decision = .{ .trunk = &.{1}, .branches = &.{&.{2}}, .counts = &.{0} };
    try std.testing.expectError(error.Refused, engine.decide(&empty));
    try std.testing.expectEqualStrings("no labels to score", empty.problem);
    var lost: api.Decision = .{ .trunk = &.{}, .branches = &.{&.{2}}, .counts = &.{1}, .labels = &.{3}, .logits = logits[0..1] };
    try std.testing.expectError(error.Failed, engine.decide(&lost));
    var pictured: api.Decision = .{ .trunk = &.{1}, .pictures = &.{"data:image/png;base64,"}, .branches = &.{&.{2}}, .counts = &.{1}, .labels = &.{3}, .logits = logits[0..1] };
    try std.testing.expectError(error.Refused, engine.decide(&pictured));
    try std.testing.expect(std.mem.indexOf(u8, pictured.problem, "text only") != null);
    try std.testing.expectEqual(@as(usize, 3), Decider.calls);
    host.stop();
    try std.testing.expectError(error.Failed, engine.decide(&d));
}

test "a decision queued behind a closing host fails without running" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{}, 1, 0);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    const plain = target.backend();
    var vt = plain.vtable.*;
    vt.score = Closer.score;
    vt.decide = Decider.decide;
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, .{ .ptr = plain.ptr, .vtable = &vt }, clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 1 });
    defer host.scores.asks.deinit(gpa);
    Closer.host = &host;
    Closer.calls = 0;
    Decider.calls = 0;
    var first: [1]f64 = .{0};
    var second: [1]f64 = .{0};
    var asks = [2]scoring.Ask{
        .{ .prompt = &.{1}, .labels = &.{2}, .logits = &first },
        .{ .kind = .decide, .prompt = &.{1}, .labels = &.{2}, .logits = &second, .branches = &.{&.{3}}, .counts = &.{1} },
    };
    for (&asks) |*ask| try host.scores.asks.append(gpa, ask);
    scoring.answer(&host);
    try std.testing.expectEqual(@as(usize, 1), Closer.calls);
    try std.testing.expectEqual(@as(usize, 0), Decider.calls);
    try std.testing.expect(asks[1].failed and asks[1].done);
}

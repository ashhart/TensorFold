//! Streams a release ends: the live list sweeps them in that step, and a discard reaps a peer its release ended.
const std = @import("std");
const Engine = @import("engine.zig").Engine;
const sm = @import("stream.zig");
const fake = @import("fake.zig");
const shared = @import("engine_test.zig");
const model = shared.model;
const p1 = shared.p1;
const p2 = shared.p2;

const gpa = std.testing.allocator;

/// A release ends a paused peer whose migration fails.
fn endWaiting(ctx: *anyopaque, released: *sm.Stream) void {
    const waiting: *sm.Stream = @ptrCast(@alignCast(ctx));
    if (released == waiting or waiting.finished) return;
    waiting.paused = false;
    waiting.finished = true;
    waiting.reason = .@"error";
}

test "a stream another's release ends leaves the live list in that step, before its job can be freed" {
    var cfg = try model();
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    var waiting = try sm.Stream.init(gpa, .{ .id = "waiting", .prompt = &p2, .max_new = 40, .eos = &.{96}, .drafts = false });
    defer waiting.deinit(gpa);
    var short = try sm.Stream.init(gpa, .{ .id = "short", .prompt = &p1, .max_new = 3, .eos = &.{96}, .drafts = false });
    defer short.deinit(gpa);
    try engine.addStream(&waiting); // first in the live list: the sweep is past it when the short one releases
    waiting.paused = true;
    target.release_hook = endWaiting;
    target.release_hook_ctx = &waiting;
    try engine.addStream(&short);
    try std.testing.expect(!short.finished); // its reply ends in a step, released by that step's sweep
    var steps: usize = 0;
    while (!short.finished and steps < 8) : (steps += 1) try engine.step();
    try std.testing.expect(short.finished);
    try std.testing.expect(waiting.finished and waiting.reason == .@"error");
    try std.testing.expectEqual(@as(usize, 0), engine.live.items.len);
}

test "a discard whose release ends another stream leaves neither in the live list" {
    var cfg = try model();
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    var waiting = try sm.Stream.init(gpa, .{ .id = "waiting", .prompt = &p2, .max_new = 40, .eos = &.{96}, .drafts = false });
    defer waiting.deinit(gpa);
    var other = try sm.Stream.init(gpa, .{ .id = "other", .prompt = &p1, .max_new = 40, .eos = &.{96}, .drafts = false });
    defer other.deinit(gpa);
    try engine.addStream(&waiting);
    try engine.addStream(&other);
    try engine.step();
    waiting.paused = true;
    target.release_hook = endWaiting;
    target.release_hook_ctx = &waiting;
    engine.discard(&other);
    try std.testing.expect(waiting.finished and waiting.reason == .@"error");
    try std.testing.expectEqual(@as(usize, 0), engine.live.items.len);
}

test "discard reaps a peer even when trace allocations fail" {
    var cfg = try model();
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    var waiting = try sm.Stream.init(gpa, .{ .id = "waiting", .prompt = &p2, .max_new = 40, .drafts = false });
    defer waiting.deinit(gpa);
    var other = try sm.Stream.init(gpa, .{ .id = "other", .prompt = &p1, .max_new = 40, .drafts = false });
    defer other.deinit(gpa);
    try engine.addStream(&waiting);
    try engine.addStream(&other);
    waiting.paused = true;
    target.release_hook = endWaiting;
    target.release_hook_ctx = &waiting;
    var failing = std.testing.FailingAllocator.init(gpa, .{ .fail_index = 0 });
    var log = @import("events.zig").Log.init(failing.allocator());
    defer log.deinit();
    engine.log = &log;
    engine.discard(&other);
    try std.testing.expect(waiting.finished);
    try std.testing.expectEqual(@as(usize, 0), engine.live.items.len);
    try std.testing.expectEqual(@as(usize, 0), target.lanes.count());
}

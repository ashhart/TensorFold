//! A stepping target's prompt through the drafter wrapper: a step at a time, each step's rows absorbed.
const std = @import("std");
const Config = @import("config.zig").Config;
const Engine = @import("engine.zig").Engine;
const sm = @import("stream.zig");
const fake = @import("fake.zig");
const Drafted = @import("drafted.zig").Drafted;
const SuffixLookup = @import("proposer.zig").SuffixLookup;
const dt = @import("drafted_test.zig");

const gpa = std.testing.allocator;
const Case = dt.Case;
const run = dt.run;
const freeOut = dt.freeOut;
const base = dt.base;
const FakeDrafter = dt.FakeDrafter;
const long = dt.long;
const p1 = dt.p1;
const sampled = dt.sampled;

// FakeDrafter's open, counting opens that found the stream's target lane already open (its cache, a pool lane's run)
var lane_target: *const fake.Fake = undefined;
var opened_on_lane: u32 = 0;
var fake_open: *const fn (*anyopaque, *sm.Stream) anyerror!void = undefined;
fn countedOpen(ptr: *anyopaque, s: *sm.Stream) anyerror!void {
    opened_on_lane += @intFromBool(lane_target.lanes.contains(s));
    return fake_open(ptr, s);
}

test "a stepping target's prompt goes in a step at a time through the wrapper: the drafter absorbs each step's rows, the reply is the whole prefill's" {
    const cases = [_]Case{ .{ .prompt = &long }, .{ .prompt = &p1, .sampling = sampled } };
    const whole = try run(&cases, .wrapped, false);
    defer freeOut(whole);
    var target: fake.Fake = .{ .gpa = gpa, .prefill_chunks = 3 };
    defer target.deinit();
    var head: FakeDrafter = .{ .facts_ = .{ .depth = 6, .step_ms = 0.5 } };
    defer head.deinit();
    var hd = head.drafter();
    var vt = hd.vtable.*;
    fake_open, vt.open, lane_target, opened_on_lane = .{ vt.open, countedOpen, &target, 0 };
    hd.vtable = &vt;
    var wrapped = try Drafted.init(gpa, target.stepped(), hd);
    defer wrapped.deinit();
    try std.testing.expect(wrapped.backend().vtable.prefill_step != null and wrapped.backend().vtable.reports_fill);
    var cfg = try Config.init(gpa, wrapped.facts(base()), 16, 15);
    defer cfg.deinit(gpa);
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, wrapped.backend(), clock.clock());
    defer engine.deinit();
    var streams: [cases.len]sm.Stream = undefined;
    var proposers: [cases.len]SuffixLookup = undefined;
    for (cases, &streams, &proposers) |c, *s, *p| {
        p.* = try SuffixLookup.init(gpa, .{ .min_match = 4 });
        s.* = try sm.Stream.init(gpa, .{ .id = "s", .prompt = c.prompt, .max_new = c.max_new, .eos = &.{96}, .sampling = c.sampling, .proposer = p.proposer() });
    }
    defer for (&streams, &proposers) |*s, *p| {
        s.deinit(gpa);
        p.deinit();
    };
    for (&streams) |*s| {
        var steps: usize = 0;
        while (!try engine.fillStream(s, steps == 0)) steps += 1;
        try std.testing.expectEqual(@as(usize, 2), steps); // three steps: two not done, then the open
    }
    try std.testing.expectEqual(@as(u32, 2), head.opens);
    // each opened after the target's first step: its lane (a pool lane's run, which the drafter borrows) exists
    try std.testing.expectEqual(@as(u32, 2), opened_on_lane);
    while (engine.activeCount() > 0) try engine.step();
    for (whole.tokens, &streams) |want, *s| try std.testing.expectEqualSlices(u32, want, s.emitted());
}

const Stop = struct {
    flag: bool = false,
    fn check(ptr: *anyopaque) bool {
        const x: *Stop = @ptrCast(@alignCast(ptr));
        return x.flag;
    }
};

/// A restored prompt of `restored` rows stepped through the wrapper; with `cancel`, cancelled after its second step.
fn restoredSteps(restored: u32, cancel: bool) !void {
    var target: fake.Fake = .{ .gpa = gpa, .prefill_chunks = 4 };
    defer target.deinit();
    var head: FakeDrafter = .{ .facts_ = .{ .depth = 6, .step_ms = 0.5 } };
    defer head.deinit();
    var wrapped = try Drafted.init(gpa, target.stepped(), head.drafter());
    defer wrapped.deinit();
    var cfg = try Config.init(gpa, wrapped.facts(base()), 16, 15);
    defer cfg.deinit(gpa);
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, wrapped.backend(), clock.clock());
    defer engine.deinit();
    var early = try sm.Stream.init(gpa, .{ .id = "early", .prompt = long[0..restored], .max_new = 1, .eos = &.{96} });
    defer early.deinit(gpa);
    try target.backend().prefill(&early);
    const saved = try target.save(&early, restored);
    defer target.drop(saved);
    target.backend().release(&early);
    var stop: Stop = .{};
    var p = try SuffixLookup.init(gpa, .{ .min_match = 4 });
    defer p.deinit();
    var s = try sm.Stream.init(gpa, .{ .id = "s", .prompt = &long, .max_new = 8, .eos = &.{96}, .proposer = p.proposer(), .reuse = .{ .saved = saved, .at = restored }, .cancel_check = .{ .ptr = &stop, .check = Stop.check } });
    defer s.deinit(gpa);
    var steps: usize = 0;
    while (true) : (steps += 1) {
        if (cancel and steps == 2) {
            stop.flag = true;
            try std.testing.expectError(error.Cancelled, engine.fillStream(&s, false));
            return;
        }
        if (try engine.fillStream(&s, steps == 0)) break;
    }
    try std.testing.expectEqual(@as(usize, 3), steps);
    while (engine.activeCount() > 0) try engine.step();
}

test "a restored prompt stepped through the wrapper: partial and whole restores finish, a cancel after step 2 ends cleanly" {
    try restoredSteps(8, false);
    try restoredSteps(long.len, false);
    try restoredSteps(8, true);
    try restoredSteps(long.len, true);
}

test "a drafted backend forwards optional prompt cancellation and its failure before release" {
    const Cancel = struct {
        calls: usize = 0,
        fn call(ptr: *anyopaque, _: *sm.Stream) !void {
            const target: *fake.Fake = @ptrCast(@alignCast(ptr));
            const x: *@This() = @ptrCast(@alignCast(target.prefill_hook_ctx.?));
            x.calls += 1;
            return error.PeerFailed;
        }
    };
    for ([_]bool{ false, true }) |stepped| for ([_]bool{ false, true }) |enabled| {
        var target: fake.Fake = .{ .gpa = gpa };
        defer target.deinit();
        var cancel: Cancel = .{};
        target.prefill_hook_ctx = &cancel;
        var table = (if (stepped) target.stepped() else target.backend()).vtable.*;
        table.cancel_prefill = if (enabled) Cancel.call else null;
        var head: FakeDrafter = .{};
        defer head.deinit();
        var wrapped = try Drafted.init(gpa, .{ .ptr = &target, .vtable = &table }, head.drafter());
        defer wrapped.deinit();
        const backend = wrapped.backend();
        try std.testing.expectEqual(enabled, backend.vtable.cancel_prefill != null);
        var s = try sm.Stream.init(gpa, .{ .id = "cancel", .prompt = &p1, .max_new = 8 });
        defer s.deinit(gpa);
        if (enabled) try std.testing.expectError(error.PeerFailed, backend.vtable.cancel_prefill.?(backend.ptr, &s));
        try std.testing.expectEqual(@as(usize, @intFromBool(enabled)), cancel.calls);
        try std.testing.expectEqual(@as(usize, 0), target.lanes.count());
        try std.testing.expectEqual(@as(u32, 0), head.releases);
    };
}

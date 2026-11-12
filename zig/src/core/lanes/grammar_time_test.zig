//! Scripted host grammar costs, exact masks and pricing-clock isolation at the actual planned-window seam.
const std = @import("std");
const t = std.testing;
const be = @import("backend.zig");
const grammar = @import("grammar.zig");
const win = @import("windows.zig");
const sm = @import("stream.zig");
const timing = @import("grammar_time.zig");
const fake = @import("fake.zig");
const Engine = @import("engine.zig").Engine;
const et = @import("engine_test.zig");

const Clock = struct {
    ns: i128 = 0,
    reads: usize = 0,
    prices: usize = 0,
    fn clock(c: *@This()) be.Clock {
        return .{ .ptr = c, .vtable = &.{ .start = start, .elapsed_ms = elapsed, .host_ns = now } };
    }
    fn start(_: *anyopaque) void {}
    fn elapsed(ptr: *anyopaque, _: be.Mark) f64 {
        const c: *@This() = @ptrCast(@alignCast(ptr));
        c.prices += 1;
        return 10;
    }
    fn now(ptr: *anyopaque) i128 {
        const c: *@This() = @ptrCast(@alignCast(ptr));
        c.reads += 1;
        return c.ns;
    }
};
const Rules = struct {
    clock: *Clock,
    at: usize = 0,
    fail: bool = false,
    fn cast(p: *anyopaque) *@This() {
        return @ptrCast(@alignCast(p));
    }
    fn rules(r: *@This()) grammar.Rules {
        return .{ .ptr = r, .vtable = &.{ .accept = accept, .rollback = rollback, .terminated = terminated, .fill = fill, .drafts = drafts, .free = free } };
    }
    fn accept(p: *anyopaque, _: u32) anyerror!bool {
        const r = cast(p);
        r.clock.ns += 2;
        r.at += 1;
        return true;
    }
    fn rollback(p: *anyopaque, n: usize) anyerror!void {
        const r = cast(p);
        r.clock.ns += @intCast(3 * n);
        r.at -= n;
    }
    fn terminated(_: *anyopaque) bool {
        return false;
    }
    fn fill(p: *anyopaque, bits: []u32) anyerror!void {
        const r = cast(p);
        r.clock.ns += 5;
        if (r.fail) return error.MaskFailed;
        @memset(bits, 0x55aa55aa);
    }
    fn drafts(_: *anyopaque, _: []u32) anyerror!usize {
        return 0;
    }
    fn free(_: *anyopaque) void {}
};

fn stream() !sm.Stream {
    var s = try sm.Stream.init(t.allocator, .{ .id = "measured", .prompt = &.{1}, .max_new = 8, .eos = &.{96}, .drafts = false });
    s.pending = 9;
    return s;
}

test "grammar timing: exact per-stream and round sums reset by step with unchanged masks and no pricing reads" {
    var cfg = try et.model();
    defer cfg.deinit(t.allocator);
    var target: fake.Fake = .{ .gpa = t.allocator };
    defer target.deinit();
    var clock: Clock = .{};
    var e = Engine.init(t.allocator, &cfg, target.backend(), clock.clock());
    defer e.deinit();
    var log = @import("events.zig").Log.init(t.allocator);
    defer log.deinit();
    e.log = &log;
    var s1 = try stream();
    defer s1.deinit(t.allocator);
    var s2 = try stream();
    defer s2.deinit(t.allocator);
    var r1: Rules = .{ .clock = &clock };
    var r2 = r1;
    var c1 = grammar.Constraint.init(r1.rules(), 1, null);
    var c2 = grammar.Constraint.init(r2.rules(), 1, null);
    s1.grammar = &c1;
    s2.grammar = &c2;
    e.steps = 1;
    for ([_]*sm.Stream{ &s1, &s2 }) |s| {
        var p: win.Plan = .{ .stream = s, .position = 1, .kind = .forced, .tokens = &.{ 2, 3 } };
        try win.constrain(&e, &p);
        try t.expectEqualSlices(u32, &.{ 0x55aa55aa, 0x55aa55aa, 0x55aa55aa }, p.masks);
        try t.expectEqual(timing.Sum{ .windows = 1, .fills = 3, .window_ns = 25, .fill_ns = 15 }, s.grammar_round.total);
        const plain = try s.grammar.?.window(e.arena.allocator(), &.{ 9, 2, 3 }, &.{ -1, 0, 1 }, true, 3);
        try t.expectEqualSlices(u32, plain.masks, p.masks);
    }
    try t.expectEqual(timing.Sum{ .windows = 2, .fills = 6, .window_ns = 50, .fill_ns = 30 }, e.grammar_round.total);
    try t.expectEqual(@as(usize, 0), clock.prices);
    try t.expectEqual(@as(usize, 0), r1.at);
    try t.expectEqual(@as(usize, 0), r2.at);
    try @import("trail.zig").state(&e);
    const line = try log.line(t.allocator, 2);
    defer t.allocator.free(line);
    try t.expectEqualStrings("{\"ev\":\"grammar_round\",\"fill_ns\":30,\"fills\":6,\"step\":1,\"window_ns\":50,\"windows\":2}", line);
    e.steps = 2;
    var p: win.Plan = .{ .stream = &s1, .position = 1, .kind = .forced, .tokens = &.{} };
    try win.constrain(&e, &p);
    try t.expectEqual(timing.Sum{ .windows = 1, .fills = 1, .window_ns = 5, .fill_ns = 5 }, e.grammar_round.total);
    try t.expectEqual(timing.Sum{ .windows = 2, .fills = 4, .window_ns = 30, .fill_ns = 20 }, s1.grammar_time);
    try t.expectEqual(@as(u64, 1), s2.grammar_round.step);
}

test "grammar timing: unconstrained windows do not read clocks allocate masks or report work" {
    var cfg = try et.model();
    defer cfg.deinit(t.allocator);
    var target: fake.Fake = .{ .gpa = t.allocator };
    defer target.deinit();
    var clock: Clock = .{};
    var e = Engine.init(t.allocator, &cfg, target.backend(), clock.clock());
    defer e.deinit();
    var s = try stream();
    defer s.deinit(t.allocator);
    var p: win.Plan = .{ .stream = &s, .position = 1, .kind = .forced, .tokens = &.{ 2, 3 } };
    try win.constrain(&e, &p);
    try t.expectEqual(@as(usize, 0), clock.reads);
    try t.expectEqual(@as(usize, 0), clock.prices);
    try t.expectEqual(@as(usize, 0), p.masks.len);
    try t.expectEqual(timing.Sum{}, e.grammar_round.total);
}

test "grammar timing: failed fills retain their time and error and release the borrowed profile" {
    var cfg = try et.model();
    defer cfg.deinit(t.allocator);
    var target: fake.Fake = .{ .gpa = t.allocator };
    defer target.deinit();
    var clock: Clock = .{};
    var e = Engine.init(t.allocator, &cfg, target.backend(), clock.clock());
    defer e.deinit();
    var r: Rules = .{ .clock = &clock, .fail = true };
    var c = grammar.Constraint.init(r.rules(), 1, null);
    var s = try stream();
    defer s.deinit(t.allocator);
    s.grammar = &c;
    var p: win.Plan = .{ .stream = &s, .position = 1, .kind = .forced, .tokens = &.{} };
    try t.expectError(error.MaskFailed, win.constrain(&e, &p));
    try t.expect(c.profile == null);
    try t.expectEqual(timing.Sum{ .windows = 1, .fills = 1, .window_ns = 5, .fill_ns = 5 }, s.grammar_time);
}

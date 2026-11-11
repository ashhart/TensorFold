//! Plain greedy picks share the verify's final buffer; other rounds keep their existing draw submission.
const std = @import("std");

pub fn run(s: anytype, plain: bool, fused: bool, windows: anytype) !void {
    const r = s.runner();
    var greedy = false;
    var all_greedy = true;
    for (windows) |w| {
        const row_greedy = w.stream.sampling == null and w.masks.len == 0;
        greedy = greedy or row_greedy;
        all_greedy = all_greedy and row_greedy;
    }
    const picks_tail = plain and all_greedy and r.reduce == null;
    std.debug.assert(r.tail == null and !r.picks_tail);
    if (picks_tail or fused) r.tail = s.tail(picks_tail);
    r.picks_tail = picks_tail;
    defer {
        r.tail = null;
        r.picks_tail = false;
    }
    try s.verify(plain);
    if (fused) try s.prefetch() else if (greedy and !picks_tail) try s.argmax();
    try s.read();
}

/// A completed GPU pick keeps its non-finite refusal and lower-id tie rule at the host boundary.
pub fn picked(picks: anytype, out: []u32) !void {
    for (out, picks) |*t, p| {
        if (p.token == std.math.maxInt(u32) or p.nonfinite != 0) return error.NonfiniteLogits;
        t.* = p.token;
    }
}

const Recording = struct {
    const Tail = struct { ptr: *anyopaque, encode: *const fn (*anyopaque) anyerror!void };
    const Runner = struct { tail: ?Tail = null, picks_tail: bool = false, reduce: ?u8 = null };
    const Stream = struct { sampling: ?u8 = null };
    const Window = struct { stream: *const Stream, masks: []const u32 = &.{} };
    const Pick = struct { token: u32 = 7, nonfinite: u32 = 0 };
    r: Runner = .{},
    commits: usize = 0,
    waits: usize = 0,
    after_verify: usize = 0,
    host_draws: usize = 0,
    prefetched: bool = false,
    pick: Pick = .{},
    output: [1]u32 = .{0},
    fail_verify: bool = false,
    mixed: bool = false,
    pub fn runner(s: *Recording) *Runner {
        return &s.r;
    }
    pub fn tail(s: *Recording, picks: bool) Tail {
        return .{ .ptr = s, .encode = if (picks) encodePicks else encodeKeep };
    }
    fn encodePicks(ptr: *anyopaque) !void {
        const s: *Recording = @ptrCast(@alignCast(ptr));
        try std.testing.expect(s.r.picks_tail);
    }
    fn encodeKeep(ptr: *anyopaque) !void {
        const s: *Recording = @ptrCast(@alignCast(ptr));
        try std.testing.expect(!s.r.picks_tail);
    }
    pub fn verify(s: *Recording, _: bool) !void {
        if (s.r.tail) |t| try t.encode(t.ptr);
        if (s.fail_verify) return error.VerifyGpuFailure;
        s.commits += 16;
        s.waits += 1;
    }
    pub fn prefetch(s: *Recording) !void {
        s.prefetched = true;
    }
    pub fn argmax(s: *Recording) !void {
        s.after_verify += 1;
        s.commits += 1;
        s.waits += 1;
    }
    pub fn read(s: *Recording) !void {
        if (s.mixed) s.host_draws += 1;
        try picked(&[_]Pick{s.pick}, &s.output);
    }
};

test "qwen27 picks: plain greedy adds no command buffer or wait after the verify" {
    const stream: Recording.Stream = .{};
    const windows = [_]Recording.Window{.{ .stream = &stream }};
    var r: Recording = .{};
    try run(&r, true, false, &windows);
    try std.testing.expectEqual(@as(usize, 16), r.commits);
    try std.testing.expectEqual(@as(usize, 1), r.waits);
    try std.testing.expectEqual(@as(usize, 0), r.after_verify);
    try std.testing.expectEqual(@as(u32, 7), r.output[0]);
    try std.testing.expect(r.r.tail == null and !r.r.picks_tail);
}

test "qwen27 picks: nonfinite picks refuse and every error clears the verify tail" {
    const stream: Recording.Stream = .{};
    const windows = [_]Recording.Window{.{ .stream = &stream }};
    for ([_]Recording.Pick{ .{ .nonfinite = 1 }, .{ .token = std.math.maxInt(u32) } }) |p| {
        var r: Recording = .{ .pick = p };
        try std.testing.expectError(error.NonfiniteLogits, run(&r, true, false, &windows));
        try std.testing.expect(r.r.tail == null and !r.r.picks_tail);
        try std.testing.expectEqual(@as(usize, 0), r.after_verify);
    }
    var r: Recording = .{ .fail_verify = true };
    try std.testing.expectError(error.VerifyGpuFailure, run(&r, true, false, &windows));
    try std.testing.expect(r.r.tail == null and !r.r.picks_tail);
}

test "qwen27 picks: mixed sampled masked and ranked rounds retain the separate draw" {
    const greedy: Recording.Stream = .{};
    const sampled: Recording.Stream = .{ .sampling = 1 };
    const mixed = [_]Recording.Window{ .{ .stream = &greedy }, .{ .stream = &sampled } };
    var r: Recording = .{ .mixed = true };
    try run(&r, true, false, &mixed);
    try std.testing.expectEqual(@as(usize, 1), r.after_verify);
    try std.testing.expectEqual(@as(usize, 1), r.host_draws);
    try std.testing.expect(r.r.tail == null and !r.r.picks_tail);
    const masked = [_]Recording.Window{.{ .stream = &greedy, .masks = &.{1} }};
    var mask: Recording = .{ .mixed = true };
    try run(&mask, true, false, &masked);
    try std.testing.expectEqual(@as(usize, 0), mask.after_verify);
    try std.testing.expectEqual(@as(usize, 1), mask.host_draws);
    const windows = [_]Recording.Window{.{ .stream = &greedy }};
    var rank: Recording = .{ .r = .{ .reduce = 1 } };
    try run(&rank, true, false, &windows);
    try std.testing.expectEqual(@as(usize, 1), rank.after_verify);
    var drafted: Recording = .{};
    try run(&drafted, false, true, &windows);
    try std.testing.expect(drafted.prefetched);
    try std.testing.expectEqual(@as(usize, 0), drafted.after_verify);
    try std.testing.expect(drafted.r.tail == null and !drafted.r.picks_tail);
}

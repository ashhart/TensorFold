//! Failure ownership, rank verdict counts and bounded-cap regressions for the graph cache.

const std = @import("std");
const cache = @import("graph_cache.zig");
const Stream = @import("stream.zig").Stream;
const no_stream: Stream = .{ .d = undefined, .handle = null };
const expect = std.testing.expect;
const equal = std.testing.expectEqual;

const Engine = struct {
    captures: usize = 0,
    live: usize = 0,
    launches: usize = 0,
    bodies: usize = 0,
    handles: [32]u8 = undefined,

    fn engine(e: *Engine) cache.Engine {
        return .{ .ctx = e, .capture = capture, .launch = launch, .free = free };
    }
    fn capture(p: *anyopaque, _: Stream, _: cache.Body) !*anyopaque {
        const e: *Engine = @ptrCast(@alignCast(p));
        e.captures += 1;
        e.live += 1;
        return &e.handles[e.captures % e.handles.len];
    }
    fn launch(p: *anyopaque, _: *anyopaque, _: Stream) !void {
        const e: *Engine = @ptrCast(@alignCast(p));
        e.launches += 1;
    }
    fn free(p: *anyopaque, _: *anyopaque) void {
        const e: *Engine = @ptrCast(@alignCast(p));
        std.debug.assert(e.live > 0);
        e.live -= 1;
    }
    fn body(e: *Engine) cache.Body {
        return .{ .ctx = e, .run = run };
    }
    fn run(p: *anyopaque, _: Stream) !void {
        const e: *Engine = @ptrCast(@alignCast(p));
        e.bodies += 1;
    }
};

const Peer = struct {
    verdicts: []const bool,
    at: usize = 0,
    throw_at: ?usize = null,
    fn agree(p: *Peer) cache.Agree {
        return .{ .ctx = p, .all = all };
    }
    fn all(ctx: *anyopaque, ok: bool) !bool {
        const p: *Peer = @ptrCast(@alignCast(ctx));
        const at = p.at;
        p.at += 1;
        if (p.throw_at == at) return error.CollectiveFailed;
        if (at >= p.verdicts.len) return error.UnexpectedCollective;
        return ok and p.verdicts[at];
    }
};

test "graph cache regression: capture agreement throw frees the not-yet-inserted executable" {
    var e: Engine = .{};
    // per miss: room, then the capture's verdict
    var p: Peer = .{ .verdicts = &.{ true, true }, .throw_at = 1 };
    var c = cache.Cache(u32).init(std.testing.allocator, e.engine(), .{});
    defer c.deinit();
    c.agree = p.agree();
    try std.testing.expectError(error.CollectiveFailed, c.run(1, no_stream, 7, e.body()));
    try equal(@as(usize, 1), e.captures);
    try equal(@as(usize, 0), e.live);
    try equal(@as(usize, 0), c.count());
    try equal(@as(usize, 2), p.at);
}

test "graph cache regression: local map OOM declines before any capture and can recover" {
    var e: Engine = .{};
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    var c = cache.Cache(u32).init(failing.allocator(), e.engine(), .{});
    defer c.deinit();
    try equal(cache.Outcome.eager, try c.run(1, no_stream, 7, e.body()));
    try expect(failing.has_induced_failure);
    try equal(@as(usize, 0), e.captures);
    try equal(@as(usize, 0), e.live);
    try expect(!c.failed);
    failing.fail_index = std.math.maxInt(usize);
    try equal(cache.Outcome.captured, try c.run(1, no_stream, 7, e.body()));
    try equal(@as(usize, 1), e.live);
}

test "graph cache regression: peer map OOM makes a locally prepared rank eager before capture" {
    var e: Engine = .{};
    var p: Peer = .{ .verdicts = &.{false} };
    var c = cache.Cache(u32).init(std.testing.allocator, e.engine(), .{});
    defer c.deinit();
    c.agree = p.agree();
    try equal(cache.Outcome.eager, try c.run(1, no_stream, 7, e.body()));
    try equal(@as(usize, 1), p.at);
    try equal(@as(usize, 0), e.captures);
    try equal(@as(usize, 1), e.bodies);
}

test "graph cache regression: local OOM still contributes the room verdict" {
    var e: Engine = .{};
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    var p: Peer = .{ .verdicts = &.{true} };
    var c = cache.Cache(u32).init(failing.allocator(), e.engine(), .{});
    defer c.deinit();
    c.agree = p.agree();
    try equal(cache.Outcome.eager, try c.run(1, no_stream, 7, e.body()));
    try equal(@as(usize, 1), p.at);
    try equal(@as(usize, 0), e.captures);
}

test "graph cache regression: hits make no collective; a fingerprint change shared by every rank recaptures" {
    var e: Engine = .{};
    var p: Peer = .{ .verdicts = &.{ true, true, true, true, true, true } };
    var c = cache.Cache(u32).init(std.testing.allocator, e.engine(), .{});
    defer c.deinit();
    c.agree = p.agree();
    _ = try c.run(1, no_stream, 7, e.body());
    _ = try c.run(2, no_stream, 7, e.body());
    try equal(@as(usize, 4), p.at);
    for (0..4) |_| try equal(cache.Outcome.replayed, try c.run(1, no_stream, 7, e.body()));
    try equal(@as(usize, 4), p.at);
    // a new layout generation on every rank: both graphs go, the next miss agrees twice as any other
    try equal(cache.Outcome.captured, try c.run(1, no_stream, 8, e.body()));
    try equal(@as(usize, 6), p.at);
    try equal(@as(u64, 2), c.stats.dropped);
    try equal(@as(usize, 1), c.count());
    try equal(@as(usize, 1), e.live);
}

var room = true;
fn available() bool {
    return room;
}

test "graph cache regression: zero maximum remains eager across pressure recovery, with no collective" {
    var e: Engine = .{};
    var p: Peer = .{ .verdicts = &.{} };
    var c = cache.Cache(u32).init(std.testing.allocator, e.engine(), .{ .max = 0 });
    defer c.deinit();
    c.room = available;
    c.agree = p.agree();
    for ([_]bool{ false, true, false, true }) |ok| {
        room = ok;
        try equal(cache.Outcome.eager, try c.run(1, no_stream, 7, e.body()));
        try equal(@as(u32, 0), c.cap);
        try equal(@as(usize, 0), c.count());
    }
    try equal(@as(usize, 0), e.captures);
    try equal(@as(usize, 0), p.at);
}

test "graph cache regression: default floor cannot enlarge a two-entry maximum" {
    var e: Engine = .{};
    var c = cache.Cache(u32).init(std.testing.allocator, e.engine(), .{ .max = 2 });
    defer c.deinit();
    c.room = available;
    try equal(@as(u32, 2), c.settings.min_keep);
    room = false;
    _ = try c.run(1, no_stream, 7, e.body());
    room = true;
    for (0..8) |key| {
        _ = try c.run(@intCast(key), no_stream, 7, e.body());
        try expect(c.count() <= 2 and c.cap <= 2 and e.live <= 2);
    }
}

test "graph cache regression: zero keep floor recovers after reducing the cap to zero" {
    var e: Engine = .{};
    var c = cache.Cache(u32).init(std.testing.allocator, e.engine(), .{ .max = 2, .min_keep = 0 });
    defer c.deinit();
    c.room = available;
    room = false;
    try equal(cache.Outcome.eager, try c.run(1, no_stream, 7, e.body()));
    try equal(@as(u32, 0), c.cap);
    room = true;
    try equal(cache.Outcome.captured, try c.run(1, no_stream, 7, e.body()));
    try equal(@as(u32, 2), c.cap);
}

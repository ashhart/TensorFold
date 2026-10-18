//! Two in-process ranks with fake engines: equal effective settings vote on misses only; a mismatch is a stall.

const std = @import("std");
const cache = @import("graph_cache.zig");
const Stream = @import("stream.zig").Stream;
const no_stream: Stream = .{ .d = undefined, .handle = null };
const equal = std.testing.expectEqual;
const Settings = cache.Settings;

const Engine = struct {
    captures: usize = 0,
    live: usize = 0,
    handles: [8]u8 = undefined,

    fn engine(e: *Engine) cache.Engine {
        return .{ .ctx = e, .capture = capture, .launch = launch, .free = free };
    }
    fn capture(p: *anyopaque, _: Stream, _: cache.Body) !*anyopaque {
        const e: *Engine = @ptrCast(@alignCast(p));
        e.captures += 1;
        e.live += 1;
        return &e.handles[e.captures % e.handles.len];
    }
    fn launch(_: *anyopaque, _: *anyopaque, _: Stream) !void {}
    fn free(p: *anyopaque, _: *anyopaque) void {
        const e: *Engine = @ptrCast(@alignCast(p));
        e.live -= 1;
    }
    fn body(e: *Engine) cache.Body {
        return .{ .ctx = e, .run = run };
    }
    fn run(_: *anyopaque, _: Stream) !void {}
};

/// The two ranks' votes in call order; a rank waits at most `wait_ns` for the other's vote of the same round.
const Board = struct {
    votes: [2][16]bool = undefined,
    cast: [2]std.atomic.Value(u32) = .{ .init(0), .init(0) },
    wait_ns: u64,

    fn vote(b: *Board, rank: u1, ok: bool) !bool {
        const r = b.cast[rank].load(.monotonic);
        if (r == b.votes[rank].len) return error.TooManyVotes;
        b.votes[rank][r] = ok;
        b.cast[rank].store(r + 1, .release);
        const other = 1 - rank;
        const deadline = now() + b.wait_ns;
        while (b.cast[other].load(.acquire) <= r) {
            if (now() > deadline) return error.PeerNeverVoted;
            std.Thread.yield() catch {};
        }
        return ok and b.votes[other][r];
    }

    fn now() u64 {
        var ts: std.c.timespec = undefined;
        _ = std.c.clock_gettime(.MONOTONIC, &ts);
        return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
    }
};

const Rank = struct {
    board: *Board,
    id: u1,
    engine: Engine = .{},
    outcomes: [8]cache.Outcome = undefined,
    err: ?anyerror = null,

    fn all(ctx: *anyopaque, ok: bool) anyerror!bool {
        const r: *Rank = @ptrCast(@alignCast(ctx));
        return r.board.vote(r.id, ok);
    }

    /// One rank's steps over `keys` with `settings` exactly as its process resolved them.
    fn steps(r: *Rank, settings: Settings, ks: []const u32) void {
        var c = cache.Cache(u32).init(std.testing.allocator, r.engine.engine(), settings);
        defer c.deinit();
        c.agree = .{ .ctx = r, .all = all };
        for (ks, 0..) |k, i| r.outcomes[i] = c.run(k, no_stream, 7, r.engine.body()) catch |e| {
            r.err = e;
            return;
        };
    }
};

const keys = [_]u32{ 1, 2, 1, 2, 1 };

test "graph cache ranks: equal enabled settings capture together and vote on each miss, never on a hit" {
    var board: Board = .{ .wait_ns = 5 * std.time.ns_per_s };
    var ranks = [2]Rank{ .{ .board = &board, .id = 0 }, .{ .board = &board, .id = 1 } };
    const s = (Settings{ .max = 4 }).effective(null);
    const t = try std.Thread.spawn(.{}, Rank.steps, .{ &ranks[1], s, &keys });
    ranks[0].steps(s, &keys);
    t.join();
    const want = [_]cache.Outcome{ .captured, .captured, .replayed, .replayed, .replayed };
    for (&ranks) |*r| {
        try equal(@as(?anyerror, null), r.err);
        try std.testing.expectEqualSlices(cache.Outcome, &want, r.outcomes[0..keys.len]);
        try equal(@as(u32, 4), board.cast[r.id].load(.acquire)); // two misses: room and capture each, hits none
        try equal(@as(usize, 0), r.engine.live);
    }
}

test "graph cache ranks: TF_GRAPH_CACHE 0, off and OFF are one effective setting, and an off cache never votes" {
    const d: Settings = .{};
    const off = d.effective("0").digest();
    try std.testing.expectEqualSlices(u8, &off, &d.effective("off").digest());
    try std.testing.expectEqualSlices(u8, &off, &d.effective("OFF").digest());
    try std.testing.expectEqualSlices(u8, &off, &(Settings{ .max = 0, .hold = true }).digest());
    try std.testing.expect(!std.mem.eql(u8, &off, &d.effective(null).digest()));
    const held: Settings = .{ .hold = true, .min_keep = 2 };
    try std.testing.expectEqualSlices(u8, &held.digest(), &(Settings{ .hold = true, .min_keep = 30 }).digest());
    try std.testing.expectEqualSlices(u8, &d.effective("1").digest(), &d.effective(null).digest());
    var board: Board = .{ .wait_ns = 0 };
    var ranks = [2]Rank{ .{ .board = &board, .id = 0 }, .{ .board = &board, .id = 1 } };
    ranks[0].steps(d.effective("0"), &keys);
    ranks[1].steps(d.effective("OFF"), &keys);
    for (&ranks) |*r| {
        try equal(@as(?anyerror, null), r.err);
        for (r.outcomes[0..keys.len]) |o| try equal(cache.Outcome.eager, o);
        try equal(@as(u32, 0), board.cast[r.id].load(.acquire));
        try equal(@as(usize, 0), r.engine.captures);
    }
}

test "graph cache ranks: one rank switched off leaves the other waiting at its first miss; digests show it first" {
    const d: Settings = .{};
    try std.testing.expect(!std.mem.eql(u8, &d.effective(null).digest(), &d.effective("off").digest()));
    var board: Board = .{ .wait_ns = 50 * std.time.ns_per_ms };
    var ranks = [2]Rank{ .{ .board = &board, .id = 0 }, .{ .board = &board, .id = 1 } };
    ranks[1].steps(d.effective("off"), &keys);
    try equal(@as(?anyerror, null), ranks[1].err);
    try equal(@as(u32, 0), board.cast[1].load(.acquire)); // the off rank ran every step eagerly without a vote
    ranks[0].steps(d.effective(null), &keys);
    try equal(@as(?anyerror, error.PeerNeverVoted), ranks[0].err); // in production this wait has no deadline
    try equal(@as(u32, 1), board.cast[0].load(.acquire));
    try equal(@as(usize, 0), ranks[0].engine.live);
}

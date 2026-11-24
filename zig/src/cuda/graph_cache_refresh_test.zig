//! graph_cache in refresh mode on a fake engine: every hit records again and votes; a failed record goes eager.
const std = @import("std");
const t = std.testing;
const gc = @import("graph_cache.zig");
const Stream = @import("stream.zig").Stream;

/// A fake driver: captures and refreshes count (and may fail), launches name the executable they ran.
const Fake = struct {
    captures: u32 = 0,
    refreshes: u32 = 0,
    launched: ?*anyopaque = null,
    launches: u32 = 0,
    live: u32 = 0,
    fail_refresh: bool = false,
    reshape: bool = false, // the next refresh's launches have another shape: a new executable replaces the old
    slots: [8]u8 = undefined,

    fn engine(f: *Fake) gc.Engine {
        return .{ .ctx = f, .capture = cap, .launch = lau, .free = fre, .refresh = ref };
    }
    fn cap(ctx: *anyopaque, _: Stream, body: gc.Body) anyerror!*anyopaque {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        try body.run(body.ctx, undefined); // a capture records the body: its host side runs, nothing launches
        f.captures += 1;
        f.live += 1;
        return &f.slots[f.live % f.slots.len];
    }
    fn ref(ctx: *anyopaque, exec: *anyopaque, _: Stream, body: gc.Body) anyerror!*anyopaque {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        if (f.fail_refresh) return error.CaptureFailed;
        try body.run(body.ctx, undefined);
        f.refreshes += 1;
        if (!f.reshape) return exec;
        f.reshape = false;
        fre(ctx, exec);
        f.live += 1;
        return &f.slots[(f.live + 3) % f.slots.len];
    }
    fn lau(ctx: *anyopaque, exec: *anyopaque, _: Stream) anyerror!void {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        f.launched = exec;
        f.launches += 1;
    }
    fn fre(ctx: *anyopaque, _: *anyopaque) void {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        f.live -= 1;
    }
};

/// The body's host side: how many times it ran, recorded or eager.
const Body = struct {
    runs: u32 = 0,
    fn run(ctx: *anyopaque, _: Stream) anyerror!void {
        const b: *Body = @ptrCast(@alignCast(ctx));
        b.runs += 1;
    }
    fn body(b: *Body) gc.Body {
        return .{ .ctx = b, .run = run };
    }
};

/// The other ranks' vote: true unless told otherwise; counts the votes taken.
const Votes = struct {
    others: bool = true,
    taken: u32 = 0,
    fn all(ctx: *anyopaque, ok: bool) anyerror!bool {
        const v: *Votes = @ptrCast(@alignCast(ctx));
        v.taken += 1;
        return ok and v.others;
    }
};

test "graph cache refresh: each hit records the body once, votes and launches what the refresh returned" {
    var f: Fake = .{};
    var votes: Votes = .{};
    var c = gc.Cache(u32).init(t.allocator, f.engine(), .{ .on = true, .max = 4 });
    defer c.deinit();
    c.agree = .{ .ctx = &votes, .all = Votes.all };
    var b: Body = .{};
    const s: Stream = undefined;
    // a miss captures (the body's host side runs once), then each hit refreshes (once each) and replays
    try t.expectEqual(gc.Outcome.captured, try c.run(7, s, 1, b.body()));
    for (0..3) |_| try t.expectEqual(gc.Outcome.replayed, try c.run(7, s, 1, b.body()));
    try t.expectEqual([4]u32{ 1, 3, 4, 4 }, [4]u32{ f.captures, f.refreshes, f.launches, b.runs });
    try t.expectEqual(@as(u32, 1 + 1 + 3), votes.taken); // the miss's room and capture, then every hit
    // launches of another shape: the refresh's new executable is the one launched and held
    f.reshape = true;
    _ = try c.run(7, s, 1, b.body());
    const swapped = f.launched;
    _ = try c.run(7, s, 1, b.body());
    try t.expectEqual(swapped, f.launched);
    try t.expectEqual(@as(u32, 1), f.live);
}

test "graph cache refresh: a failed record on any rank sends every rank eager, the graphs freed, for good" {
    for ([_]bool{ true, false }) |here| {
        var f: Fake = .{};
        var votes: Votes = .{};
        var c = gc.Cache(u32).init(t.allocator, f.engine(), .{ .on = true, .max = 4 });
        defer c.deinit();
        c.agree = .{ .ctx = &votes, .all = Votes.all };
        var b: Body = .{};
        const s: Stream = undefined;
        _ = try c.run(3, s, 1, b.body());
        if (here) f.fail_refresh = true else votes.others = false; // this rank's record fails, or another rank's
        try t.expectEqual(gc.Outcome.eager, try c.run(3, s, 1, b.body()));
        try t.expectEqual(@as(u32, 0), f.live);
        f.fail_refresh = false;
        votes.others = true;
        const taken = votes.taken;
        try t.expectEqual(gc.Outcome.eager, try c.run(3, s, 1, b.body()));
        try t.expectEqual(taken, votes.taken); // an off cache never votes again
        try t.expectEqual(@as(u32, 1), f.launches); // only the capture's launch ever ran a graph
    }
}

test "graph cache refresh: an engine without refresh replays as captured, as before" {
    var f: Fake = .{};
    var e = f.engine();
    e.refresh = null;
    var c = gc.Cache(u32).init(t.allocator, e, .{ .on = true, .max = 4 });
    defer c.deinit();
    var b: Body = .{};
    const s: Stream = undefined;
    _ = try c.run(1, s, 1, b.body());
    for (0..3) |_| try t.expectEqual(gc.Outcome.replayed, try c.run(1, s, 1, b.body()));
    try t.expectEqual([3]u32{ 1, 0, 1 }, [3]u32{ f.captures, f.refreshes, b.runs });
}

/// Launches as (kernel, argument) words: a body writes them where `sink` points, the log or a recording.
const Launches = struct {
    var sink: *std.ArrayList(u64) = undefined;
    var log: std.ArrayList(u64) = .empty;
    var recorded: [4]std.ArrayList(u64) = @splat(.empty);
    var used: usize = 0;

    /// A capture or refresh: the body writes into the executable's recording, which a launch then plays.
    fn engine() gc.Engine {
        return .{ .ctx = undefined, .capture = cap, .launch = lau, .free = fre, .refresh = ref };
    }
    fn record(r: *std.ArrayList(u64), body: gc.Body) !void {
        r.clearRetainingCapacity();
        const was = sink;
        sink = r;
        defer sink = was;
        try body.run(body.ctx, undefined);
    }
    fn cap(_: *anyopaque, _: Stream, body: gc.Body) anyerror!*anyopaque {
        const r = &recorded[used];
        used += 1;
        try record(r, body);
        return r;
    }
    fn ref(_: *anyopaque, exec: *anyopaque, _: Stream, body: gc.Body) anyerror!*anyopaque {
        try record(@ptrCast(@alignCast(exec)), body);
        return exec;
    }
    fn lau(_: *anyopaque, exec: *anyopaque, _: Stream) anyerror!void {
        const r: *std.ArrayList(u64) = @ptrCast(@alignCast(exec));
        try log.appendSlice(t.allocator, r.items);
    }
    fn fre(_: *anyopaque, _: *anyopaque) void {}
};

/// A window's launches: three kernels, the second taking this step's position.
const Window = struct {
    pos: u64,
    fn run(ctx: *anyopaque, _: Stream) anyerror!void {
        const w: *Window = @ptrCast(@alignCast(ctx));
        for ([_]u64{ 1, 2, 3 }, [_]u64{ 0, w.pos, 0 }) |k, arg| try Launches.sink.appendSlice(t.allocator, &.{ k, arg });
    }
};

test "graph cache refresh: eager and graphed steps launch the same kernels with each step's own arguments" {
    defer {
        Launches.log.deinit(t.allocator);
        for (&Launches.recorded) |*r| r.deinit(t.allocator);
    }
    var logs: [2][]u64 = undefined;
    for ([_]bool{ false, true }, 0..) |graphed, i| {
        Launches.log = .empty;
        Launches.sink = &Launches.log;
        Launches.used = 0;
        var c = gc.Cache(u32).init(t.allocator, Launches.engine(), .{ .on = graphed, .max = 4 });
        defer c.deinit();
        for (0..6) |step| {
            var w: Window = .{ .pos = 100 + step };
            _ = try c.run(@intCast(step % 2), undefined, 1, .{ .ctx = &w, .run = Window.run });
        }
        logs[i] = try t.allocator.dupe(u64, Launches.log.items);
        Launches.log.deinit(t.allocator);
        Launches.log = .empty;
    }
    defer for (logs) |l| t.allocator.free(l);
    try t.expectEqualSlices(u64, logs[0], logs[1]);
    try t.expectEqual(@as(usize, 6 * 6), logs[1].len);
    try t.expectEqual(@as(u64, 105), logs[1][logs[1].len - 3]); // the last step's own position, not a captured one
}

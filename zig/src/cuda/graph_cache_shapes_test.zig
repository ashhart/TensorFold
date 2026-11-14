//! A captured shape replayed after a predecessor of another shape writes what its eager step writes (#2138).

const std = @import("std");
const cache = @import("graph_cache.zig");
const Stream = @import("stream.zig").Stream;
const no_stream: Stream = .{ .d = undefined, .handle = null };
const expect = std.testing.expect;
const equal = std.testing.expectEqual;

/// A window's captured geometry: its rows and its context bucket, as a family's graph key.
const Shape = struct { rows: u8, ctx: u8 };

const shapes = [_]Shape{ .{ .rows = 1, .ctx = 0 }, .{ .rows = 4, .ctx = 0 }, .{ .rows = 8, .ctx = 0 }, .{ .rows = 1, .ctx = 1 }, .{ .rows = 4, .ctx = 1 }, .{ .rows = 16, .ctx = 1 }, .{ .rows = 2, .ctx = 2 }, .{ .rows = 3, .ctx = 2 }, .{ .rows = 5, .ctx = 3 }, .{ .rows = 7, .ctx = 3 }, .{ .rows = 12, .ctx = 4 }, .{ .rows = 16, .ctx = 4 } };

/// What row `r` of a step of `s` writes under fingerprint `fp` (its buffers' generation).
fn word(s: Shape, r: usize, fp: u64) u64 {
    return std.hash.Wyhash.hash(fp, &.{ s.rows, s.ctx, @as(u8, @intCast(r)) });
}

/// A device whose step writes one word a row; a capture keeps the launches as made, args by value, and replays them.
const Device = struct {
    out: [16]u64 = @splat(0),
    recording: ?*Graph = null,
    graphs: [64]Graph = undefined,
    made: usize = 0,
    live: usize = 0,
    eager_steps: usize = 0,
    replays: usize = 0,

    const Graph = struct { shape: Shape, fp: u64, rows: u8 = 0, words: [16]u64 = undefined, live: bool = false };

    fn emit(d: *Device, r: usize, w: u64) void {
        if (d.recording) |g| {
            g.words[r] = w;
            g.rows = @intCast(r + 1);
        } else d.out[r] = w;
    }

    fn engine(d: *Device) cache.Engine {
        return .{ .ctx = d, .capture = capture, .launch = launch, .free = free };
    }

    fn capture(ctx: *anyopaque, stream: Stream, body: cache.Body) anyerror!*anyopaque {
        const d: *Device = @ptrCast(@alignCast(ctx));
        const step: *Step = @ptrCast(@alignCast(body.ctx));
        const g = &d.graphs[d.made % d.graphs.len];
        try expect(!g.live);
        g.* = .{ .shape = step.shape, .fp = step.fp, .live = true };
        d.made += 1;
        d.live += 1;
        d.recording = g;
        defer d.recording = null;
        try body.run(body.ctx, stream); // captured: nothing runs, the launches are kept
        return g;
    }

    fn launch(ctx: *anyopaque, exec: *anyopaque, _: Stream) anyerror!void {
        const d: *Device = @ptrCast(@alignCast(ctx));
        const g: *Graph = @ptrCast(@alignCast(exec));
        try expect(g.live);
        for (g.words[0..g.rows], 0..) |w, r| d.out[r] = w;
        d.replays += 1;
    }

    fn free(ctx: *anyopaque, exec: *anyopaque) void {
        const d: *Device = @ptrCast(@alignCast(ctx));
        const g: *Graph = @ptrCast(@alignCast(exec));
        std.debug.assert(g.live);
        g.live = false;
        d.live -= 1;
    }
};

/// One step's body: the shape and fingerprint the host set before it, its launches through `dev`.
const Step = struct {
    dev: *Device,
    shape: Shape = undefined,
    fp: u64 = 0,

    fn body(s: *Step) cache.Body {
        return .{ .ctx = s, .run = run };
    }

    fn run(ctx: *anyopaque, _: Stream) anyerror!void {
        const s: *Step = @ptrCast(@alignCast(ctx));
        if (s.dev.recording == null) s.dev.eager_steps += 1;
        for (0..s.shape.rows) |r| s.dev.emit(r, word(s.shape, r, s.fp));
    }
};

var room_now = true;
fn roomNow() bool {
    return room_now;
}

const Stats = @FieldType(cache.Cache(Shape), "stats");

/// `steps` steps over `shapes` in a seeded order; each step's rows equal its eager step's, whatever ran before it.
fn replayAfterOthers(settings: cache.Settings, seed: u64, steps: usize, pressure: bool) !Stats {
    var dev: Device = .{};
    var step: Step = .{ .dev = &dev };
    var c = cache.Cache(Shape).init(std.testing.allocator, dev.engine(), settings);
    defer c.deinit();
    c.room = roomNow;
    room_now = true;
    defer room_now = true;
    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    var fp: u64 = 7;
    var prev: ?Shape = null;
    var switched: usize = 0;
    for (0..steps) |_| {
        if (rnd.uintLessThan(u32, 97) == 0) fp += 1; // buffers moved: every graph goes
        if (pressure) room_now = rnd.uintLessThan(u32, 9) != 0;
        const s = shapes[rnd.uintLessThan(usize, shapes.len)];
        step.shape = s;
        step.fp = fp;
        dev.out = @splat(0xdead); // a predecessor's rows are no help
        const eager_before = dev.eager_steps;
        const replays_before = dev.replays;
        const outcome = try c.run(s, no_stream, fp, step.body());
        for (0..s.rows) |r| try equal(word(s, r, fp), dev.out[r]);
        switch (outcome) {
            .replayed, .captured => try equal(eager_before, dev.eager_steps),
            .eager => try equal(replays_before, dev.replays),
        }
        try equal(c.count(), dev.live);
        if (prev) |p| switched += @intFromBool(!std.meta.eql(p, s));
        prev = s;
    }
    try expect(switched > steps / 2);
    const stats = c.stats;
    c.dropAll();
    try equal(@as(usize, 0), dev.live);
    return stats;
}

test "graph cache shapes: a replay after a predecessor of another shape writes its own shape's eager rows" {
    const st = try replayAfterOthers(.{ .max = 48 }, 2138, 4000, false);
    try expect(st.replayed > 3000 and st.captured >= shapes.len and st.eager == 0);
    try expect(st.dropped > 0); // fingerprint changes recaptured every shape
}

test "graph cache shapes: under a cap smaller than the shapes, evicted keys recapture and never replay another's graph" {
    for ([_]u32{ 1, 2, 5 }) |max| {
        const st = try replayAfterOthers(.{ .max = max, .min_keep = 1 }, 2138 + max, 4000, false);
        try expect(st.evicted > 100 and st.replayed > 0);
    }
}

test "graph cache shapes: under the memory floor, shed or held, eager and replayed steps write the same rows" {
    for ([_]bool{ false, true }) |hold| {
        const st = try replayAfterOthers(.{ .max = 6, .min_keep = 2, .hold = hold }, 2138 + @as(u64, @intFromBool(hold)), 4000, true);
        try expect(st.eager > 100 and st.replayed > 100 and st.floor_hits > 100);
    }
}

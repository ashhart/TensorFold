//! A body's graph capture, and a sequence's own graphs (not the engine's captures) on its buffers, replayed after.

const std = @import("std");
const platform = @import("sys");
const cuda = @import("cuda");
const engine = @import("cuda_engine.zig");
const Engine = engine.Engine;

/// One sequence's captured graphs, by the step they run (a window's or a head level's id).
pub const Graphs = std.AutoHashMapUnmanaged(u32, cuda.graph.Exec);

pub fn free(gpa: std.mem.Allocator, m: *Graphs) void {
    var it = m.valueIterator();
    while (it.next()) |x| x.deinit();
    m.deinit(gpa);
    m.* = .{};
}

/// Captures and launches since load, for TF_GRAPH_LOG's line a request.
pub const Stats = struct {
    captures: u64 = 0,
    capture_ns: u64 = 0,
    replays: u64 = 0,
    eager: u64 = 0,

    pub fn since(now: Stats, then: Stats) Stats {
        return .{ .captures = now.captures - then.captures, .capture_ns = now.capture_ns - then.capture_ns, .replays = now.replays - then.replays, .eager = now.eager - then.eager };
    }
};

/// Whether TF_GRAPH_LOG=1 asks for a line a request: its captures, their time, its replays and eager windows.
pub fn logFromEnv() bool {
    const v = platform.c.getenv("TF_GRAPH_LOG") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

/// One graph of `body` on the bound sequence's buffers, instantiated and uploaded.
pub fn record(e: *Engine, body: Engine.Body) !cuda.graph.Exec {
    const t0 = std.Io.Clock.awake.now(e.io);
    try cuda.graph.beginCapture(e.stream, .thread_local);
    recordBody(e, body) catch |err| {
        if (cuda.graph.endCapture(e.stream)) |g| {
            var x = g;
            x.deinit();
        } else |_| {}
        return err;
    };
    var g = try cuda.graph.endCapture(e.stream);
    defer g.deinit();
    const exec = try g.instantiate();
    try exec.upload(e.stream);
    try e.stream.synchronize();
    e.stats.captures += 1;
    e.stats.capture_ns += @intCast(std.Io.Clock.awake.now(e.io).toNanoseconds() - t0.toNanoseconds());
    return exec;
}

fn recordBody(e: *Engine, body: Engine.Body) !void {
    switch (body) {
        .window => |w| {
            const f = e.forward(null);
            try f.windowSegs(&.{.{ .b = f.b, .row0 = 0, .rows = w.rows, .sampled = f.sampled, .mask = w.mask, .slot = f.slot, .untaught = f.untaught }});
            if (w.feed) return e.ops().serialFeed(e.b.sampled, e.b.ids, e.b.meta, e.history_dev);
            try e.ops().download(std.mem.sliceAsBytes(e.pinned.slice(u32)[engine.pin_sampled..][0..w.rows]), e.b.sampled);
        },
        .head => |x| try x.h.level(x.rows, x.j, x.vocab),
    }
}

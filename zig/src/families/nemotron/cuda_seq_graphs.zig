//! Graphs a sequence other than the engine's own captures on its own buffers, once each, and replays after.

const std = @import("std");
const cuda = @import("cuda");

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
    const v = std.c.getenv("TF_GRAPH_LOG") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

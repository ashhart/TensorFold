//! TF_FLASHNEXT_GRAPH_TAIL=1: a lone verify's tail copies and greedy row best inside its window graph (opt-in).
const std = @import("std");
const cuda = @import("cuda");
const mtp = @import("cuda_mtp.zig");

/// The switch; default off. The startup check refused every malformed value already, so a plain default stands here.
pub fn on() bool {
    return @import("sys").environment.read(bool, "TF_FLASHNEXT_GRAPH_TAIL", false) catch false;
}

/// One device copy of `n` bytes (none when n is 0).
pub const Copy = struct { dst: u64 = 0, src: u64 = 0, n: usize = 0 };

/// A verify's work around its pass, in its order: `before`, the pass, `after` (its src the pass's rows), the row best.
pub const Tail = struct {
    before: Copy = .{},
    after: Copy = .{},
    head: ?*const mtp.Head = null,
    scores: u64 = 0,
    vocab: usize = 0,
    rows: usize = 0,
    best: u64 = 0,

    /// The host values a graph of it holds, for the window key: zero where a part is absent.
    pub fn values(t: *const Tail) [8]u64 {
        return .{ t.before.dst, t.before.src, t.before.n, t.after.dst, t.after.n, @intFromBool(t.head != null), t.scores, t.best };
    }

    pub fn first(t: *const Tail, s: cuda.Stream) !void {
        try copy(s, t.before);
    }

    /// `ple_nrow`: the pass's n-gram rows, the after copy's source.
    pub fn last(t: *const Tail, s: cuda.Stream, ple_nrow: u64) !void {
        var after = t.after;
        after.src = ple_nrow;
        try copy(s, after);
        if (t.head) |h| try h.best(s, t.scores, t.vocab, t.vocab, t.rows, t.best);
    }
};

fn copy(s: cuda.Stream, c: Copy) !void {
    if (c.n == 0) return;
    try s.d.check(s.d.api.cuMemcpyDtoDAsync_v2(c.dst, c.src, c.n, s.handle), "cuMemcpyDtoDAsync");
}

test "graph tail: the key values mark each absent part as zero and hold every buffer and size" {
    const none: Tail = .{};
    try std.testing.expectEqual(@as([8]u64, @splat(0)), none.values());
    const t: Tail = .{ .before = .{ .dst = 10, .src = 20, .n = 30 }, .after = .{ .dst = 40, .n = 50 }, .scores = 60, .best = 70 };
    try std.testing.expectEqual([8]u64{ 10, 20, 30, 40, 50, 0, 60, 70 }, t.values());
}

test {
    _ = @import("cuda_graph_tail_test.zig");
}

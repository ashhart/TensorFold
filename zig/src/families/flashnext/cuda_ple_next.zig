//! The next prompt pass's n-gram rows, gathered while the GPU runs this one; taken only for the same history and ids.

const std = @import("std");
const ple = @import("cuda_ple.zig");
const dims = @import("cuda_prompt_wt.zig").dims;

pub const Next = struct {
    thread: ?std.Thread = null,
    /// The history before the first id and the ids the rows are gathered for; rows 0: nothing gathered.
    history: [ple.max_n]i64 = undefined,
    hist_len: usize = 0,
    ids: std.ArrayList(u32) = .empty,
    rows: usize = 0,
    emb: []u16 = &.{},
    xs: []f32 = &.{},
    failed: bool = false,

    /// Joins the gather, then frees the rows.
    pub fn deinit(a: *Next, gpa: std.mem.Allocator) void {
        a.join();
        a.ids.deinit(gpa);
        gpa.free(a.emb);
        gpa.free(a.xs);
        a.* = .{};
    }

    fn join(a: *Next) void {
        if (a.thread) |th| th.join();
        a.thread = null;
    }

    /// Whether the rows gathered (or in flight) are those of `ids` after `history`: same history, ids a prefix.
    pub fn covers(a: *const Next, history: []const i64, ids: []const u32) bool {
        if (a.rows < ids.len or ids.len == 0 or a.hist_len != history.len) return false;
        return std.mem.eql(i64, a.history[0..a.hist_len], history) and std.mem.eql(u32, a.ids.items[0..ids.len], ids);
    }

    /// What `Ngram.rows(history, ids, emb, xs)` writes, from the rows `covers` named; the gather's own on a failure.
    pub fn take(a: *Next, ngram: *const ple.Ngram, history: []i64, ids: []const u32, emb: []u16, xs: []f32) !void {
        a.join();
        a.rows = 0;
        if (a.failed) return ngram.rows(history, ids, emb, xs);
        @memcpy(emb, a.emb[0 .. ids.len * dims]);
        @memcpy(xs, a.xs[0 .. ids.len * (dims / 32)]);
        // the history moved past ids, as the gather moves it
        for (ids) |id| {
            std.mem.copyForwards(i64, history[0 .. history.len - 1], history[1..]);
            history[history.len - 1] = id;
        }
    }

    /// Gathers `next` after `history` on a host thread in place of any earlier rows; an empty `next` keeps them.
    pub fn start(a: *Next, gpa: std.mem.Allocator, ngram: *const ple.Ngram, history: []const i64, next: []const u32) void {
        if (next.len == 0) return;
        a.join();
        a.rows = 0;
        if (history.len > ple.max_n) return;
        a.grow(gpa, next.len) catch return;
        a.ids.clearRetainingCapacity();
        a.ids.appendSlice(gpa, next) catch return;
        @memcpy(a.history[0..history.len], history);
        a.hist_len = history.len;
        a.failed = false;
        a.thread = std.Thread.spawn(.{}, run, .{ a, ngram }) catch return;
        a.rows = next.len;
    }

    fn grow(a: *Next, gpa: std.mem.Allocator, n: usize) !void {
        if (a.emb.len >= n * dims) return;
        gpa.free(a.emb);
        gpa.free(a.xs);
        a.emb = &.{};
        a.xs = &.{};
        a.emb = try gpa.alloc(u16, n * dims);
        a.xs = try gpa.alloc(f32, n * (dims / 32));
    }

    fn run(a: *Next, ngram: *const ple.Ngram) void {
        var hist = a.history;
        const n = a.ids.items.len;
        ngram.rows(hist[0..a.hist_len], a.ids.items, a.emb[0 .. n * dims], a.xs[0 .. n * (dims / 32)]) catch {
            a.failed = true;
        };
    }
};

const t = std.testing;
const lanes_test = @import("cuda_ple_gather_lanes.zig");

test "gather ahead: a covered prefix takes the rows and history the pass's own gather writes, on both formats" {
    const ids = [_]u32{ 7, 12, 3, 5, 40, 1, 9, 2, 7, 12 };
    for ([_]lanes_test.Kind{ .nvfp4, .mlx }) |kind| {
        var tmp = t.tmpDir(.{});
        defer tmp.cleanup();
        var g = try lanes_test.load(&tmp, kind);
        defer g.deinit();
        var a: Next = .{};
        defer a.deinit(t.allocator);
        const start_hist = [2]i64{ 12, 99 };
        for ([_]usize{ ids.len, 6, 1 }) |n| {
            a.start(t.allocator, &g, &start_hist, &ids);
            try t.expect(a.covers(&start_hist, ids[0..n]));
            var want_hist = start_hist;
            var got_hist = start_hist;
            const want = try t.allocator.alloc(u16, n * dims);
            defer t.allocator.free(want);
            const got = try t.allocator.alloc(u16, n * dims);
            defer t.allocator.free(got);
            var want_xs: [ids.len * (dims / 32)]f32 = undefined;
            var got_xs: [ids.len * (dims / 32)]f32 = @splat(-1);
            try g.rows(&want_hist, ids[0..n], want, want_xs[0 .. n * (dims / 32)]);
            try a.take(&g, &got_hist, ids[0..n], got, got_xs[0 .. n * (dims / 32)]);
            try t.expectEqualSlices(u16, want, got);
            try t.expectEqualSlices(u8, std.mem.sliceAsBytes(want_xs[0 .. n * (dims / 32)]), std.mem.sliceAsBytes(got_xs[0 .. n * (dims / 32)]));
            try t.expectEqualSlices(i64, &want_hist, &got_hist);
            // taken once: the next pass gathers its own rows unless another start names them
            try t.expect(!a.covers(&start_hist, ids[0..n]));
        }
    }
}

test "gather ahead: another history, other ids, a longer pass or no start are not covered" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var g = try lanes_test.load(&tmp, .nvfp4);
    defer g.deinit();
    var a: Next = .{};
    defer a.deinit(t.allocator);
    const ids = [_]u32{ 7, 12, 3, 5 };
    const hist = [2]i64{ 99, 99 };
    try t.expect(!a.covers(&hist, &ids));
    a.start(t.allocator, &g, &hist, &ids);
    try t.expect(a.covers(&hist, &ids));
    try t.expect(!a.covers(&[2]i64{ 99, 7 }, &ids));
    try t.expect(!a.covers(&[1]i64{99}, ids[0..1]));
    try t.expect(!a.covers(&hist, &[_]u32{ 7, 12, 4 }));
    try t.expect(!a.covers(&hist, &[_]u32{ 7, 12, 3, 5, 1 }));
    try t.expect(!a.covers(&hist, ids[0..0]));
    // a pass with no next (a verify window between two prompt passes) leaves them for the prompt's next pass
    a.start(t.allocator, &g, &hist, ids[0..0]);
    try t.expect(a.covers(&hist, &ids));
    // another sequence's prompt pass takes the slot: its rows, not the first one's
    a.start(t.allocator, &g, &[2]i64{ 5, 40 }, ids[1..]);
    try t.expect(!a.covers(&hist, &ids));
    try t.expect(a.covers(&[2]i64{ 5, 40 }, ids[1..]));
}

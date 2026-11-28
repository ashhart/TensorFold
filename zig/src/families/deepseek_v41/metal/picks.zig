//! A window's keyed picks, a row a thread: each row's pick is a function of its logits, draw and sampling alone.
const std = @import("std");
const host = @import("dsv41");

const lanes = host.lanes;

/// Rows of a window picked at once (later rows wait for a thread of their own).
pub const rows = 8;

/// Each row's keyed pick (row r drawn at `draws[r]`): rows past the first on threads, each with its own row of `vals`.
pub fn keyed(logits: []const f32, draws: []const u64, s: lanes.Sampling, vals: []f64, ids: []const u64, out: []u32) !void {
    const V = ids.len;
    var at: usize = 0;
    while (at < draws.len) : (at += rows) {
        const n = @min(rows, draws.len - at);
        var jobs: [rows]Job = undefined;
        var threads: [rows]?std.Thread = @splat(null);
        for (jobs[0..n], 0..) |*j, r| j.* = .{ .row = logits[(at + r) * V ..][0..V], .vals = vals[r * V ..][0..V], .ids = ids, .draw = draws[at + r], .s = s };
        // a thread that cannot start leaves its row to this one
        for (1..n) |r| threads[r] = std.Thread.spawn(.{ .stack_size = 1 << 20 }, Job.run, .{&jobs[r]}) catch null;
        for (0..n) |r| if (threads[r] == null) jobs[r].run();
        for (threads[1..n]) |t| if (t) |th| th.join();
        for (jobs[0..n], out[at..][0..n]) |j, *o| o.* = try j.got;
    }
}

const Job = struct {
    row: []const f32,
    vals: []f64,
    ids: []const u64,
    draw: u64,
    s: lanes.Sampling,
    got: anyerror!u32 = error.NotPicked,

    fn run(j: *Job) void {
        var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer arena.deinit();
        for (j.vals, j.row) |*v, x| v.* = x;
        const id = lanes.sampling.choose(arena.allocator(), j.vals, j.ids, j.draw, j.s) catch |e| {
            j.got = e;
            return;
        };
        j.got = @intCast(id);
    }
};

test "keyed picks of a window's rows on threads equal each row picked alone" {
    const a = std.testing.allocator;
    const V = 3001;
    const n = 11;
    var prng = std.Random.DefaultPrng.init(4103);
    const logits = try a.alloc(f32, n * V);
    defer a.free(logits);
    for (logits) |*x| x.* = prng.random().floatNorm(f32) * 3;
    const ids = try a.alloc(u64, V);
    defer a.free(ids);
    for (ids, 0..) |*id, i| id.* = i;
    const vals = try a.alloc(f64, rows * V);
    defer a.free(vals);
    var draws: [n]u64 = undefined;
    for (&draws, 0..) |*d, i| d.* = 900 + i;
    // the default top-k and top-p, the whole vocabulary (one pass), a nucleus without top-k (the sort)
    const cases = [_]lanes.Sampling{ .{ .seed = 7, .temperature = 0.9 }, .{ .seed = 8, .temperature = 1.0, .top_k = 0, .top_p = 1.0 }, .{ .seed = 9, .temperature = 0.7, .top_k = 0, .top_p = 0.8 } };
    for (cases) |s| {
        var got: [n]u32 = undefined;
        try keyed(logits, &draws, s, vals, ids, &got);
        for (0..n) |r| {
            const one = vals[0..V];
            for (one, logits[r * V ..][0..V]) |*v, x| v.* = x;
            const want = try lanes.sampling.choose(a, one, ids, draws[r], s);
            try std.testing.expectEqual(@as(u32, @intCast(want)), got[r]);
        }
    }
}

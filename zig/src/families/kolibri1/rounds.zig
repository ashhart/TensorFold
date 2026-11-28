//! Kolibri-1's decode rounds of several rows as CUDA graphs: one a shape, captured at its first round, then replayed.
const std = @import("std");
const cuda = @import("cuda");
const fwd = @import("forward.zig");
const kern = @import("kernels.zig");
const tree = @import("tree.zig");
const gc = cuda.graph_cache;

/// A round's launches past its staged inputs: its rows, its streams and tree attention's key groups.
pub const Key = struct { rows: u32, streams: u32, groups: u32 };

/// A backend's round graphs; the cache is made at the first round, where the backend's address stays put.
pub const Rounds = struct {
    engine: gc.CudaEngine,
    cache: ?gc.Cache(Key) = null,

    pub fn deinit(r: *Rounds) void {
        if (r.cache) |*c| c.deinit();
    }

    /// Every graph freed: attach replaces the caches and the page table they read.
    pub fn drop(r: *Rounds) void {
        if (r.cache) |*c| c.dropAll();
    }

    /// Every row's fp32 logits at `out` for a decode round: inputs staged as `forward` stages them, then the graph.
    pub fn forward(r: *Rounds, m: *fwd.Model, o: kern.Ops, chains: []const fwd.Chain, out: u64) !void {
        const st = try m.stage(o, chains, false);
        var all: [fwd.round_rows]usize = undefined;
        for (all[0..st.n], 0..) |*row, i| row.* = i;
        var body: Body = .{ .m = m, .o = o, .st = st, .streams = chains.len, .pick = all[0..st.n], .out = out };
        if (r.cache == null) {
            r.cache = gc.Cache(Key).init(m.gpa, r.engine.engine(), .{});
            r.cache.?.name = "kolibri1 rounds";
        }
        const key: Key = .{ .rows = @intCast(st.n), .streams = @intCast(chains.len), .groups = @intCast(st.ng) };
        _ = try r.cache.?.run(key, o.s, fingerprint(m, out), .{ .ctx = &body, .run = Body.run });
    }
};

/// The addresses a round's graph holds: a drafter's tap copies (their rows, their count) among them.
pub fn fingerprint(m: *const fwd.Model, out: u64) u64 {
    return gc.fingerprintOf(&.{ out, m.origin, m.table, m.s.x, m.tap_rows, m.taps.len });
}

/// A round's launches as the eager forward makes them, on the stream the cache runs them on.
const Body = struct {
    m: *fwd.Model,
    o: kern.Ops,
    st: fwd.Model.Staged,
    streams: usize,
    pick: []const usize,
    out: u64,

    fn run(ctx: *anyopaque, s: cuda.Stream) anyerror!void {
        const b: *Body = @ptrCast(@alignCast(ctx));
        var o = b.o;
        o.s = s;
        try b.m.launch(o, b.st.n, null, b.st.t, b.streams, b.st.ng, @min(b.m.max_rows, fwd.round_rows), b.pick, b.out);
    }
};

test "kolibri rounds: a plan's offsets follow its rows and streams alone, which a round's key holds" {
    const gpa = std.testing.allocator;
    const a = try tree.build(gpa, &.{ .{ .rows = 3, .committed = 700 }, .{ .rows = 1, .committed = 5 } });
    defer gpa.free(a.flat);
    const b = try tree.build(gpa, &.{ .{ .rows = 2, .committed = 9 }, .{ .rows = 2, .committed = 4000 } });
    defer gpa.free(b.flat);
    try std.testing.expectEqual(a.streamsAt(), b.streamsAt());
    try std.testing.expectEqual(a.parentsAt(), b.parentsAt());
    const c = try tree.build(gpa, &.{ .{ .rows = 2, .committed = 9 }, .{ .rows = 1, .committed = 9 }, .{ .rows = 1, .committed = 9 } });
    defer gpa.free(c.flat);
    try std.testing.expect(c.parentsAt() != a.parentsAt()); // a third stream moves the parents: the key holds streams
}

test "kolibri rounds: naming a drafter's taps changes the fingerprint, so no round graph without tap copies replays" {
    var m: fwd.Model = undefined;
    m.origin = 0x1000;
    m.table = 0x2000;
    m.s.x = 0x3000;
    m.tap_rows = 0;
    m.taps = &.{};
    const before = fingerprint(&m, 0x4000);
    m.tap_rows = 0x5000;
    m.taps = &.{ 1, 3 };
    try std.testing.expect(fingerprint(&m, 0x4000) != before);
}

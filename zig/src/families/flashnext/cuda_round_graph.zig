//! A single bounded decode graph, recaptured when sequence owners, scratch or attention geometry change.
const std = @import("std");
const cuda = @import("cuda");
const dims = 2560;

pub fn enabled() bool {
    const value = std.c.getenv("TF_FLASHNEXT_DECODE_GRAPH") orelse return false;
    return std.mem.eql(u8, std.mem.span(value), "1");
}

pub const Key = struct {
    count: usize,
    owners: [32]u64 = @splat(0),
    chunks: [32]u32 = @splat(0),
    score_tiles: [32]u32 = @splat(0),
    scratch: u64,
    scores: u64,
    output: u64,

    pub fn init(seqs: anytype, scratch: u64, scores: u64, output: u64) !Key {
        if (seqs.len == 0 or seqs.len > 32) return error.UnexpectedTensor;
        var key: Key = .{ .count = seqs.len, .scratch = scratch, .scores = scores, .output = output };
        for (seqs, 0..) |q, i| {
            if (q.rank_id == 0) return error.UnsupportedRankOperation;
            key.owners[i] = q.rank_id;
            const end = q.fed + 1;
            key.chunks[i] = @intCast((@min(end, 2051) + 511) / 512);
            key.score_tiles[i] = if (end / 4 > 512) @intCast(((end + 3) / 4 + 63) / 64) else 0;
        }
        return key;
    }
};

pub const Cache = struct {
    exec: ?cuda.graph.Exec = null,
    key: ?Key = null,

    pub fn clear(c: *Cache) void {
        if (c.exec) |*e| e.deinit();
        c.* = .{};
    }

    /// CPU histories and pinned inputs advance once per round, outside stream capture and replay.
    fn prepare(pass: anytype, ids: []const u32) !void {
        const s = pass.s;
        try s.ready();
        for (s.hostIds(ids.len), ids) |*out, token| out.* = @intCast(token);
        const positions = s.hostPos(ids.len * 4);
        @memset(positions, 0);
        for (pass.lanes, 0..) |lane, i| {
            positions[i * 4] = @intCast(lane.pos);
            try pass.m.ngram.rows(lane.seq.history[0..], ids[i..][0..1], s.hostEmb(ids.len)[i * dims ..][0..dims], s.hostXs(ids.len)[i * (dims / 32) ..][0 .. dims / 32]);
        }
    }

    pub fn run(c: *Cache, key: Key, pass: anytype, ids: []const u32) !void {
        try prepare(pass, ids);
        if (c.key == null or !std.meta.eql(c.key.?, key)) {
            c.clear();
            try cuda.graph.beginCapture(pass.stream, .thread_local);
            pass.prepared = true;
            pass.chunk(ids, 0, null) catch |err| {
                if (cuda.graph.endCapture(pass.stream)) |g| {
                    var discarded = g;
                    discarded.deinit();
                } else |_| {}
                return err;
            };
            var graph = try cuda.graph.endCapture(pass.stream);
            defer graph.deinit();
            c.exec = try graph.instantiate();
            errdefer c.clear();
            try c.exec.?.upload(pass.stream);
            c.key = key;
        }
        try c.exec.?.launchOn(pass.stream);
    }
};

test "decode graph keys invalidate on owner order, attention seams and score tile seams" {
    const Seq = struct { rank_id: u64, fed: usize };
    var first: Seq = .{ .rank_id = 1, .fed = 10 };
    var second: Seq = .{ .rank_id = 2, .fed = 10 };
    const base = try Key.init(@as([]const *Seq, &.{ &first, &second }), 1, 2, 3);
    try std.testing.expect(!std.meta.eql(base, try Key.init(@as([]const *Seq, &.{ &second, &first }), 1, 2, 3)));
    first.fed = 511;
    const before = try Key.init(@as([]const *Seq, &.{&first}), 1, 2, 3);
    first.fed = 512;
    try std.testing.expect(!std.meta.eql(before, try Key.init(@as([]const *Seq, &.{&first}), 1, 2, 3)));
    first.fed = 2048;
    const dense = try Key.init(@as([]const *Seq, &.{&first}), 1, 2, 3);
    first.fed = 2051;
    try std.testing.expect(!std.meta.eql(dense, try Key.init(@as([]const *Seq, &.{&first}), 1, 2, 3)));
    first.fed = 2303;
    const score = try Key.init(@as([]const *Seq, &.{&first}), 1, 2, 3);
    first.fed = 2304;
    try std.testing.expect(!std.meta.eql(score, try Key.init(@as([]const *Seq, &.{&first}), 1, 2, 3)));
}

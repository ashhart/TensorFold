//! A single bounded decode graph, recaptured when sequence owners or buffers, scratch or attention geometry change.
const std = @import("std");
const cuda = @import("cuda");

pub fn enabled() bool {
    // the startup check refused every malformed value already, so a plain default stands here
    return @import("sys").environment.read(bool, "TF_FLASHNEXT_DECODE_GRAPH", false) catch false;
}

pub const Key = struct {
    count: usize,
    owners: [32]u64 = @splat(0),
    /// Each lane's buffers (window_graph.pieces): a drafted window's settle swaps its states, so the key moves too.
    buffers: [32]u64 = @splat(0),
    caps: [32]u64 = @splat(0),
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
            key.buffers[i] = @import("cuda_window_graph.zig").pieces(q);
            key.caps[i] = q.cap;
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
    /// Rounds captured and replayed since start (logged each 64th capture and 1024th replay); a clear keeps them.
    captures: u64 = 0,
    replays: u64 = 0,

    pub fn clear(c: *Cache) void {
        if (c.exec) |*e| e.deinit();
        c.* = .{ .captures = c.captures, .replays = c.replays };
    }

    pub fn run(c: *Cache, key: Key, pass: anytype, ids: []const u32) !void {
        try @import("cuda_window_graph.zig").stageWindow(pass, ids, 0);
        // the key holds no pocket or its row scales (Engine.taught_rows), so a pass with one is refused
        if (pass.m.pocket != null) return error.UnexpectedTensor;
        if (c.key == null or !std.meta.eql(c.key.?, key)) {
            c.clear();
            try cuda.graph.beginCapture(pass.stream, .thread_local);
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
            c.captures += 1;
            if (c.captures % 64 == 1) std.log.info("flashnext: decode round graphs: {d} captured, {d} replayed", .{ c.captures, c.replays });
        } else {
            c.replays += 1;
            if (c.replays % 1024 == 0) std.log.info("flashnext: decode round graphs: {d} captured, {d} replayed", .{ c.captures, c.replays });
        }
        try c.exec.?.launchOn(pass.stream);
    }
};

test {
    _ = @import("cuda_round_graph_test.zig");
}

//! Future token hashes issue bounded file hints synchronously; no detached worker borrows a prompt or shard.
const std = @import("std");
const ple = @import("cuda_ple.zig");
pub fn enabled(value: ?[]const u8) bool {
    return if (value) |v| std.mem.eql(u8, v, "1") else false;
}

pub fn warm(g: anytype, tokens: []const u32, from: usize, to: usize) void {
    if (!g.warm_ahead or g.format != .nvfp4 or from >= to or to > tokens.len or from < g.hash.n - 1) return;
    var history: [ple.max_n]i64 = undefined;
    const hist = history[0 .. g.hash.n - 1];
    for (tokens[from - hist.len .. from], hist) |token, *h| h.* = token;
    var ids: [ple.max_heads]i64 = undefined;
    for (tokens[from..to]) |token| {
        ple.lastIds(g.hash, hist, token, ids[0..g.hash.heads]) catch return;
        for (ids[0..g.hash.heads]) |id| {
            const at = g.hash.locate(id) catch return;
            g.readers[at.shard].hint(g.tables[at.shard], at.local);
        }
        for (0..hist.len - 1) |i| hist[i] = hist[i + 1];
        hist[hist.len - 1] = token;
    }
}

test "PLE read-ahead: future hashes equal a full token history and flags leave no borrowed work" {
    const Recorder = struct {
        var ids: [128]i64 = undefined;
        var count: usize = 0;
        start: i64,
        pub fn hint(r: @This(), _: u8, row: usize) void {
            ids[count] = r.start + @as(i64, @intCast(row));
            count += 1;
        }
    };
    const hash = try ple.Hash.init(.{ .vocab = 100, .ngram = 3, .per = 2, .base = 13, .divisor = 2, .shards = 2, .seed = 1234, .eos = 99, .ple_dim = 64, .ple_index = 0 });
    var g: struct { hash: ple.Hash, warm_ahead: bool = false, format: enum { nvfp4, mlx } = .nvfp4, readers: [2]Recorder, tables: [2]u8 = .{ 0, 0 } } = .{ .hash = hash, .readers = .{ .{ .start = hash.starts[0] }, .{ .start = hash.starts[1] } } };
    const tokens = [_]u32{ 7, 3, 9, 22, 4, 7, 9, 3 };
    const defaults: ple.Ngram = .{ .gpa = std.testing.allocator, .hash = hash, .scale = 1 };
    try std.testing.expect(!defaults.warm_ahead);
    Recorder.count = 0;
    for ([_]?[]const u8{ null, "0", "", "2", "true", "01" }) |value| {
        g.warm_ahead = enabled(value);
        warm(&g, &tokens, 4, tokens.len);
        try std.testing.expectEqual(@as(usize, 0), Recorder.count);
    }
    g.warm_ahead = enabled("1");
    warm(&g, &tokens, 4, tokens.len);
    var hist = [_]i64{ 99, 99 };
    var ids: [ple.max_heads]i64 = undefined;
    var at: usize = 0;
    for (tokens, 0..) |token, i| {
        try ple.lastIds(hash, &hist, token, ids[0..hash.heads]);
        if (i >= 4) {
            try std.testing.expectEqualSlices(i64, ids[0..hash.heads], Recorder.ids[at..][0..hash.heads]);
            at += hash.heads;
        }
        hist = .{ hist[1], token };
    }
    try std.testing.expectEqual(at, Recorder.count);
    const before = Recorder.count;
    g.warm_ahead = false;
    warm(&g, &tokens, 4, tokens.len);
    g.warm_ahead = true;
    g.format = .mlx;
    warm(&g, &tokens, 4, tokens.len);
    g.format = .nvfp4;
    warm(&g, &tokens, 1, tokens.len);
    warm(&g, &tokens, 4, tokens.len + 1);
    warm(&g, &tokens, tokens.len, tokens.len);
    try std.testing.expectEqual(before, Recorder.count);
}

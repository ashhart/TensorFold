//! The n-gram table rows of a window on the host, where the tables live: the HIP engine gathers these rows' bytes.
const std = @import("std");
const config = @import("../config.zig");

pub const heads = 16;
pub const n = 3;
/// Tokens before a window the hashes read.
pub const context = n - 1;

pub const Hasher = struct {
    ref: config.PleRef,
    per: usize,

    pub fn init(c: *const config.Config) !Hasher {
        return .{ .ref = try config.pleRef(c), .per = c.ngram_heads };
    }

    /// Row ids [tokens.len][heads] for `tokens` after `history` (oldest first); eos starts a new segment.
    pub fn ids(h: *const Hasher, history: [context]i64, tokens: []const u32, out: [][heads]u32) void {
        var seq: [context + 1]i64 = undefined; // the row and the context tokens before it, oldest first
        @memcpy(seq[0..context], &history);
        var run: usize = 0; // tokens of the segment before the row, counted up to n - 1
        for (history) |t| run = if (t == h.ref.eos) 0 else @min(run + 1, context);
        for (tokens, out) |tok, *row| {
            seq[context] = tok;
            for (row, 0..) |*id, head| {
                var mixed: u64 = 0;
                for (0..2 + head / h.per) |q| {
                    const t = if (run >= q) seq[context - q] else h.ref.eos;
                    const term = @as(u64, @bitCast(t)) *% @as(u64, @bitCast(h.ref.multipliers[q]));
                    mixed = if (q == 0) term else mixed ^ term;
                }
                id.* = @intCast(@mod(@as(i64, @bitCast(mixed)), h.ref.sizes[head]) + h.ref.offsets[head]);
            }
            run = if (tok == h.ref.eos) 0 else @min(run + 1, context);
            std.mem.copyForwards(i64, seq[0..context], seq[1 .. context + 1]);
        }
    }
};

/// The history after `tokens`: their last `context` tokens, the older history before them.
pub fn after(history: [context]i64, tokens: []const u32) [context]i64 {
    var out = history;
    for (tokens) |t| {
        std.mem.copyForwards(i64, out[0 .. context - 1], out[1..context]);
        out[context - 1] = t;
    }
    return out;
}

test "the hashes of a window with eos inside equal the reference's rows" {
    var c = try config.parse(std.testing.allocator, @embedFile("../fixtures/config.json"));
    defer c.deinit();
    const h = try Hasher.init(&c);
    const tokens = [_]u32{ 5, 248044, 77, 90000, 248044, 248044, 3, 4, 248319, 0, 12 };
    var rows: [tokens.len][heads]u32 = undefined;
    h.ids(.{ 11, 248044 }, &tokens, &rows);
    // the Python embedding's ids() on the same history and tokens
    const first = [_]u32{ 15389869, 18392917, 16611476, 15060967, 3987706, 9663979, 5318554, 11040574, 12308622, 3745007, 15226805 };
    const last = [_]u32{ 315720898, 301167364, 315908561, 316215863, 315670648, 300986548, 306094789, 314073144, 315610311, 302201836, 314647527 };
    for (rows, first, last) |row, a, b| {
        try std.testing.expectEqual(a, row[0]);
        try std.testing.expectEqual(b, row[heads - 1]);
    }
    try std.testing.expectEqualSlices(u32, &.{ 15389869, 39778609, 55713969, 62213332 }, rows[0][0..4]);
    try std.testing.expectEqualSlices(u32, &.{ 160450612, 188182008, 207605318, 236922708 }, rows[3][8..12]);
    // a window cut in two hashes the same rows when the second part starts from the first part's history
    var split: [tokens.len][heads]u32 = undefined;
    h.ids(.{ 11, 248044 }, tokens[0..4], split[0..4]);
    h.ids(after(.{ 11, 248044 }, tokens[0..4]), tokens[4..], split[4..]);
    try std.testing.expectEqualSlices([heads]u32, &rows, &split);
}

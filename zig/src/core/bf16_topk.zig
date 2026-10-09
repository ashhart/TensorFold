//! Exact bounded BF16 top-k contracts, CPU oracle and result decoding.
const std = @import("std");
pub const max_rows = 16;
pub const max_k = 16;
pub const invalid_token = std.math.maxInt(u32);
pub const Params = extern struct { rows: u32, vocab: u32, k: u32, stride: u32 };
pub const Entry = extern struct { token: u32 = invalid_token, score: f32 = 0 };
pub const Row = extern struct { nonfinite: u32 = 0, count: u32 = 0, entries: [max_k]Entry = @splat(.{}) };
pub fn check(p: Params) !void {
    if (p.rows == 0 or p.rows > max_rows or p.k == 0 or p.k > max_k or p.k > p.vocab or p.stride < p.vocab) return error.BadTopKShape;
}
fn before(a: Entry, b: Entry) bool {
    return a.score > b.score or (a.score == b.score and a.token < b.token);
}
pub fn oracle(words: []const u16, k: u32) !Row {
    if (k == 0 or k > max_k or k > words.len) return error.BadTopKShape;
    var out = Row{};
    for (words, 0..) |raw, id| {
        const score: f32 = @bitCast(@as(u32, raw) << 16);
        if (!std.math.isFinite(score)) {
            out.nonfinite |= if (std.math.isNan(score)) @as(u32, 1) else if (score > 0) @as(u32, 2) else @as(u32, 4);
            continue;
        }
        const item = Entry{ .token = @intCast(id), .score = score };
        if (out.count == k and !before(item, out.entries[k - 1])) continue;
        var at = @min(out.count, k - 1);
        while (at > 0 and before(item, out.entries[at - 1])) : (at -= 1) out.entries[at] = out.entries[at - 1];
        out.entries[at] = item;
        out.count = @min(out.count + 1, k);
    }
    if (out.nonfinite != 0) return .{ .nonfinite = out.nonfinite };
    return out;
}
pub fn decode(row: *const Row, p: Params) ![]const Entry {
    try check(p);
    if (row.nonfinite > 7 or row.count != (if (row.nonfinite == 0) p.k else @as(u32, 0))) return error.BadTopKResult;
    for (row.entries[0..row.count], 0..) |entry, i| {
        const bits: u32 = @bitCast(entry.score);
        if (entry.token >= p.vocab or !std.math.isFinite(entry.score) or bits & 0xffff != 0) return error.BadTopKResult;
        if (i > 0 and !before(row.entries[i - 1], entry)) return error.BadTopKResult;
        for (row.entries[0..i]) |previous| if (entry.token == previous.token) return error.BadTopKResult;
    }
    for (row.entries[row.count..]) |entry| if (entry.token != invalid_token or @as(u32, @bitCast(entry.score)) != 0) return error.BadTopKResult;
    if (row.nonfinite != 0) return error.NonfiniteTopK;
    return row.entries[0..row.count];
}
test "top-k orders score and token ties while preserving signed-zero score bits" {
    const p = Params{ .rows = 1, .vocab = 6, .k = 6, .stride = 6 };
    const r = try oracle(&.{ 0x8000, 0, 0x3f80, 0x3f80, 0xbf80, 0x0001 }, 6);
    const entries = try decode(&r, p);
    for (entries, [_]u32{ 2, 3, 5, 0, 1, 4 }) |entry, id| try std.testing.expectEqual(id, entry.token);
    try std.testing.expectEqual(@as(u32, 0x80000000), @as(u32, @bitCast(entries[3].score)));
}
test "nonfinite row rejects every entry and aggregates masks" {
    const r = try oracle(&.{ 0x7fc1, 0x7f80, 0xff80, 0x3f80 }, 2);
    try std.testing.expectEqual(@as(u32, 7), r.nonfinite);
    try std.testing.expectEqual(@as(u32, 0), r.count);
    try std.testing.expectError(error.NonfiniteTopK, decode(&r, .{ .rows = 1, .vocab = 4, .k = 2, .stride = 4 }));
}
test "shape and result guards reject malformed top-k" {
    try std.testing.expectError(error.BadTopKShape, check(.{ .rows = 17, .vocab = 32, .k = 1, .stride = 32 }));
    try std.testing.expectError(error.BadTopKShape, check(.{ .rows = 1, .vocab = 1, .k = 2, .stride = 1 }));
    try std.testing.expectError(error.BadTopKShape, check(.{ .rows = 1, .vocab = 32, .k = 0, .stride = 32 }));
    const p = Params{ .rows = 1, .vocab = 4, .k = 2, .stride = 4 };
    const good = try oracle(&.{ 0x3f80, 0, 0xbf80, 0xbf80 }, 2);
    var rows: [8]Row = @splat(good);
    rows[0].count = 17;
    rows[1].nonfinite = 8;
    rows[2].entries[0].token = 4;
    rows[3].entries[1].token = rows[3].entries[0].token;
    rows[4].entries[0].score = 0.1;
    rows[5].entries[2].token = 0;
    rows[6].entries[1].score = 2;
    rows[7].entries[2].score = -0.0;
    for (&rows) |*row| try std.testing.expectError(error.BadTopKResult, decode(row, p));
}

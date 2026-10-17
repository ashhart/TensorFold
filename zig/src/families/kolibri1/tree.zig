//! The host side of cuda/tree_attention for serial chains: each row's stream, each stream's table row, the parents.
const std = @import("std");

/// The longest chain a stream's window may hold (cuda/tree_attention's path limit).
pub const max_rows = 128;

/// A stream's window: its rows and the keys it has committed before them.
pub const Window = struct { rows: usize, committed: usize };

/// int32 lists in one buffer: row_stream [n], streams [s][4] (first row, rows, committed, 0), parents [n].
pub const Plan = struct {
    flat: []i32,
    rows: usize,
    streams: usize,

    /// Byte offsets of the three lists inside `flat`.
    pub fn streamsAt(p: Plan) usize {
        return p.rows * 4;
    }

    pub fn parentsAt(p: Plan) usize {
        return (p.rows + 4 * p.streams) * 4;
    }
};

/// The plan of `windows`, each a chain (row 0's parent -1, row r's r - 1, global rows); the caller frees `flat`.
pub fn build(gpa: std.mem.Allocator, windows: []const Window) !Plan {
    var n: usize = 0;
    for (windows) |w| {
        if (w.rows < 1 or w.rows > max_rows) return error.WindowWidth;
        n += w.rows;
    }
    const flat = try gpa.alloc(i32, 2 * n + 4 * windows.len);
    var start: usize = 0;
    for (windows, 0..) |w, s| {
        for (0..w.rows) |r| {
            flat[start + r] = @intCast(s);
            flat[n + 4 * windows.len + start + r] = if (r == 0) -1 else @intCast(start + r - 1);
        }
        for ([_]usize{ start, w.rows, w.committed, 0 }, 0..) |v, k| flat[n + 4 * s + k] = @intCast(v);
        start += w.rows;
    }
    return .{ .flat = flat, .rows = n, .streams = windows.len };
}

/// The stream row a one-row step writes before its graph replays: first row 0, one row, `committed` keys.
pub fn stepRow(committed: usize) [4]i32 {
    return .{ 0, 1, @intCast(committed), 0 };
}

test "two chains: rows' streams, the streams' table and global parents" {
    const gpa = std.testing.allocator;
    const p = try build(gpa, &.{ .{ .rows = 2, .committed = 600 }, .{ .rows = 1, .committed = 40 } });
    defer gpa.free(p.flat);
    const want = [_]i32{ 0, 0, 1, 0, 2, 600, 0, 2, 1, 40, 0, -1, 0, -1 };
    try std.testing.expectEqualSlices(i32, &want, p.flat);
    try std.testing.expectEqual(@as(usize, 12), p.streamsAt());
    try std.testing.expectEqual(@as(usize, 44), p.parentsAt());
}

test "a window past the path limit, or an empty one, is refused" {
    const gpa = std.testing.allocator;
    try std.testing.expectError(error.WindowWidth, build(gpa, &.{.{ .rows = max_rows + 1, .committed = 0 }}));
    try std.testing.expectError(error.WindowWidth, build(gpa, &.{.{ .rows = 0, .committed = 0 }}));
    try std.testing.expectEqualSlices(i32, &.{ 0, 1, 77, 0 }, &stepRow(77));
}

//! Spark2.5's facts for core's prompt tail on CUDA: nothing but the head reads a prompt chunk's last layer.
const std = @import("std");
const tail = @import("core").prompt_tail;

/// The tail starts on a 64-row boundary of its chunk, so its rows keep the tiles they have in the whole chunk.
pub const align_rows = 64;

/// The last layer's first row of a chunk (`last`: the prompt's last); null runs every row, `rows` none.
pub fn first(rows: usize, last: bool) ?usize {
    if (rows == 0) return null;
    if (!last) return rows;
    const from = (tail.start(.{ .total = rows, .nseg = 1 }) orelse return null) / align_rows * align_rows;
    return if (from == 0) null else from;
}

test "Spark2.5 prompt tail: an earlier chunk runs no row past its caches, the last a 64-aligned tile with its last row" {
    try std.testing.expectEqual(@as(?usize, 512), first(512, false));
    try std.testing.expectEqual(@as(?usize, 384), first(512, true)); // 512 - 128, already aligned
    try std.testing.expectEqual(@as(?usize, 320), first(500, true)); // 372 rounds down to 320: 180 rows run
    try std.testing.expectEqual(@as(?usize, 64), first(192, true));
    try std.testing.expectEqual(@as(?usize, null), first(191, true)); // 63 rounds down to 0
    try std.testing.expectEqual(@as(?usize, null), first(128, true)); // one tile: every row
    try std.testing.expectEqual(@as(?usize, null), first(0, false));
    for ([_]usize{ 129, 200, 300, 511, 512 }) |rows| {
        const s = first(rows, true) orelse continue;
        try std.testing.expect(s % align_rows == 0 and rows - s >= tail.rows and rows - s < tail.rows + align_rows);
    }
}

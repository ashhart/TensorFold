//! MiniMax-M3's facts for core's prompt tail on CUDA: the prompt rows its last layer runs past its cache writes.
const std = @import("std");
const tail = @import("core").prompt_tail;

/// The tail starts on a 64-row boundary of its chunk, so its rows keep the tiles they have in the whole chunk.
pub const align_rows = 64;

/// What reads a chunk's last layer, as the engine knows it on every rank alike.
pub const Chunk = struct {
    rows: usize, // the chunk's rows
    start: usize, // its first row's position in the prompt
    len: usize, // the prompt's tokens
    /// DSpark's reach (its window and a window's rows): it absorbs the prompt's last rows' layer-59 taps; null: none.
    draft: ?usize = null,
    /// A reference check's per-layer dumps read every row.
    watched: bool = false,
};

/// The last layer's first row: rows before it stop at their cache writes. Null runs every row; `rows` runs none.
pub fn first(c: Chunk) ?usize {
    if (c.watched or c.rows == 0) return null;
    const last = c.start + c.rows == c.len;
    // the prompt's last chunk keeps its last row (the first token's logits); an earlier chunk's head is unread
    var from: usize = if (!last) c.rows else (tail.start(.{ .total = c.rows, .nseg = 1 }) orelse return null) / align_rows * align_rows;
    if (c.draft) |reach| { // DSpark's absorb reads the taps of the rows from here: they run the layer too
        const read = @max(c.start, c.len -| reach);
        if (read < c.start + c.rows) from = @min(from, (read - c.start) / align_rows * align_rows);
    }
    return if (from == 0) null else from;
}

test "MiniMax-M3 prompt tail: an earlier chunk runs no row past its caches, the last keeps a 64-aligned tile" {
    // a 5,000-token prompt in 2,048-row chunks: 0, 2048 and the last 904 rows from 4096
    try std.testing.expectEqual(@as(?usize, 2048), first(.{ .rows = 2048, .start = 0, .len = 5000 }));
    try std.testing.expectEqual(@as(?usize, 2048), first(.{ .rows = 2048, .start = 2048, .len = 5000 }));
    // the last: 904 - 128 = 776, down to 768, so rows 768..904 (136 rows, the last among them) run the layer
    try std.testing.expectEqual(@as(?usize, 768), first(.{ .rows = 904, .start = 4096, .len = 5000 }));
    try std.testing.expectEqual(@as(?usize, 1920), first(.{ .rows = 2048, .start = 2048, .len = 4096 }));
    // a last chunk of one tile or less runs every row, as does a watched forward
    try std.testing.expectEqual(@as(?usize, null), first(.{ .rows = 128, .start = 0, .len = 128 }));
    try std.testing.expectEqual(@as(?usize, null), first(.{ .rows = 100, .start = 2048, .len = 2148 }));
    try std.testing.expectEqual(@as(?usize, null), first(.{ .rows = 2048, .start = 0, .len = 5000, .watched = true }));
    try std.testing.expectEqual(@as(?usize, 64), first(.{ .rows = 192, .start = 0, .len = 192 })); // 192 - 128 = 64
    try std.testing.expectEqual(@as(?usize, null), first(.{ .rows = 191, .start = 0, .len = 191 })); // 63 rounds down to 0
}

test "MiniMax-M3 prompt tail: with DSpark, the rows its absorb reads run the last layer in every chunk" {
    const reach = 4096 + 16; // a 4,096-key window and a 16-row window past it
    // a 10,000-token prompt: the drafter reads rows from 10,000 - 4,112 = 5,888
    try std.testing.expectEqual(@as(?usize, 2048), first(.{ .rows = 2048, .start = 2048, .len = 10000, .draft = reach })); // none read
    try std.testing.expectEqual(@as(?usize, 1792), first(.{ .rows = 2048, .start = 4096, .len = 10000, .draft = reach })); // 5,888, aligned
    try std.testing.expectEqual(@as(?usize, 1728), first(.{ .rows = 2048, .start = 4096, .len = 9950, .draft = reach })); // 5,838 -> 1,742 -> 1,728
    try std.testing.expectEqual(@as(?usize, null), first(.{ .rows = 2048, .start = 6144, .len = 10000, .draft = reach })); // all read
    try std.testing.expectEqual(@as(?usize, null), first(.{ .rows = 1808, .start = 8192, .len = 10000, .draft = reach }));
    // a short prompt: the drafter reads it whole, so no chunk skips a row
    try std.testing.expectEqual(@as(?usize, null), first(.{ .rows = 2048, .start = 0, .len = 3000, .draft = reach }));
    try std.testing.expectEqual(@as(?usize, null), first(.{ .rows = 952, .start = 2048, .len = 3000, .draft = reach }));
}

//! A prompt chunk keeps one row's logits: past the last layer's keys and values, only its tile feeds the head.
const std = @import("std");

/// Rows the tail keeps: one prompt GEMM row tile, so q, o and the MLP stay on the prompt kernels (rows > 16).
pub const rows: usize = 128;

/// What decides the tail, as the family reports it.
pub const Facts = struct {
    /// The chunk's rows, and the parts it is cut into.
    total: usize,
    nseg: usize,
    /// Something besides the head reads the last layer's other rows: hidden readouts, hooks, taps, images.
    readers: bool = false,
    /// The family's last layer can run its q, attention, o and MLP over a row range.
    fits: bool = true,
};

/// The tail's first row, or null when the chunk runs its last layer over every row.
pub fn start(f: Facts) ?usize {
    if (f.total <= rows or f.nseg != 1) return null;
    if (f.readers or !f.fits) return null;
    return f.total - rows;
}

/// What reads a prompt chunk's last layers past their state writes.
pub const Reads = struct {
    /// The chunk's logits are wanted: the prompt's last chunk.
    logits: bool,
    /// Something else reads its rows: taps, a head that absorbs them, prompt logprobs, dumps, lessons.
    readers: bool,
};

/// True when no row of the chunk is read: its last layers stop at their state writes.
pub fn unread(r: Reads) bool {
    return !r.logits and !r.readers;
}

test "prompt tail: a chunk stops at its state writes only when nothing reads it" {
    try std.testing.expect(unread(.{ .logits = false, .readers = false }));
    try std.testing.expect(!unread(.{ .logits = true, .readers = false }));
    try std.testing.expect(!unread(.{ .logits = false, .readers = true }));
}

test "prompt tail: a plain chunk past one tile ends on the tile that holds its kept row" {
    for ([_]usize{ 129, 512, 2048, 4096, 4097 }) |total| {
        const s = start(.{ .total = total, .nseg = 1 }).?;
        try std.testing.expectEqual(total - 1, s + rows - 1);
    }
    try std.testing.expectEqual(@as(?usize, null), start(.{ .total = rows, .nseg = 1 }));
}

test "prompt tail: readers, a layer that does not fit and a cut chunk each refuse it" {
    const open: Facts = .{ .total = 4096, .nseg = 1 };
    try std.testing.expect(start(open) != null);
    var f = open;
    f.readers = true;
    try std.testing.expectEqual(@as(?usize, null), start(f));
    f = open;
    f.fits = false;
    try std.testing.expectEqual(@as(?usize, null), start(f));
    f = open;
    f.nseg = 2;
    try std.testing.expectEqual(@as(?usize, null), start(f));
}

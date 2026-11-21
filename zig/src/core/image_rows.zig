//! Image placeholder rows of a prompt: which rows of a pass's image buffer each prompt row takes.
const std = @import("std");

/// An image's placeholder rows [at, at + n) of the prompt take buffer rows [row, row + n); `image` is its index.
pub const Span = struct { at: u32, n: u32, row: u32, image: u32 = 0 };

/// What a span gives prompt rows [lo, hi): `n` rows from prompt row `at` on, taken from buffer row `row` on.
pub const Cut = struct { at: usize, row: usize, n: usize };

/// The part of `sp` prompt rows [lo, hi) cross, or null for none.
pub fn cut(sp: Span, lo: usize, hi: usize) ?Cut {
    const from = @max(lo, sp.at);
    const to = @min(hi, @as(usize, sp.at) + sp.n);
    return if (from < to) .{ .at = from, .row = sp.row + (from - sp.at), .n = to - from } else null; // offset first
}

/// Rows the spans hold in all.
pub fn total(spans: []const Span) u32 {
    var n: u32 = 0;
    for (spans) |s| n += s.n;
    return n;
}

test "a cut near the top of the u32 range takes its offset before adding the buffer row" {
    const sp: Span = .{ .at = 0xffffff00, .n = 128, .row = 256 };
    try std.testing.expectEqual(Cut{ .at = 0xffffff00, .row = 256, .n = 0x40 }, cut(sp, 0xffffff00, 0xffffff40).?);
    try std.testing.expectEqual(Cut{ .at = 0xffffff40, .row = 320, .n = 0x40 }, cut(sp, 0xffffff40, 0xffffffff).?);
    try std.testing.expect(cut(sp, 0xffffff80, 0xffffffff) == null);
    const top: Span = .{ .at = 0xffffff80, .n = 0x100, .row = 0xffffff00 }; // ends past 2^32 on both sides
    try std.testing.expectEqual(Cut{ .at = 0xffffffc0, .row = 0xffffff40, .n = 0x40 }, cut(top, 0xffffffc0, 0x100000000).?);
}

test "a chunk takes the rows it crosses: a span cut at both ends, and none past it" {
    const first: Span = .{ .at = 5, .n = 256, .row = 0 };
    const second: Span = .{ .at = 300, .n = 300, .row = 256, .image = 1 };
    try std.testing.expectEqual(Cut{ .at = 200, .row = 195, .n = 61 }, cut(first, 200, 400).?);
    try std.testing.expectEqual(Cut{ .at = 300, .row = 256, .n = 100 }, cut(second, 200, 400).?);
    try std.testing.expectEqual(Cut{ .at = 5, .row = 0, .n = 256 }, cut(first, 0, 1000).?);
    try std.testing.expect(cut(first, 261, 300) == null and cut(second, 600, 700) == null);
    try std.testing.expectEqual(@as(u32, 556), total(&.{ first, second }));
}

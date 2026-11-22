//! TF_Q27_DFLASH_ABSORB_BATCH: which streams' new taps share one absorb pass, and where each stream's rows sit in it.
const std = @import("std");

/// The most rows one stacked absorb takes; a larger round keeps the per-stream absorb.
pub const max_rows = 128;

/// A stream's rows [first, first + rows) of the stacked pass.
pub const Span = struct { stream: u32, first: u32, rows: u32 };

/// Spans in stream order for `rows` (0: the stream absorbs nothing); null unless 2+ streams absorb within `max_rows`.
pub fn plan(rows: []const u32, out: []Span) ?[]const Span {
    if (out.len < rows.len) return null;
    var n: usize = 0;
    var total: u32 = 0;
    for (rows, 0..) |count, stream| {
        if (count == 0) continue;
        if (count > max_rows - total) return null;
        out[n] = .{ .stream = @intCast(stream), .first = total, .rows = count };
        total += count;
        n += 1;
    }
    return if (n < 2) null else out[0..n];
}

test "qwen27 window: absorb batch plan keeps stream order and uneven rows" {
    var out: [4]Span = undefined;
    const spans = plan(&.{ 3, 0, 1, 5 }, &out).?;
    try std.testing.expectEqualSlices(Span, &.{ .{ .stream = 0, .first = 0, .rows = 3 }, .{ .stream = 2, .first = 3, .rows = 1 }, .{ .stream = 3, .first = 4, .rows = 5 } }, spans);
}

test "qwen27 window: absorb batch plan declines one stream and more than 128 rows" {
    var out: [4]Span = undefined;
    try std.testing.expectEqual(@as(?[]const Span, null), plan(&.{ 0, 7, 0, 0 }, &out));
    try std.testing.expectEqual(@as(?[]const Span, null), plan(&.{ 100, 29 }, &out));
    try std.testing.expectEqual(@as(usize, 2), plan(&.{ 100, 28 }, &out).?.len);
}

//! Where a prompt pass cuts its chunks: a row's bits are the same wherever the cuts fall, so states keep at any row.
const std = @import("std");

/// The end of the chunk starting at `at`: `step` rows on, the prompt's end, or the first of `cuts` before them.
pub fn end(at: usize, len: usize, step: usize, cuts: []const u32) usize {
    var e = @min(len, at + step);
    for (cuts) |m| if (m > at and m < e) {
        e = m;
    };
    return e;
}

test "chunks run `step` rows from where a pass starts, and a cut ends one" {
    const step = 2048;
    var ends: [4]usize = undefined;
    var at: usize = 0;
    for (&ends) |*c| {
        c.* = end(at, 5000, step, &.{ 4990, 1500 });
        at = c.*;
    }
    try std.testing.expectEqualSlices(usize, &.{ 1500, 3548, 4990, 5000 }, &ends);
    try std.testing.expectEqual(@as(usize, 4148), end(2100, 5000, step, &.{})); // a resumed pass: 2048 rows from its start
    try std.testing.expectEqual(@as(usize, 2048), end(0, 5000, step, &.{2048})); // a cut at a chunk's end adds none
    try std.testing.expectEqual(@as(usize, 1000), end(0, 1000, step, &.{ 0, 1000 })); // cuts at the start or end cut nothing
}

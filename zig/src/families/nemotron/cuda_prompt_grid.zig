//! Prompt chunks: the zero-anchored grid, also cut at a pass's keeps. A row's bits do not depend on the chunk cuts.
const std = @import("std");

/// The end of the chunk starting at `at`: the next multiple of `step`, or a keep before it, never past `len`.
pub fn end(at: usize, len: usize, step: usize, keeps: []const u32) usize {
    var e = @min(len, at - at % step + step);
    for (keeps) |k| if (k > at and k < e) {
        e = k;
    };
    return e;
}

/// Where the last chunk of a pass from `from` to `len` starts.
pub fn lastStart(from: usize, len: usize, step: usize, keeps: []const u32) usize {
    var s = from;
    while (true) {
        const e = end(s, len, step, keeps);
        if (e == len) return s;
        s = e;
    }
}

/// A kept state a pass may start from: a position inside the prompt.
pub fn resumable(from: usize, len: usize) bool {
    return from > 0 and from < len;
}

test "a pass cuts the grid and its keeps, and resumes anywhere inside the prompt" {
    const step = 2048;
    var cuts: [3]usize = undefined;
    var at: usize = 0;
    for (&cuts) |*c| {
        c.* = end(at, 5000, step, &.{});
        at = c.*;
    }
    try std.testing.expectEqualSlices(usize, &.{ 2048, 4096, 5000 }, &cuts);
    try std.testing.expectEqual(@as(usize, 2048), end(100, 5000, step, &.{})); // an off-grid start returns to the grid
    try std.testing.expectEqual(@as(usize, 3000), end(2048, 5000, step, &.{ 100, 3000, 3500 })); // the first keep inside
    try std.testing.expectEqual(@as(usize, 1000), end(0, 1000, step, &.{1000})); // a keep at the end is the end
    try std.testing.expectEqual(@as(usize, 4096), lastStart(0, 5000, step, &.{}));
    try std.testing.expectEqual(@as(usize, 4500), lastStart(3000, 5000, step, &.{4500}));
    try std.testing.expectEqual(@as(usize, 4096), lastStart(4096, 4097, step, &.{}));
    try std.testing.expectEqual(@as(usize, 300), lastStart(300, 1000, step, &.{}));
    try std.testing.expect(resumable(2048, 5000) and resumable(3001, 5000));
    try std.testing.expect(!resumable(0, 5000) and !resumable(4096, 4096));
}

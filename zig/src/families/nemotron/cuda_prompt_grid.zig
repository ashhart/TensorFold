//! Prompt chunks on the zero-anchored grid: a pass that resumes or keeps states cuts where a cache-off pass does.
const std = @import("std");

/// The end of the chunk starting at `at`: the next multiple of `step`, never past `len`.
pub fn end(at: usize, len: usize, step: usize) usize {
    return @min(len, at - at % step + step);
}

/// A kept state a pass may start from: a grid point inside the prompt.
pub fn resumable(from: usize, len: usize, step: usize) bool {
    return from > 0 and from < len and from % step == 0;
}

test "a resumed or kept pass cuts the cache-off grid, and only grid points resume" {
    const step = 2048;
    var cuts: [3]usize = undefined;
    var at: usize = 0;
    for (&cuts) |*c| {
        c.* = end(at, 5000, step);
        at = c.*;
    }
    try std.testing.expectEqualSlices(usize, &.{ 2048, 4096, 5000 }, &cuts);
    try std.testing.expectEqual(@as(usize, 4096), end(2048, 5000, step)); // resumed at 2048: the same later cuts
    try std.testing.expectEqual(@as(usize, 2048), end(100, 5000, step)); // an off-grid start returns to the grid
    try std.testing.expectEqual(@as(usize, 1000), end(0, 1000, step));
    try std.testing.expect(resumable(2048, 5000, step) and resumable(4096, 5000, step));
    try std.testing.expect(!resumable(0, 5000, step) and !resumable(3000, 5000, step) and !resumable(4096, 4096, step));
}

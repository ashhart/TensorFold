//! Prompt chunks: the zero-anchored grid, also cut at a pass's keeps. A row's bits do not depend on the chunk cuts.
const std = @import("std");

/// A chunk: where it ends, and the one keep inside it whose state it stages without a cut (0: none).
pub const Cut = struct { end: usize, staged: usize = 0 };

/// The chunk from `at` to the next multiple of `step` (≤ `len`): it stages a lone keep inside, else ends at the first.
pub fn cut(at: usize, len: usize, step: usize, keeps: []const u32) Cut {
    const e = @min(len, at - at % step + step);
    var first: usize = 0;
    var inside: usize = 0;
    for (keeps) |k| if (k > at and k < e) {
        if (inside == 0) first = k;
        inside += 1;
    };
    return switch (inside) {
        0 => .{ .end = e },
        1 => .{ .end = e, .staged = first },
        else => .{ .end = first },
    };
}

/// Where the last chunk of a pass from `from` to `len` starts.
pub fn lastStart(from: usize, len: usize, step: usize, keeps: []const u32) usize {
    var s = from;
    while (true) {
        const e = cut(s, len, step, keeps).end;
        if (e == len) return s;
        s = e;
    }
}

/// A kept state a pass may start from: a position inside the prompt.
pub fn resumable(from: usize, len: usize) bool {
    return from > 0 and from < len;
}

test "a chunk stages a lone keep inside it, cuts at the first of several, and resumes anywhere inside the prompt" {
    const step = 2048;
    var cuts: [3]usize = undefined;
    var at: usize = 0;
    for (&cuts) |*c| {
        c.* = cut(at, 5000, step, &.{}).end;
        at = c.*;
    }
    try std.testing.expectEqualSlices(usize, &.{ 2048, 4096, 5000 }, &cuts);
    try std.testing.expectEqual(@as(usize, 2048), cut(100, 5000, step, &.{}).end); // an off-grid start returns to the grid
    try std.testing.expectEqual(Cut{ .end = 3000 }, cut(2048, 5000, step, &.{ 100, 3000, 3500 })); // the first of two
    try std.testing.expectEqual(Cut{ .end = 4096, .staged = 3500 }, cut(3000, 5000, step, &.{ 100, 3000, 3500 }));
    try std.testing.expectEqual(@as(usize, 4200), lastStart(3000, 5000, step, &.{ 4200, 4500 })); // 4500 staged
    try std.testing.expectEqual(@as(usize, 4096), lastStart(0, 5000, step, &.{4500})); // staged: no cut
    try std.testing.expectEqual(Cut{ .end = 1000 }, cut(0, 1000, step, &.{1000})); // a keep at the end is the end
    try std.testing.expectEqual(@as(usize, 4096), lastStart(4096, 4097, step, &.{}));
    try std.testing.expectEqual(@as(usize, 300), lastStart(300, 1000, step, &.{}));
    try std.testing.expect(resumable(2048, 5000) and resumable(3001, 5000));
    try std.testing.expect(!resumable(0, 5000) and !resumable(4096, 4096));
}

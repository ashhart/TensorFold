//! A MiniMax-M3 drafter's hold as one pass over several streams' blocks: each block stacked at its row offset.
const std = @import("std");

/// Blocks a pass takes: the default 8 streams' 64 rows, one 64-row tile of the tensor product; more take more passes.
pub const max_blocks = 8;

/// Items the next pass takes of `left`.
pub fn take(left: usize) usize {
    return @min(left, max_blocks);
}

/// The first row of the pass's block `slot`: per-row work runs over every block at once, the rest at this offset.
pub fn row(slot: usize, block: usize) usize {
    return slot * block;
}

/// Each block's ids: its pending token, then `fill` (the mask or noise token) through its block's rows.
pub fn fillIds(ids: []u32, pendings: []const u32, block: usize, fill: u32) void {
    for (pendings, 0..) |p, s| {
        const at = row(s, block);
        ids[at] = p;
        @memset(ids[at + 1 ..][0 .. block - 1], fill);
    }
}

/// A block's sequence slot: 16 words each, its pending token first, the drafts after (the walk's or the steps' output).
pub fn seqAt(slot: usize) usize {
    return slot * 16;
}

test "MiniMax-M3 hold batch: two streams' ids are the two single holds' ids, block after block" {
    var two: [2 * 8]u32 = undefined;
    fillIds(&two, &.{ 11, 22 }, 8, 7);
    var a: [8]u32 = undefined;
    var b: [8]u32 = undefined;
    fillIds(&a, &.{11}, 8, 7);
    fillIds(&b, &.{22}, 8, 7);
    try std.testing.expectEqualSlices(u32, &a, two[0..8]);
    try std.testing.expectEqualSlices(u32, &b, two[8..16]);
    // a shorter DSpark block stacks the same way
    var three: [3 * 5]u32 = undefined;
    fillIds(&three, &.{ 1, 2, 3 }, 5, 9);
    try std.testing.expectEqualSlices(u32, &.{ 1, 9, 9, 9, 9, 2, 9, 9, 9, 9, 3, 9, 9, 9, 9 }, &three);
}

test "MiniMax-M3 hold batch: blocks and sequences never overlap, and passes take eight streams at a time" {
    for (0..max_blocks - 1) |s| {
        try std.testing.expectEqual(row(s, 8) + 8, row(s + 1, 8));
        try std.testing.expectEqual(seqAt(s) + 16, seqAt(s + 1));
    }
    try std.testing.expectEqual(@as(usize, 64), row(max_blocks, 8));
    var left: usize = 17;
    var passes: [3]usize = undefined;
    var n: usize = 0;
    while (left > 0) : (n += 1) {
        passes[n] = take(left);
        left -= passes[n];
    }
    try std.testing.expectEqualSlices(usize, &.{ 8, 8, 1 }, passes[0..n]);
}
